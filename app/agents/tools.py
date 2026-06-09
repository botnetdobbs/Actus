import asyncio
import contextvars
import inspect
import time
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel
import structlog

log = structlog.get_logger()

# Tracks active agents to detect cycles and cap nesting depth
MAX_INVOKE_DEPTH = 5
MAX_PARALLEL_AGENTS = 10
_invoke_stack: contextvars.ContextVar[list[str]] = contextvars.ContextVar(
    "invoke_stack", default=[]
)


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    output: Any
    error: str = ""
    duration_ms: float = 0.0


_tools: dict[str, Callable] = {}
_tool_descriptions: dict[str, str] = {}
_tool_schemas: dict[str, dict] = {}


def _clean_annotation(annotation) -> str:
    if annotation is inspect.Parameter.empty:
        return "any"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    if hasattr(annotation, "__origin__"):
        return str(annotation)\
            .replace("typing.", "")\
            .replace("builtins.", "")
    return str(annotation).replace("typing.", "").replace("builtins.", "")


def tool(name: str, description: str, *, params: dict[str, str] | None = None):
    """Register a callable as an agent tool.

    params: optional mapping of parameter name → description, included in
    OpenAI function-calling schema so the model understands each argument's purpose.
    """
    def decorator(fn: Callable):
        _tools[name] = fn
        _tool_descriptions[name] = description

        try:
            hints = get_type_hints(fn)
        except Exception:
            hints = {}

        params_dict: dict[str, dict] = {}
        for param_name, param in inspect.signature(fn).parameters.items():
            type_hint = hints.get(param_name, param.annotation)
            entry: dict = {
                "type": _clean_annotation(type_hint),
                "required": param.default is inspect.Parameter.empty,
            }
            if param.default is not inspect.Parameter.empty:
                entry["default"] = param.default
            if params and param_name in params:
                entry["description"] = params[param_name]
            params_dict[param_name] = entry

        _tool_schemas[name] = {
            "name": name,
            "description": description,
            "parameters": params_dict,
        }
        return fn
    return decorator


async def run_tool(name: str, timeout_seconds: float = 30.0, **kwargs) -> ToolResult:
    if name not in _tools:
        return ToolResult(tool_name=name, success=False,
                          output=None, error=f"Unknown tool: {name}")
    fn = _tools[name]
    start = time.monotonic()
    try:
        if inspect.iscoroutinefunction(fn):
            output = await asyncio.wait_for(fn(**kwargs), timeout=timeout_seconds)
        else:
            loop = asyncio.get_running_loop()
            output = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: fn(**kwargs)),
                timeout=timeout_seconds,
            )
        duration_ms = (time.monotonic() - start) * 1000
        log.info("tool_executed", tool=name,
                 duration_ms=round(duration_ms, 1), success=True)
        return ToolResult(tool_name=name, success=True,
                          output=output, duration_ms=duration_ms)
    except asyncio.TimeoutError:
        log.error("tool_timeout", tool=name, timeout=timeout_seconds)
        return ToolResult(tool_name=name, success=False,
                          output=None, error=f"Timeout after {timeout_seconds}s")
    except Exception as e:
        log.error("tool_failed", tool=name, error=str(e), exc_info=True)
        return ToolResult(tool_name=name, success=False,
                          output=None, error=str(e))


def list_tools() -> list[dict]:
    return list(_tool_schemas.values())


_PYTHON_TO_JSON_TYPE: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


def to_openai_tools(schemas: list[dict]) -> list[dict]:
    """Convert bespoke tool schemas to the OpenAI/LiteLLM function-calling format."""
    result = []
    for schema in schemas:
        params = schema.get("parameters", {})
        properties: dict = {}
        required: list[str] = []
        for param_name, meta in params.items():
            raw_type = meta.get("type", "any")
            json_type = _PYTHON_TO_JSON_TYPE.get(raw_type, "string")
            prop: dict = {"type": json_type}
            if "description" in meta:
                prop["description"] = meta["description"]
            properties[param_name] = prop
            if meta.get("required", False):
                required.append(param_name)
        result.append({
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return result


def _supports_native_tools(model: str) -> bool:
    """Return True for models with reliable native function calling (non-Ollama)."""
    return not model.startswith("ollama/")


# ── Built-in tools ────────────────────────────────────────────────────────────

@tool("semantic_search",
      "Search ontology objects by semantic similarity to a natural-language query")
def semantic_search(query: str, type_name: str = "", top_k: int = 5) -> list[dict]:
    from app.rag.retriever import retrieve
    return retrieve(query, type_name=type_name or None, top_k=top_k)


@tool("invoke_agent",
      "Run another registered agent by ID and return its result. "
      "Use for sequential delegation to a specialist agent.")
async def invoke_agent(agent_id: str, query: str = "") -> dict:
    from app.agents.builder import get_agent
    from app.agents.orchestrator import AGENT_TOTAL_TIMEOUT, run_agent

    current_stack = _invoke_stack.get()

    if len(current_stack) >= MAX_INVOKE_DEPTH:
        return {
            "status": "error",
            "error": f"Max agent invocation depth ({MAX_INVOKE_DEPTH}) exceeded",
            "result": None, "confidence": None,
        }
    if agent_id in current_stack:
        return {
            "status": "error",
            "error": f"Circular invocation detected: '{agent_id}' is already running",
            "result": None, "confidence": None,
        }
    try:
        config = get_agent(agent_id)
    except KeyError:
        return {"status": "error", "error": f"Unknown agent: '{agent_id}'",
                "result": None, "confidence": None}

    try:
        full = await asyncio.wait_for(
            run_agent(config, extra_context={"query": query} if query else None),
            timeout=AGENT_TOTAL_TIMEOUT,
        )
        return {
            "status": full.get("status"),
            "result": full.get("result"),
            "confidence": full.get("confidence"),
            "error": full.get("error"),
        }
    except asyncio.TimeoutError:
        log.error("invoke_agent_timeout", agent_id=agent_id)
        return {"status": "timeout", "error": "Agent run timed out", "result": None, "confidence": None}
    except Exception as e:
        log.error("invoke_agent_failed", agent_id=agent_id, error=str(e))
        return {"status": "error", "error": str(e), "result": None, "confidence": None}


@tool("invoke_agents_parallel",
      "Run multiple registered agents in parallel and return all results. "
      "Use when agents are independent and can run simultaneously. "
      "agent_ids must be a list of registered agent ID strings.")
async def invoke_agents_parallel(agent_ids: list[str], query: str = "") -> list[dict]:
    if not agent_ids:
        return []

    if len(agent_ids) > MAX_PARALLEL_AGENTS:
        return [{
            "status": "error",
            "result": None,
            "confidence": None,
            "error": (f"Batch rejected: {len(agent_ids)} agents requested, "
                      f"max is {MAX_PARALLEL_AGENTS}. Split into smaller batches."),
        }]

    current_stack = _invoke_stack.get()

    if len(current_stack) >= MAX_INVOKE_DEPTH:
        return [{"status": "error", "error": f"Max invocation depth ({MAX_INVOKE_DEPTH}) exceeded",
                 "result": None, "confidence": None}]

    async def _run_one(aid: str) -> dict:
        from app.agents.builder import get_agent
        from app.agents.orchestrator import AGENT_TOTAL_TIMEOUT, run_agent

        if aid in current_stack:
            return {
                "status": "error",
                "error": f"Circular invocation: '{aid}' is already running",
                "result": None, "confidence": None,
            }
        try:
            config = get_agent(aid)
        except KeyError:
            return {"status": "error", "error": f"Unknown agent: '{aid}'",
                    "result": None, "confidence": None}

        try:
            full = await asyncio.wait_for(
                run_agent(config, extra_context={"query": query} if query else None),
                timeout=AGENT_TOTAL_TIMEOUT,
            )
            return {
                "status": full.get("status"),
                "result": full.get("result"),
                "confidence": full.get("confidence"),
                "error": full.get("error"),
            }
        except asyncio.TimeoutError:
            log.error("invoke_agent_timeout", agent_id=aid)
            return {"status": "timeout", "error": "Agent run timed out",
                    "result": None, "confidence": None}
        except Exception as e:
            log.error("invoke_agent_failed", agent_id=aid, error=str(e))
            return {"status": "error", "error": str(e), "result": None, "confidence": None}

    results = await asyncio.gather(
        *[_run_one(aid) for aid in agent_ids],
        return_exceptions=True,
    )
    return [
        r if isinstance(r, dict) else {"status": "error", "error": str(r),
                                        "result": None, "confidence": None}
        for r in results
    ]
