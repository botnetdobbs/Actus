import asyncio
import uuid
import json
import structlog
from app.agents.builder import AgentConfig
from app.agents.tools import run_tool, _tool_schemas
from app.context.models import AgentContext
from app.context.store import save_context
from app.config import get_settings
from app.llm.pii import scrub_pii
from app.llm.router import call_llm_with_retry
from app.observability.metrics import active_agent_runs, agent_runs_total

log = structlog.get_logger()

_settings = get_settings()

AGENT_TOTAL_TIMEOUT = 300  # 5 minutes max for an entire agent run
LLM_PER_CALL_TIMEOUT = 60  # 60 seconds per LLM call

TOOL_CALL_PROMPT = """
Available tools (respond ONLY with JSON):
{tools}
To call a tool:
{{"tool": "name", "args": {{"param": "value"}}}}
To finish:
{{"done": true, "result": "your final answer"}}
"""


def extract_json(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        content = "\n".join(inner).strip()
    return content


async def run_agent(config: AgentConfig, extra_context: dict | None = None) -> dict:
    run_id = str(uuid.uuid4())
    bound_log = log.bind(agent_id=config.id, run_id=run_id)
    bound_log.info("agent_run_start")
    active_agent_runs.inc()
    status = "unknown"
    try:
        result = await _run_agent_inner(config, run_id, bound_log, extra_context)
        status = result.get("status", "unknown")
        return result
    except Exception:
        status = "error"
        raise
    finally:
        active_agent_runs.dec()
        agent_runs_total.labels(agent_id=config.id, status=status).inc()


async def _run_agent_inner(
    config: AgentConfig,
    run_id: str,
    bound_log,
    extra_context: dict | None,
) -> dict:
    tools_desc = json.dumps(list(_tool_schemas.values()), indent=2)
    messages = [{"role": "system", "content": config.system_prompt + TOOL_CALL_PROMPT.format(tools=tools_desc)}]

    if extra_context:
        context_str = json.dumps(extra_context)
        clean_context, pii_found = scrub_pii(context_str)
        if pii_found:
            bound_log.warning("agent_context_pii_scrubbed")
        messages.append({"role": "user", "content": f"Context: {clean_context}"})

    context = AgentContext(agent_id=config.id, run_id=run_id)
    tool_calls_log: list[dict] = []
    iterations_used = 0
    total_tokens = 0

    api_base = _settings.ollama_base_url if "ollama" in config.model else None

    for iteration in range(config.max_iterations):
        iterations_used = iteration + 1
        bound_log.info("agent_iteration", iteration=iteration)

        try:
            response = await call_llm_with_retry(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=1024,
                timeout=LLM_PER_CALL_TIMEOUT,
                api_base=api_base,
            )
        except Exception as e:
            bound_log.error("llm_call_failed", iteration=iteration, error=str(e))
            save_context(context)
            return {"run_id": run_id, "result": None,
                    "iterations": iterations_used, "status": "error", "error": str(e)}

        if response.usage:
            total_tokens += response.usage.total_tokens
            if total_tokens > config.token_budget:
                bound_log.warning("agent_token_budget_exceeded",
                                  total_tokens=total_tokens, budget=config.token_budget)
                save_context(context)
                return {"run_id": run_id, "result": "Token budget exceeded",
                        "iterations": iterations_used, "total_tokens": total_tokens, "status": "incomplete"}

        raw_content = response.choices[0].message.content
        content = (raw_content or "").strip()
        messages.append({"role": "assistant", "content": content})

        try:
            action = json.loads(extract_json(content))
        except json.JSONDecodeError:
            bound_log.warning("llm_non_json_response", iteration=iteration)
            save_context(context)
            return {"run_id": run_id, "result": content,
                    "iterations": iterations_used, "status": "completed"}

        if action.get("done"):
            bound_log.info("agent_run_complete", iterations=iterations_used, total_tokens=total_tokens)
            save_context(context)
            return {"run_id": run_id, "result": action.get("result"),
                    "iterations": iterations_used, "total_tokens": total_tokens, "status": "completed"}

        if "tool" in action:
            tool_name = action["tool"]
            if tool_name not in config.tools:
                bound_log.warning("agent_used_unauthorised_tool", tool=tool_name)
                messages.append({"role": "user",
                                  "content": f"Tool '{tool_name}' is not available to you. "
                                             f"Available tools: {config.tools}"})
                tool_calls_log.append({"tool": tool_name, "success": False, "detail": "unauthorised"})
                continue

            result = await run_tool(tool_name, **action.get("args", {}))
            tool_calls_log.append({
                "tool": tool_name,
                "success": result.success,
                "detail": result.error if not result.success else "",
            })
            messages.append({"role": "user", "content": f"Tool result: {json.dumps(result.model_dump())}"})

    bound_log.warning("agent_max_iterations_reached", iterations=iterations_used, total_tokens=total_tokens)
    save_context(context)
    return {"run_id": run_id, "result": "Max iterations reached",
            "iterations": iterations_used, "total_tokens": total_tokens, "status": "incomplete"}


async def run_agent_with_timeout(config: AgentConfig, extra_context: dict | None = None) -> dict:
    try:
        return await asyncio.wait_for(
            run_agent(config, extra_context=extra_context),
            timeout=AGENT_TOTAL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.error("agent_total_timeout", agent_id=config.id, timeout=AGENT_TOTAL_TIMEOUT)
        return {"run_id": "unknown", "result": "Agent run exceeded time limit",
                "iterations": 0, "status": "timeout"}
