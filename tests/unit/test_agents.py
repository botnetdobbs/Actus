import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.orchestrator import extract_json, run_agent, _build_rag_query, _MAX_SCHEMA_CORRECTIONS, _MAX_TOOL_RESULT_CHARS
from app.agents.builder import AgentConfig
from app.agents.tools import MAX_INVOKE_DEPTH, MAX_PARALLEL_AGENTS, ToolResult, _invoke_stack


# ── helpers ───────────────────────────────────────────────────────────────────

def loads(s: str) -> dict:
    return json.loads(extract_json(s))


def make_config(**kwargs) -> AgentConfig:
    defaults = {
        "id": "test-agent",
        "name": "Test Agent",
        "model": "ollama/mistral",
        "max_iterations": 3,
        "tools": ["search"],
        "token_budget": 10_000,
    }
    return AgentConfig(**(defaults | kwargs))


def llm_response(content: str, total_tokens: int = 100,
                 prompt_tokens: int = 60, completion_tokens: int = 40):
    r = MagicMock()
    r.choices[0].message.content = content
    r.usage.total_tokens = total_tokens
    r.usage.prompt_tokens = prompt_tokens
    r.usage.completion_tokens = completion_tokens
    return r


DONE = llm_response('{"done": true, "result": "finished"}')
TOOL_CALL = llm_response('{"tool": "search", "args": {"query": "test"}}')
NON_JSON = llm_response("Here is a summary of the findings.")


# ── extract_json: no fences ───────────────────────────────────────────────────

def test_plain_json():
    assert loads('{"done": true, "result": "ok"}') == {"done": True, "result": "ok"}


def test_plain_json_with_whitespace():
    assert loads('  {"tool": "search"}  ') == {"tool": "search"}


# ── extract_json: fenced ──────────────────────────────────────────────────────

def test_json_fence():
    assert loads("```\n{\"done\": true}\n```") == {"done": True}


def test_json_fence_with_language_tag():
    assert loads("```json\n{\"tool\": \"lookup\", \"args\": {}}\n```") == {"tool": "lookup", "args": {}}


def test_fence_without_closing():
    assert loads("```json\n{\"done\": true}") == {"done": True}


def test_multiline_json_in_fence():
    raw = "```json\n{\n  \"tool\": \"query\",\n  \"args\": {\"limit\": 10}\n}\n```"
    assert loads(raw) == {"tool": "query", "args": {"limit": 10}}


def test_json_with_backtick_value():
    assert loads('{"code": "x = `hello`"}') == {"code": "x = `hello`"}


def test_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        loads("not json at all")


def test_invalid_json_in_fence_raises():
    with pytest.raises(json.JSONDecodeError):
        loads("```json\nnot json\n```")


# ── run_tool: error truncation ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_tool_truncates_long_error():
    from app.agents.tools import run_tool, _tools, _MAX_TOOL_ERROR_CHARS

    def boom(**kwargs):
        raise ValueError("x" * (_MAX_TOOL_ERROR_CHARS + 50))

    _tools["__boom_test__"] = boom
    try:
        result = await run_tool("__boom_test__")
    finally:
        del _tools["__boom_test__"]

    assert result.success is False
    assert len(result.error) == _MAX_TOOL_ERROR_CHARS


@pytest.mark.asyncio
async def test_run_tool_propagates_team_id_context_to_sync_tool():
    from app.agents.tools import run_tool, _tools, _team_id_context

    observed = {}

    def reads_team_id(**kwargs):
        observed["team_id"] = _team_id_context.get()
        return "ok"

    _tools["__team_ctx_test__"] = reads_team_id
    token = _team_id_context.set(123)
    try:
        await run_tool("__team_ctx_test__")
    finally:
        _team_id_context.reset(token)
        del _tools["__team_ctx_test__"]

    assert observed["team_id"] == 123


# ── run_agent: happy path ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_happy_path_completes():
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=DONE)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config())
    assert result["status"] == "completed"
    assert result["result"] == "finished"
    assert result["total_tokens"] == 100
    assert result["prompt_tokens"] == 60
    assert result["completion_tokens"] == 40
    assert result["confidence"] is None  # done signal had no confidence field


# ── run_agent: failure modes ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_non_json_response_triggers_recovery():
    # Non-JSON followed by a valid done signal — agent recovers
    with patch("app.agents.orchestrator.call_llm_with_retry",
               AsyncMock(side_effect=[NON_JSON, DONE])), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(max_iterations=3))
    assert result["status"] == "completed"
    assert result["result"] == "finished"


@pytest.mark.asyncio
async def test_non_json_at_last_iteration_returns_incomplete():
    # Non-JSON at the final iteration — returns incomplete with raw text
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=NON_JSON)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(max_iterations=1))
    assert result["status"] == "incomplete"
    assert "summary" in result["result"]  # raw LLM text preserved


@pytest.mark.asyncio
async def test_llm_call_failure_returns_error():
    with patch("app.agents.orchestrator.call_llm_with_retry",
               AsyncMock(side_effect=Exception("connection refused"))), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config())
    assert result["status"] == "error"
    assert "connection refused" in result["error"]


@pytest.mark.asyncio
async def test_max_iterations_returns_incomplete():
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=TOOL_CALL)), \
         patch("app.agents.orchestrator.run_tool",
               AsyncMock(return_value=ToolResult(tool_name="search", success=True, output="data"))), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(max_iterations=2))
    assert result["status"] == "incomplete"
    assert result["iterations"] == 2


@pytest.mark.asyncio
async def test_token_budget_exceeded_stops_run():
    over_budget = llm_response('{"tool": "search", "args": {}}', total_tokens=200)
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=over_budget)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(token_budget=50))
    assert result["status"] == "incomplete"
    assert result["total_tokens"] == 200


@pytest.mark.asyncio
async def test_tool_failure_does_not_crash():
    responses = [
        llm_response('{"tool": "search", "args": {"query": "test"}}'),
        DONE,
    ]
    failed = ToolResult(tool_name="search", success=False, output=None, error="timeout")
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(side_effect=responses)), \
         patch("app.agents.orchestrator.run_tool", AsyncMock(return_value=failed)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config())
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_unauthorised_tool_not_executed():
    responses = [
        llm_response('{"tool": "delete_all", "args": {}}'),
        DONE,
    ]
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(side_effect=responses)), \
         patch("app.agents.orchestrator.run_tool") as mock_run, \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(tools=["search"]))
    mock_run.assert_not_called()
    assert result["status"] == "completed"


# ── Path 2: pre-loop RAG context injection ────────────────────────────────────

def test_build_rag_query_static_template():
    config = make_config(rag_query_template="overdue invoices unpaid")
    assert _build_rag_query(config, None) == "overdue invoices unpaid"


def test_build_rag_query_dynamic_template():
    config = make_config(rag_query_template="invoices for client {client}")
    assert _build_rag_query(config, {"client": "Acme"}) == "invoices for client Acme"


def test_build_rag_query_template_missing_var_uses_raw():
    config = make_config(rag_query_template="invoices for {client}")
    assert _build_rag_query(config, {}) == "invoices for {client}"


def test_build_rag_query_fallback_to_extra_context_query():
    config = make_config(rag_query_template="")
    assert _build_rag_query(config, {"query": "at-risk customers"}) == "at-risk customers"


def test_build_rag_query_no_template_no_query():
    config = make_config(rag_query_template="")
    assert _build_rag_query(config, {"region": "EU"}) is None
    assert _build_rag_query(config, None) is None


@pytest.mark.asyncio
async def test_rag_context_preloaded_into_messages():
    retrieved = [{"document": "Customer name=Alice segment=enterprise", "metadata": {"type": "Customer", "object_id": 1}, "score": 0.92}]
    with patch("app.agents.orchestrator.retrieve", return_value=retrieved), \
         patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=DONE)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(
            make_config(rag_query_template="enterprise customers"),
            extra_context=None,
        )
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_rag_context_failure_is_non_fatal():
    with patch("app.agents.orchestrator.retrieve", side_effect=Exception("chroma unavailable")), \
         patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=DONE)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(rag_query_template="find something"))
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_rag_no_template_skips_retrieval():
    with patch("app.agents.orchestrator.retrieve") as mock_retrieve, \
         patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=DONE)), \
         patch("app.agents.orchestrator.save_context"):
        await run_agent(make_config())
    mock_retrieve.assert_not_called()


# ── team_id threading ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_agent_threads_explicit_team_id_to_retrieve():
    with patch("app.agents.orchestrator.retrieve", return_value=[]) as mock_retrieve, \
         patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=DONE)), \
         patch("app.agents.orchestrator.save_context"):
        await run_agent(make_config(rag_query_template="enterprise customers"), team_id=7)
    _, kwargs = mock_retrieve.call_args
    assert kwargs["team_id"] == 7


@pytest.mark.asyncio
async def test_run_agent_inherits_team_id_from_context():
    from app.agents.tools import _team_id_context
    token = _team_id_context.set(99)
    try:
        with patch("app.agents.orchestrator.retrieve", return_value=[]) as mock_retrieve, \
             patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=DONE)), \
             patch("app.agents.orchestrator.save_context"):
            await run_agent(make_config(rag_query_template="enterprise customers"))
    finally:
        _team_id_context.reset(token)
    _, kwargs = mock_retrieve.call_args
    assert kwargs["team_id"] == 99


@pytest.mark.asyncio
async def test_nested_invoke_agent_inherits_team_id():
    child_result = {"status": "completed", "result": "ok",
                    "confidence": 0.9, "iterations": 1, "total_tokens": 50}
    observed = {}

    async def fake_run_agent(config, **kwargs):
        from app.agents.tools import _team_id_context
        observed["team_id"] = _team_id_context.get()
        return child_result

    async def parent_inner(*args, **kwargs):
        from app.agents.tools import invoke_agent
        return await invoke_agent(agent_id="child", query="go")

    with patch("app.agents.builder.get_agent", return_value=make_config(id="child")), \
         patch("app.agents.orchestrator.run_agent", side_effect=fake_run_agent), \
         patch("app.agents.orchestrator._run_agent_inner", side_effect=parent_inner):
        await run_agent(make_config(id="parent"), team_id=7)

    assert observed["team_id"] == 7


# ── invoke_agent and invoke_agents_parallel ───────────────────────────────────

@pytest.mark.asyncio
async def test_invoke_agent_happy_path():
    child_result = {"status": "completed", "result": "analysis done",
                    "confidence": 0.9, "iterations": 1, "total_tokens": 50}
    with patch("app.agents.builder.get_agent", return_value=make_config(id="child")), \
         patch("app.agents.orchestrator.run_agent", AsyncMock(return_value=child_result)):
        from app.agents.tools import invoke_agent
        result = await invoke_agent(agent_id="child", query="analyse this")
    assert result["status"] == "completed"
    assert result["result"] == "analysis done"
    assert result["confidence"] == 0.9
    assert "prompt_tokens" not in result  # internal fields stripped


@pytest.mark.asyncio
async def test_invoke_agent_unknown_agent_returns_error():
    with patch("app.agents.builder.get_agent", side_effect=KeyError("no-such-agent")):
        from app.agents.tools import invoke_agent
        result = await invoke_agent(agent_id="no-such-agent")
    assert result["status"] == "error"
    assert "Unknown agent" in result["error"]


@pytest.mark.asyncio
async def test_invoke_agent_circular_detection():
    # Simulate the stack already containing the target agent
    token = _invoke_stack.set(["planner", "child-agent"])
    try:
        from app.agents.tools import invoke_agent
        result = await invoke_agent(agent_id="child-agent", query="anything")
    finally:
        _invoke_stack.reset(token)
    assert result["status"] == "error"
    assert "Circular" in result["error"]


@pytest.mark.asyncio
async def test_invoke_agent_depth_limit():
    deep_stack = [f"agent-{i}" for i in range(MAX_INVOKE_DEPTH)]
    token = _invoke_stack.set(deep_stack)
    try:
        from app.agents.tools import invoke_agent
        result = await invoke_agent(agent_id="one-more")
    finally:
        _invoke_stack.reset(token)
    assert result["status"] == "error"
    assert "depth" in result["error"].lower()


@pytest.mark.asyncio
async def test_invoke_agents_parallel_happy_path():
    r1 = {"status": "completed", "result": "result-a", "iterations": 1, "total_tokens": 10}
    r2 = {"status": "completed", "result": "result-b", "iterations": 1, "total_tokens": 10}
    configs = {"agent-a": make_config(id="agent-a"), "agent-b": make_config(id="agent-b")}
    with patch("app.agents.builder.get_agent", side_effect=lambda aid: configs[aid]), \
         patch("app.agents.orchestrator.run_agent", AsyncMock(side_effect=[r1, r2])):
        from app.agents.tools import invoke_agents_parallel
        results = await invoke_agents_parallel(agent_ids=["agent-a", "agent-b"], query="go")
    assert len(results) == 2
    assert all(r["status"] == "completed" for r in results)


@pytest.mark.asyncio
async def test_invoke_agents_parallel_partial_failure():
    r_ok = {"status": "completed", "result": "ok", "iterations": 1, "total_tokens": 10}

    def _get_agent(aid):
        if aid == "good":
            return make_config(id="good")
        raise KeyError(aid)

    with patch("app.agents.builder.get_agent", side_effect=_get_agent), \
         patch("app.agents.orchestrator.run_agent", AsyncMock(return_value=r_ok)):
        from app.agents.tools import invoke_agents_parallel
        results = await invoke_agents_parallel(agent_ids=["good", "missing"])
    assert len(results) == 2
    statuses = {r["status"] for r in results}
    assert "completed" in statuses
    assert "error" in statuses


@pytest.mark.asyncio
async def test_invoke_agents_parallel_empty_list():
    from app.agents.tools import invoke_agents_parallel
    results = await invoke_agents_parallel(agent_ids=[])
    assert results == []


@pytest.mark.asyncio
async def test_confidence_score_passed_through():
    done_with_confidence = llm_response('{"done": true, "result": "analysis done", "confidence": 0.92}')
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=done_with_confidence)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config())
    assert result["status"] == "completed"
    assert result["confidence"] == 0.92


@pytest.mark.asyncio
async def test_confidence_none_when_not_provided():
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=DONE)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config())
    assert result["confidence"] is None


def test_extract_json_handles_text_before_json():
    mixed = 'Sure! Here is my response: {"done": true, "result": "ok"}'
    assert extract_json(mixed) == '{"done": true, "result": "ok"}'


def test_extract_json_handles_nested_args():
    mixed = 'Let me call a tool: {"tool": "search", "args": {"query": "test"}}'
    result = extract_json(mixed)
    parsed = json.loads(result)
    assert parsed["tool"] == "search"
    assert parsed["args"]["query"] == "test"


def test_extract_json_handles_text_after_json():
    mixed = '{"done": true, "result": "ok"} then I will proceed with more reasoning.'
    result = extract_json(mixed)
    parsed = json.loads(result)
    assert parsed["done"] is True


@pytest.mark.asyncio
async def test_empty_action_sends_recovery_message():
    empty_response = llm_response('{}')
    with patch("app.agents.orchestrator.call_llm_with_retry",
               AsyncMock(side_effect=[empty_response, DONE])), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(max_iterations=3))
    assert result["status"] == "completed"
    assert result["iterations"] == 2  # recovered on second iteration


@pytest.mark.asyncio
async def test_invoke_agents_parallel_cap_single_rejection():
    from app.agents.tools import invoke_agents_parallel
    too_many = [f"agent-{i}" for i in range(MAX_PARALLEL_AGENTS + 1)]
    results = await invoke_agents_parallel(agent_ids=too_many)
    assert len(results) == 1  # single rejection, not N identical errors
    assert results[0]["status"] == "error"
    assert "Batch rejected" in results[0]["error"]


@pytest.mark.asyncio
async def test_non_dict_args_do_not_crash():
    null_args = llm_response('{"tool": "search", "args": null}')
    with patch("app.agents.orchestrator.call_llm_with_retry",
               AsyncMock(side_effect=[null_args, DONE])), \
         patch("app.agents.orchestrator.run_tool",
               AsyncMock(return_value=ToolResult(tool_name="search", success=True, output="ok"))), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(max_iterations=3))
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_save_context_failure_does_not_crash_run():
    from unittest.mock import patch as _patch
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=DONE)), \
         _patch("app.agents.orchestrator.save_context",
                side_effect=ValueError("context too large")):
        result = await run_agent(make_config())
    assert result["status"] == "completed"
    assert result["result"] == "finished"


@pytest.mark.asyncio
async def test_invoke_agent_no_call_stack_in_error():
    token = _invoke_stack.set(["planner", "child"])
    try:
        from app.agents.tools import invoke_agent
        result = await invoke_agent(agent_id="child")
    finally:
        _invoke_stack.reset(token)
    assert result["status"] == "error"
    assert "call_stack" not in result


# ── Tool schema filtering ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_only_allowed_tools_in_system_prompt():
    fake_schemas = {
        "tool_a": {"name": "tool_a", "description": "Does A"},
        "tool_b": {"name": "tool_b", "description": "Does B"},
        "tool_c": {"name": "tool_c", "description": "Does C"},
    }
    captured: list[list] = []

    async def mock_llm(model, messages, **kwargs):
        captured.append(messages)
        return DONE

    with patch("app.agents.orchestrator._tool_schemas", fake_schemas), \
         patch("app.agents.orchestrator.call_llm_with_retry", side_effect=mock_llm), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(tools=["tool_a"]))

    assert result["status"] == "completed"
    system_content = captured[0][0]["content"]
    assert "tool_a" in system_content
    assert "tool_b" not in system_content
    assert "tool_c" not in system_content


@pytest.mark.asyncio
async def test_unregistered_tool_logged_as_warning():
    fake_schemas = {"existing_tool": {"name": "existing_tool", "description": "Real tool"}}
    with patch("app.agents.orchestrator._tool_schemas", fake_schemas), \
         patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=DONE)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(tools=["existing_tool", "missing_tool"]))
    # unregistered tool is silently filtered — agent still runs
    assert result["status"] == "completed"


# ── Env interpolation (builder._interpolate_env) ──────────────────────────────

def test_env_interpolation_replaces_placeholder(monkeypatch):
    from app.agents.builder import _interpolate_env
    monkeypatch.setenv("MY_SECRET", "super-secret-value")
    result = _interpolate_env({"webhook": {"secret": "${MY_SECRET}"}})
    assert result == {"webhook": {"secret": "super-secret-value"}}


def test_env_interpolation_missing_var_raises(monkeypatch):
    from app.agents.builder import _interpolate_env
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    with pytest.raises(ValueError, match="DEFINITELY_NOT_SET"):
        _interpolate_env("${DEFINITELY_NOT_SET}")


def test_env_interpolation_passthrough_non_string():
    from app.agents.builder import _interpolate_env
    assert _interpolate_env(42) == 42
    assert _interpolate_env(True) is True
    assert _interpolate_env(None) is None
    assert _interpolate_env(3.14) == 3.14


def test_env_interpolation_nested_list(monkeypatch):
    from app.agents.builder import _interpolate_env
    monkeypatch.setenv("DB_URL", "postgres://localhost/db")
    result = _interpolate_env(["static", "${DB_URL}", 42])
    assert result == ["static", "postgres://localhost/db", 42]


# ── Native tool calling ───────────────────────────────────────────────────────

def test_supports_native_tools_non_ollama():
    from app.agents.tools import _supports_native_tools
    assert _supports_native_tools("openai/gpt-4o") is True
    assert _supports_native_tools("anthropic/claude-sonnet-4-6") is True


def test_supports_native_tools_ollama_false():
    from app.agents.tools import _supports_native_tools
    assert _supports_native_tools("ollama/mistral") is False
    assert _supports_native_tools("ollama/llama3") is False


def test_to_openai_tools_schema_conversion():
    from app.agents.tools import to_openai_tools
    schemas = [{
        "name": "search",
        "description": "Search documents",
        "parameters": {
            "query": {"type": "str", "required": True},
            "top_k": {"type": "int", "required": False, "default": 5},
        },
    }]
    result = to_openai_tools(schemas)
    assert len(result) == 1
    fn = result[0]
    assert fn["type"] == "function"
    assert fn["function"]["name"] == "search"
    assert fn["function"]["parameters"]["properties"]["query"]["type"] == "string"
    assert fn["function"]["parameters"]["properties"]["top_k"]["type"] == "integer"
    assert "query" in fn["function"]["parameters"]["required"]
    assert "top_k" not in fn["function"]["parameters"]["required"]


def _native_done_response(content: str):
    r = MagicMock()
    r.choices[0].message.content = content
    r.choices[0].message.tool_calls = []
    r.usage.total_tokens = 100
    r.usage.prompt_tokens = 60
    r.usage.completion_tokens = 40
    return r


def _native_tool_response(tool_name: str, args: dict):
    import json as _json
    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = tool_name
    tc.function.arguments = _json.dumps(args)
    r = MagicMock()
    r.choices[0].message.content = None
    r.choices[0].message.tool_calls = [tc]
    r.usage.total_tokens = 80
    r.usage.prompt_tokens = 50
    r.usage.completion_tokens = 30
    return r


@pytest.mark.asyncio
async def test_native_done_no_tool_calls():
    """In native mode, empty tool_calls → done; content is the final answer."""
    with patch("app.agents.orchestrator.call_llm_with_retry",
               AsyncMock(return_value=_native_done_response("Here is the answer"))), \
         patch("app.agents.orchestrator.save_context"), \
         patch("app.agents.orchestrator._supports_native_tools", return_value=True):
        result = await run_agent(make_config(model="openai/gpt-4o", native_tools=True))
    assert result["status"] == "completed"
    assert result["result"] == "Here is the answer"
    assert result["confidence"] is None  # no confidence wrapper in native mode


@pytest.mark.asyncio
async def test_native_tool_called_then_done():
    """Native mode: tool_calls response → execute tool → done on next iteration."""
    tool_response = _native_tool_response("search", {"query": "hello"})
    done_response = _native_done_response("final answer after search")

    with patch("app.agents.orchestrator.call_llm_with_retry",
               AsyncMock(side_effect=[tool_response, done_response])), \
         patch("app.agents.orchestrator.save_context"), \
         patch("app.agents.orchestrator._supports_native_tools", return_value=True), \
         patch("app.agents.orchestrator.run_tool",
               AsyncMock(return_value=ToolResult(tool_name="search", success=True, output=["result1"]))):
        result = await run_agent(make_config(model="openai/gpt-4o", native_tools=True,
                                             max_iterations=3))
    assert result["status"] == "completed"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "search"


@pytest.mark.asyncio
async def test_native_unauthorised_tool_rejected():
    """Native mode: tool_call for a tool not in config.tools is rejected."""
    tool_response = _native_tool_response("forbidden_tool", {"x": 1})
    done_response = _native_done_response("ok")

    with patch("app.agents.orchestrator.call_llm_with_retry",
               AsyncMock(side_effect=[tool_response, done_response])), \
         patch("app.agents.orchestrator.save_context"), \
         patch("app.agents.orchestrator._supports_native_tools", return_value=True):
        result = await run_agent(make_config(model="openai/gpt-4o", native_tools=True,
                                             tools=[], max_iterations=5))
    assert result["tool_calls"][0]["detail"] == "unauthorised"


# ── Output schema validation ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_output_schema_none_when_no_schema():
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=DONE)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config())
    assert result["output_schema_valid"] is None


@pytest.mark.asyncio
async def test_output_schema_valid_on_conforming_result():
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    good_result_response = llm_response('{"done": true, "result": "{\\"answer\\": \\"yes\\"}"}')
    with patch("app.agents.orchestrator.call_llm_with_retry",
               AsyncMock(return_value=good_result_response)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(output_schema=schema))
    assert result["status"] == "completed"
    assert result["output_schema_valid"] is True


@pytest.mark.asyncio
async def test_output_schema_invalid_triggers_correction_then_valid():
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    bad_result = llm_response('{"done": true, "result": "not json at all"}')
    good_result = llm_response('{"done": true, "result": "{\\"answer\\": \\"yes\\"}"}')
    with patch("app.agents.orchestrator.call_llm_with_retry",
               AsyncMock(side_effect=[bad_result, good_result])), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(output_schema=schema, max_iterations=3))
    assert result["status"] == "completed"
    assert result["output_schema_valid"] is True


@pytest.mark.asyncio
async def test_output_schema_invalid_at_final_iteration_returns_false():
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    bad = llm_response('{"done": true, "result": "not json at all"}')
    with patch("app.agents.orchestrator.call_llm_with_retry",
               AsyncMock(return_value=bad)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(output_schema=schema, max_iterations=1))
    assert result["status"] == "completed"
    assert result["output_schema_valid"] is False


# ── Run trace ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trace_included_in_result():
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=DONE)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config())
    assert "trace" in result
    assert isinstance(result["trace"], list)
    assert len(result["trace"]) == 1
    entry = result["trace"][0]
    assert entry["iteration"] == 0
    assert "llm_response" in entry
    assert "messages_sent_count" in entry
    assert "tokens_this_iteration" in entry


# ── _cap_trace: embedded truncated flag ──────────────────────────────────────

def test_cap_trace_embeds_truncated_false_when_small():
    from app.agents.audit import _cap_trace
    trace = [{"iteration": 0, "llm_response": "ok"}]
    result = json.loads(_cap_trace(trace))
    assert result["truncated"] is False
    assert result["iterations"] == trace


def test_cap_trace_embeds_truncated_true_when_over_limit():
    from app.agents.audit import _cap_trace, _TRACE_SIZE_LIMIT
    big_entry = {"iteration": 0, "tool_calls": ["x" * 1000], "tool_result": "y" * 1000}
    # Build enough entries to exceed the limit
    trace = [big_entry] * ((_TRACE_SIZE_LIMIT // 2000) + 5)
    result = json.loads(_cap_trace(trace))
    assert result["truncated"] is True
    for entry in result["iterations"]:
        assert "tool_calls" not in entry
        assert "tool_result" not in entry


# ── Native multi-tool trace captures all calls ────────────────────────────────

@pytest.mark.asyncio
async def test_native_multi_tool_trace_captures_all_calls():
    import json as _json
    tool_names = ["search", "lookup"]
    tool_calls_list = []
    for i, name in enumerate(tool_names):
        tc = MagicMock()
        tc.id = f"call_{i}"
        tc.function.name = name
        tc.function.arguments = _json.dumps({"q": f"query{i}"})
        tool_calls_list.append(tc)

    multi_tool_response = MagicMock()
    multi_tool_response.choices[0].message.content = None
    multi_tool_response.choices[0].message.tool_calls = tool_calls_list
    multi_tool_response.usage.total_tokens = 80
    multi_tool_response.usage.prompt_tokens = 50
    multi_tool_response.usage.completion_tokens = 30

    done_response = MagicMock()
    done_response.choices[0].message.content = "all done"
    done_response.choices[0].message.tool_calls = []
    done_response.usage.total_tokens = 40
    done_response.usage.prompt_tokens = 25
    done_response.usage.completion_tokens = 15

    with patch("app.agents.orchestrator.call_llm_with_retry",
               AsyncMock(side_effect=[multi_tool_response, done_response])), \
         patch("app.agents.orchestrator.save_context"), \
         patch("app.agents.orchestrator._supports_native_tools", return_value=True), \
         patch("app.agents.orchestrator.run_tool",
               AsyncMock(return_value=ToolResult(tool_name="search", success=True, output="ok"))):
        result = await run_agent(make_config(model="openai/gpt-4o", native_tools=True,
                                             tools=["search", "lookup"], max_iterations=5))

    assert result["status"] == "completed"
    tool_iteration = result["trace"][0]
    assert "tool_calls" in tool_iteration
    assert len(tool_iteration["tool_calls"]) == 2
    names = [tc["name"] for tc in tool_iteration["tool_calls"]]
    assert "search" in names and "lookup" in names


# ── Tool result truncation ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tool_result_truncated_at_limit():
    big_output = "X" * (_MAX_TOOL_RESULT_CHARS + 500)
    responses = [TOOL_CALL, DONE]
    captured_messages: list = []

    async def mock_llm(model, messages, **kwargs):
        captured_messages.append(messages[:])
        return responses.pop(0)

    with patch("app.agents.orchestrator.call_llm_with_retry", side_effect=mock_llm), \
         patch("app.agents.orchestrator.save_context"), \
         patch("app.agents.orchestrator.run_tool",
               AsyncMock(return_value=ToolResult(tool_name="search", success=True, output=big_output))):
        result = await run_agent(make_config(max_iterations=3))

    assert result["status"] == "completed"
    # The tool result message sent to the LLM must be capped
    tool_result_msg = next(
        (m["content"] for msgs in captured_messages for m in msgs
         if isinstance(m.get("content"), str) and "[truncated]" in m["content"]),
        None,
    )
    assert tool_result_msg is not None, "Expected a truncated tool result message"


# ── Schema correction budget ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_schema_correction_capped_at_max_corrections():
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    bad = llm_response('{"done": true, "result": "not valid json schema output"}')
    # Supply _MAX_SCHEMA_CORRECTIONS bad results + more bad results beyond the cap
    responses = [bad] * (_MAX_SCHEMA_CORRECTIONS + 3)
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(side_effect=responses)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(output_schema=schema, max_iterations=20))
    # After _MAX_SCHEMA_CORRECTIONS + 1 done signals, corrections are exhausted → returns completed
    assert result["status"] == "completed"
    assert result["output_schema_valid"] is False


@pytest.mark.asyncio
async def test_schema_errors_returned_on_invalid_result():
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    bad = llm_response('{"done": true, "result": "not json at all"}')
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=bad)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(output_schema=schema, max_iterations=1))
    assert result["output_schema_valid"] is False
    assert result["output_schema_errors"] is not None
    assert len(result["output_schema_errors"]) > 0


# ── Run trace ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trace_has_tool_call_entry():
    with patch("app.agents.orchestrator.call_llm_with_retry",
               AsyncMock(side_effect=[TOOL_CALL, DONE])), \
         patch("app.agents.orchestrator.save_context"), \
         patch("app.agents.orchestrator.run_tool",
               AsyncMock(return_value=ToolResult(tool_name="search", success=True, output="results"))):
        result = await run_agent(make_config(max_iterations=3))
    assert len(result["trace"]) == 2  # tool call iteration + done iteration
    # tool_calls is now a list of calls per iteration
    assert "tool_calls" in result["trace"][0]
    assert len(result["trace"][0]["tool_calls"]) == 1
    assert result["trace"][0]["tool_calls"][0]["name"] == "search"
