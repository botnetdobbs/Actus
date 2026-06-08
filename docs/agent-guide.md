# Actus Agent Guide

Everything you need to build, register, invoke, and test agents on this platform.

---

## Architecture overview

```
POST /automation/trigger/{agent_id}
        │  extra_context (file_path, question, …)
        ▼
  Workflow row created → returns {workflow_id}
        │
        ├──────────────────────────────────────────────┐
        │  background task                              │  SSE client
        ▼                                              ▼
  run_agent_with_timeout()          ← 600 s   GET /automation/workflows/{id}/stream
        │                                              │
        ▼                                              ├─ immediate status event (from DB)
  _run_agent_inner(event_queue)                        │
        │                                              │  live queue path (same process):
        ├─ pre-loop RAG retrieval                      ├─ iteration_start events
        ├─ extra_context injected as user message      ├─ tool_call / tool_result events
        │                                              ├─ done event
        └─ ReAct loop  (up to max_iterations)          │
                │                                      │  DB-poll fallback (reconnect/LB):
                ├─ call_llm_with_retry()   ← 120 s     ├─ polls every 1 s until terminal
                │        asyncio.wait_for              │
                ├─ emit iteration_start/tool events ───┘
                ├─ run_tool()              ← 30 s / 600 s for long-running tools
                └─ repeat until done / budget / max_iterations / timeout
        │
        ▼
  Workflow.status → completed / failed / timeout
  AgentRunLog row written
  SSE sentinel emitted → final WorkflowResponse delivered to stream
```

---

## YAML agent configuration

Every `.yaml` file in `config/agents/` is loaded at startup and hot-reloadable via `POST /automation/reload`.

```yaml
# Required
id: my_agent                    # used in API calls: POST /automation/trigger/my_agent
name: My Agent                  # human-readable label
description: What this agent does

# LLM
model: anthropic/claude-haiku-4-5-20251001   # any LiteLLM model string
temperature: 0.1                             # 0.0–2.0
max_response_tokens: 1024                    # max tokens per LLM response

# Loop controls
max_iterations: 8               # hard cap on ReAct iterations
token_budget: 50000             # cumulative token limit across all iterations

# Tools the agent may call (must match @tool name strings)
tools:
  - chunk_and_index_document
  - search_document
  - cleanup_document

# RAG pre-loop context (runs BEFORE the loop starts)
# If set, retrieves top-k documents and injects them as the first user message.
# Use {key} placeholders filled from extra_context.
# Set to "" to disable — mandatory for document agents to avoid unrelated pre-loading.
rag_query_template: "invoices for client {client_name}"
rag_top_k: 5

# Optional: override Ollama base URL for this agent
api_base: ""

# Optional: run on a cron schedule (uses APScheduler)
schedule:
  cron: "0 9 * * 1-5"          # weekdays at 09:00

# System prompt — the agent's personality and step instructions
system_prompt: |
  You are a specialist agent. Follow these steps exactly:

  STEP 1 — …
  STEP 2 — …
  …

  RULES: …
```

### Field reference

| Field | Type | Default | Notes |
|---|---|---|---|
| `id` | str | required | URL-safe, unique across all agents |
| `name` | str | required | Display name |
| `description` | str | `""` | Shown in agent listings |
| `model` | str | `ollama/mistral` | Any LiteLLM model string |
| `temperature` | float | `0.7` | 0.1 for factual/structured tasks |
| `max_response_tokens` | int | `1024` | Per LLM call; max 8192 |
| `max_iterations` | int | `5` | Set higher for multi-step tasks |
| `token_budget` | int | `10000` | Cumulative; raise for large docs or many steps |
| `tools` | list[str] | `[]` | Only listed tools are callable |
| `rag_query_template` | str | `""` | `""` disables pre-loop RAG entirely |
| `rag_top_k` | int | `5` | Documents retrieved pre-loop |
| `api_base` | str | `""` | Overrides `OLLAMA_BASE_URL` for this agent |
| `schedule.cron` | str | None | APScheduler cron expression |

---

## Writing a tool

### The `@tool` decorator

```python
# app/my_module/my_tools.py
from app.agents.tools import tool

@tool(
    "do_something",
    "One-sentence description shown to the LLM in the tool manifest.",
)
def do_something(input_text: str, limit: int = 10) -> dict:
    # sync functions are run in a thread-pool executor automatically
    return {"result": input_text[:limit]}
```

Async tools work identically:

```python
@tool("fetch_data", "Fetch data from the external API.")
async def fetch_data(record_id: int) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.get(f"https://api.example.com/records/{record_id}")
    return r.json()
```

### What the decorator does

1. Registers the function in `_tools[name]`
2. Introspects type hints and defaults to build `_tool_schemas[name]` the JSON manifest the LLM receives
3. Both sync and async functions are supported; sync functions run in `asyncio.get_running_loop().run_in_executor(None, …)` to avoid blocking the event loop

### Tool return values

Return any JSON-serialisable value: `dict`, `list`, `str`, `int`. The orchestrator serialises it with `json.dumps(output, default=str)` and appends it as a user message to the conversation.

On failure, raise an exception, the orchestrator catches it and reports `{"error": "…"}` to the LLM.

### Registering tools at startup

Import the module containing your `@tool` functions in `app/main.py` for its side-effects:

```python
# app/main.py
import app.my_module.my_tools  # noqa: F401, registers tools on import
```

Tools are global — once registered, any agent that lists the tool name in its `tools:` YAML can call it.

### Long-running tools

The default tool timeout is **30 seconds**. If your tool does heavy I/O (file parsing, embedding generation, external API calls that take >30 s), add its name to `_LONG_RUNNING_TOOLS` in `app/agents/orchestrator.py`:

```python
# app/agents/orchestrator.py
_LONG_RUNNING_TOOLS = {
    "invoke_agent",
    "invoke_agents_parallel",
    "chunk_and_index_document",   # ← add yours here
    "my_slow_tool",
}
```

Long-running tools inherit `AGENT_TOTAL_TIMEOUT` (600 s) as their timeout instead of 30 s. Add only tools that genuinely need it — a 30 s limit is a good guardrail against hung tools.

---

## The ReAct loop

Each iteration:

1. **LLM call** full message history sent to the model, 120 s timeout via `asyncio.wait_for`
2. **Parse** extract JSON from response (handles markdown fences, leading/trailing text)
3. **Act** one of three paths:
   - `{"done": true, "result": "…", "confidence": 0.9}` → agent finishes
   - `{"tool": "name", "args": {…}}` → tool executed, result appended, loop continues
   - Invalid JSON or empty action → recovery message injected, loop continues

### LLM response format the agent must use

```json
// Call a tool
{"tool": "search_document", "args": {"session_id": "abc", "query": "main findings"}}

// Finish
{"done": true, "result": "The document covers …", "confidence": 0.85}
```

`confidence` is optional (0.0 - 1.0). The system prompt should tell the agent when to use each signal.

### Tool authorisation

Agents can only call tools listed in their `tools:` YAML key. Attempts to call unlisted tools are blocked by the orchestrator; the LLM receives an error message and is told what tools are available.

---

## Pre-loop RAG context

If `rag_query_template` is non-empty, the orchestrator runs a semantic search **before** the first LLM iteration and prepends the results as a user message:

```
Retrieved context (pre-loaded):
[1] Customer name=Acme segment=enterprise; revenue=120000 (score: 0.91)
[2] Customer name=Globex segment=mid-market; revenue=45000 (score: 0.87)
…
```

Template placeholders are filled from `extra_context`:

```yaml
rag_query_template: "invoices overdue for {client_name}"
```

```json
// trigger body
{"extra_context": {"client_name": "Acme"}}
```

Set `rag_query_template: ""` to disable entirely. Always disable it for document agents (like `doc_qa`), otherwise the pre-loop retrieval pulls unrelated ontology data that inflates the token count.

---

## Passing runtime data to an agent (extra_context)

`extra_context` is a free-form dict passed at trigger time. It becomes a user message at the start of the conversation:

```
Context: {"file_path": "/tmp/actus-uploads/abc.pdf", "question": "What are the findings?"}
```

The system prompt should instruct the agent to read specific keys from this context:

```yaml
system_prompt: |
  STEP 1: Index the document at the path given in context["file_path"].
  Answer the question in context["question"].
```

The extra_context is PII-scrubbed before being sent to the LLM (`DATE_TIME` entities excluded to avoid false positives on UUIDs and file paths).

### Trigger via API

```bash
curl -X POST http://localhost:8000/automation/trigger/my_agent \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"extra_context": {"key": "value"}}'
```

Response: `{"status": "queued", "agent_id": "my_agent", "workflow_id": 42}`

### Stream the result (preferred)

```bash
curl -N "http://localhost:8000/automation/workflows/42/stream" \
  -H "Authorization: Bearer $TOKEN"
```

The stream opens immediately with the current status, then delivers per-iteration events as the agent works, and closes with the full `WorkflowResponse` when the run completes.

```
data: {"type": "status", "status": "running", "workflow_id": 42}
data: {"type": "iteration_start", "run_id": "abc", "iteration": 0}
data: {"type": "tool_call", "run_id": "abc", "iteration": 0, "tool": "my_tool", "args": {}}
data: {"type": "tool_result", "run_id": "abc", "iteration": 0, "tool": "my_tool", "success": true, "preview": "..."}
data: {"type": "done", "run_id": "abc", "status": "completed", "result": "...", "confidence": 0.9}
data: {"id": 42, "status": "completed", "result_json": "...", ...}
```

The stream is reconnectable — disconnect and re-open with the same `workflow_id` at any time.

### Poll for result (alternative)

```bash
watch -n 2 curl -s "http://localhost:8000/automation/workflows/42" \
  -H "Authorization: Bearer $TOKEN"
```

Refreshes every 2 seconds. `status` progresses: `pending` → `running` → `completed` / `failed` / `timeout`

`result_json` is the serialised agent result dict, including `result`, `confidence`, `iterations`, `tool_calls`, and token counts.

---

## Built-in tools

| Tool | Description |
|---|---|
| `semantic_search` | pgvector cosine similarity search across all indexed ontology objects |
| `invoke_agent` | Run another registered agent sequentially and return its result |
| `invoke_agents_parallel` | Run multiple agents in parallel (up to 10), returns list of results |

### Supported orchestration patterns

| Pattern | Tool | When to use |
|---|---|---|
| **Sequential pipeline** | `invoke_agent` | Agents must run in order — output of one feeds the next |
| **Parallel execution** | `invoke_agents_parallel` | Agents are independent — run concurrently, merge results |
| **Hierarchical manager-specialist** | Either | Manager agent decomposes and delegates; depth limit (5) and cycle detection enforced automatically |

### Agent-to-agent invocation

An agent's system prompt can instruct it to call other agents:

```yaml
# config/agents/planner.yaml
tools:
  - invoke_agent
  - invoke_agents_parallel

system_prompt: |
  Analyse the customer portfolio by delegating to specialist agents:
  1. Call invoke_agent with agent_id="churn_risk" and query=<customer segment>
  2. Call invoke_agent with agent_id="upsell_opportunity" and query=<same segment>
  3. Synthesise results and call done.
```

**Cycle detection** — `invoke_agent` tracks the invocation stack. If agent A calls agent B which calls agent A, the second call is rejected with `"Circular invocation detected"`. Max depth is 5.

**Parallel cap** — `invoke_agents_parallel` rejects batches larger than 10 with a single error result rather than N identical errors.

---

## Timeouts reference

| Timeout | Value | What it controls |
|---|---|---|
| `AGENT_TOTAL_TIMEOUT` | 600 s | `asyncio.wait_for` wrapping the entire `run_agent` coroutine |
| `LLM_PER_CALL_TIMEOUT` | 120 s | `asyncio.wait_for` wrapping each `call_llm_with_retry` call |
| Default tool timeout | 30 s | `run_tool` for any tool not in `_LONG_RUNNING_TOOLS` |
| Long-running tool timeout | 600 s | `run_tool` for tools in `_LONG_RUNNING_TOOLS` |

> **Why `asyncio.wait_for` and not the LiteLLM `timeout` kwarg?**  
> LiteLLM's `timeout` parameter is not reliably honoured for local Ollama inference, the HTTP read does not cancel at the asyncio level. `asyncio.wait_for` injects `CancelledError` into the coroutine stack, which is the only reliable cancellation mechanism for async code.

---

## Audit logging

Every agent run (success, error, timeout) writes a row to `agent_run_logs` via `log_agent_run()` in `app/agents/audit.py`. The record contains:

| Column | Notes |
|---|---|
| `run_id` | UUID matching the workflow |
| `agent_id` | YAML `id` field |
| `triggered_by` | User ID from the JWT |
| `model` | LiteLLM model string |
| `outcome` | `success` / `incomplete` / `error` / `timeout` |
| `tool_calls` | JSON array: `[{tool, success, detail}]` |
| `prompt_tokens` / `completion_tokens` / `total_tokens` | Token counts |
| `pii_detected` | Whether Presidio found PII in any message |
| `result_summary` | First 500 chars of the result, never the raw prompt |
| `ip_address` | Caller IP |

---

## Building a new agent — step by step

### 1. Create the tool module

```
app/my_feature/
    __init__.py
    my_tools.py
```

```python
# app/my_feature/my_tools.py
from app.agents.tools import tool

@tool("my_tool", "Does the thing. Returns {result: str}.")
def my_tool(input_text: str) -> dict:
    return {"result": input_text.upper()}
```

### 2. Register the module in main.py

```python
# app/main.py
import app.my_feature.my_tools  # noqa: F401
```

### 3. If the tool is slow, add it to _LONG_RUNNING_TOOLS

```python
# app/agents/orchestrator.py
_LONG_RUNNING_TOOLS = {
    "invoke_agent",
    "invoke_agents_parallel",
    "chunk_and_index_document",
    "my_tool",              # ← add here
}
```

### 4. Write the agent YAML

```yaml
# config/agents/my_agent.yaml
id: my_agent
name: My Agent
description: Does the thing
model: anthropic/claude-haiku-4-5-20251001
max_iterations: 5
temperature: 0.1
token_budget: 20000
max_response_tokens: 1024
rag_query_template: ""
tools:
  - my_tool
system_prompt: |
  You are My Agent. Follow these steps:

  STEP 1: Call my_tool with the input from context["input"].
  STEP 2: Call done with the result and a confidence score.

  RULES: Always call done. Never guess — use tool output only.
```

### 5. No server restart needed

Uvicorn runs with `--reload` in Docker. The tool module import is picked up on reload. The YAML is volume-mounted and re-read on `POST /automation/reload` or on the next server restart.

### 6. Trigger and verify

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -d "username=alice&password=changeme123" | jq -r .access_token)

WF_ID=$(curl -s -X POST http://localhost:8000/automation/trigger/my_agent \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"extra_context": {"input": "hello world"}}' | jq -r .workflow_id)

# Stream live progress (recommended)
curl -N "http://localhost:8000/automation/workflows/$WF_ID/stream" \
  -H "Authorization: Bearer $TOKEN"

# Or watch the poll endpoint (alternative)
watch -n 2 curl -s "http://localhost:8000/automation/workflows/$WF_ID" \
  -H "Authorization: Bearer $TOKEN"
```

---

## Testing

### Unit-test an agent (mock the LLM)

```python
# tests/test_my_agent.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.orchestrator import run_agent
from app.agents.builder import AgentConfig

def make_config(**kwargs):
    return AgentConfig(**{
        "id": "my_agent", "name": "My Agent",
        "model": "test/model", "max_iterations": 5,
        "tools": ["my_tool"], "token_budget": 10_000,
        **kwargs,
    })

def llm_response(content: str, total_tokens: int = 100):
    r = MagicMock()
    r.choices[0].message.content = content
    r.usage.total_tokens = total_tokens
    r.usage.prompt_tokens = 60
    r.usage.completion_tokens = 40
    return r

@pytest.mark.asyncio
async def test_agent_calls_tool_then_done():
    responses = [
        llm_response('{"tool": "my_tool", "args": {"input_text": "hello"}}'),
        llm_response('{"done": true, "result": "HELLO", "confidence": 0.99}'),
    ]
    from app.agents.tools import ToolResult
    tool_result = ToolResult(tool_name="my_tool", success=True, output={"result": "HELLO"})

    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(side_effect=responses)), \
         patch("app.agents.orchestrator.run_tool", AsyncMock(return_value=tool_result)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config())

    assert result["status"] == "completed"
    assert result["result"] == "HELLO"
    assert result["confidence"] == 0.99
    assert result["iterations"] == 2
```

### Unit-test a tool in isolation

```python
from unittest.mock import patch
from app.my_feature.my_tools import my_tool

def test_my_tool_uppercases_input():
    result = my_tool(input_text="hello")
    assert result == {"result": "HELLO"}
```

### Run the test suite

```bash
uv run python -m pytest tests/ -v
```

### Key patterns from existing tests

| Pattern | How |
|---|---|
| Mock the LLM | `patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(side_effect=[...]))` |
| Mock a tool call | `patch("app.agents.orchestrator.run_tool", AsyncMock(return_value=ToolResult(...)))` |
| Silence context save | `patch("app.agents.orchestrator.save_context")` |
| Simulate tool authorisation block | Give config `tools=["allowed"]` and have LLM call `"other_tool"` |
| Test token budget | Use `token_budget=50` with a response that has `total_tokens=200` |
| Test cycle detection | Set `_invoke_stack` via `token = _invoke_stack.set(["agent-a"]); …; _invoke_stack.reset(token)` |
| SQLite in tests | All DB tests use SQLite in-memory; pgvector tools no-op when `_is_postgres()` is false |
