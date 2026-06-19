# Actus Agent Guide

Everything you need to build, register, invoke, and test agents on this platform.

---

## How an agent run works

1. You trigger an agent (via API call, webhook, or cron schedule). Actus creates a workflow record and runs the agent in the background.
2. If the agent has `rag_query_template` set, Actus retrieves relevant context first and gives it to the agent.
3. The agent then loops: it asks the LLM what to do, calls a tool if needed, looks at the result, and repeats. This continues until the agent reports it's done, or it hits a limit (`max_iterations`, `token_budget`, or a timeout).
4. The final result is saved to the workflow and to the run log. If you're streaming, you see each step (`iteration_start`, `tool_call`, `tool_result`, `done`) as it happens.

You don't need to manage any of this. Your job is to write the agent's YAML config, the tools it can call, and (optionally) a system prompt that tells it how to behave.

---

## YAML agent configuration

Every `.yaml` file in `config/agents/` is loaded at startup and hot-reloadable via `POST /v1/automation/reload`.

```yaml
# Required
id: my_agent                    # used in API calls: POST /v1/automation/trigger/my_agent
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
# Set to "" to disable. Mandatory for document agents, to avoid unrelated pre-loading.
rag_query_template: "invoices for client {client_name}"
rag_top_k: 5

# Optional: override Ollama base URL for this agent
api_base: ""

# Optional: run on a cron schedule (uses APScheduler)
schedule:
  cron: "0 9 * * 1-5"          # weekdays at 09:00

# System prompt: the agent's personality and step instructions
system_prompt: |
  You are a specialist agent. Follow these steps exactly:

  STEP 1: …
  STEP 2: …
  …

  RULES: …
```

Two advanced fields aren't shown above because most agents don't need them: `output_schema` (enforce a JSON Schema on the final result) and `native_tools` (override how tool calls are formatted for the model). See the field reference below.

### Field reference

| Field | Type | Default | Notes |
|---|---|---|---|
| `id` | str | required | URL-safe, unique across all agents |
| `name` | str | required | Display name |
| `description` | str | `""` | Human-readable summary of what the agent does |
| `model` | str | `ollama/mistral` | Any LiteLLM model string |
| `temperature` | float | `0.7` | 0.1 for factual/structured tasks |
| `max_response_tokens` | int | `1024` | Per LLM call; max 8192 |
| `max_iterations` | int | `5` | Set higher for multi-step tasks |
| `token_budget` | int | `10000` | Cumulative; raise for large docs or many steps |
| `tools` | list[str] | `[]` | Only listed tools are callable |
| `rag_query_template` | str | `""` | `""` disables pre-loop RAG entirely |
| `rag_top_k` | int | `5` | Documents retrieved pre-loop |
| `system_prompt` | str | `""` | The agent's instructions; effectively required for any real agent |
| `api_base` | str | `""` | Overrides `OLLAMA_BASE_URL` for this agent |
| `schedule.cron` | str | None | Optional. APScheduler cron expression, see [Scheduling agents](#scheduling-agents) |
| `schedule.misfire_grace_time` | int \| null | null (uses 3600) | How many seconds after a missed fire Actus will still replay it on restart. Set to `0` to never replay. See [Missed fires on restart](#missed-fires-on-restart) |
| `webhook.secret` | str | None | Optional. HMAC-SHA256 secret; enables `POST /v1/automation/webhooks/{id}` |
| `output_schema` | dict (JSON Schema) | None | Optional. If set, the agent's final result must validate against this schema. An invalid result is sent back to the agent for correction (up to 2 retries) |
| `native_tools` | bool | None | Optional. Force native function-calling on (`true`) or off (`false`). By default, non-Ollama models use native tool calling and `ollama/*` models use the JSON protocol described in [The ReAct loop](#the-react-loop) |

To see which agents and tools are currently available, check `config/agents/*.yaml` for agent definitions and each agent's `tools.py` (plus `app/agents/tools.py` for shared tools) for what they can call.

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

The `@tool` decorator registers your function under the given name and builds the description the LLM sees, based on your function's type hints and defaults. Both sync and async functions are supported; sync functions are run in a background thread automatically, so a slow or blocking function won't freeze other agents.

### Tool return values

Return any JSON-serialisable value: `dict`, `list`, `str`, `int`. The orchestrator serialises it with `json.dumps(output, default=str)` and appends it as a user message to the conversation.

On failure, raise an exception. The orchestrator catches it and reports `{"error": "…"}` to the LLM.

### Registering tools at startup

Place your tools in `app/agents/{agent_id}/tools.py` and they are discovered automatically. At startup, `discover_tools()` scans every `app/agents/*/tools.py` and imports it. No changes to `main.py` needed.

The folder name must match the agent `id` in the YAML. For example, an agent with `id: trading_momentum` must live in `app/agents/trading_momentum/`. A mismatch does not prevent the tools from loading, but it breaks the convention and makes the project harder to navigate.

```
app/
└── agents/
    ├── tools.py          ← platform built-ins (semantic_search, invoke_agent, ...)
    ├── doc_qa/           ← folder name matches YAML id: doc_qa
    │   ├── tools.py      ← auto-discovered: @tool decorators registered
    │   ├── router.py     ← auto-registered: must export APIRouter with prefix+tags
    │   └── models.py     ← auto-imported: SQLModel table metadata registered
    └── crm/              ← folder name matches YAML id: crm
        └── tools.py      ← only tools.py needed if no HTTP routes or DB tables
```

There are three convention files, and all of them are optional except `tools.py` for tool-only agents:

| File | Auto-loaded when | Requirement |
|---|---|---|
| `tools.py` | always | export `@tool`-decorated functions |
| `router.py` | if present | export `router = APIRouter(prefix="…", tags=["…"])` |
| `models.py` | if present | define `SQLModel` table classes |

Tools are global. Once registered, any agent that lists the tool name in its `tools:` YAML can call it.

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

Long-running tools inherit the agent's overall timeout (10 minutes) instead of 30 s. Add only tools that genuinely need it. A 30 s limit is a good guardrail against hung tools.

---

## Scheduling agents

Add a `schedule.cron` field to an agent's YAML to run it automatically:

```yaml
schedule:
  cron: "0 9 * * 1-5"   # weekdays at 09:00 UTC
```

This is a standard 5-field crontab expression (minute, hour, day-of-month, month, day-of-week), evaluated in UTC.

A few things to keep in mind for scheduled agents:

- **No input context.** Scheduled runs always start with empty `extra_context`. The agent can't be handed data at trigger time, so it should get everything it needs from its tools, the ontology, or RAG search.
- **No live stream.** There's no workflow to watch and no `/stream` endpoint for scheduled runs. Check the outcome via the audit log (`/v1/automation/runs`).
- **Restart to apply.** Schedules are loaded once when the server starts. Adding or changing `schedule.cron` requires a server restart, not just `POST /v1/automation/reload`.

### Missed fires on restart

When Actus restarts, APScheduler checks whether any scheduled jobs were missed while the server was down. If a missed job falls within `misfire_grace_time` seconds of the current time, APScheduler runs it immediately.

The default is **3600 seconds** (1 hour). This is fine for most agents. A purge job or a weekly report that was missed should usually just run when the server comes back up.

**Set `misfire_grace_time: 0` when replaying a missed run would cause a problem.** For example, a trading agent or an email sender should not run automatically with old data after a restart. Setting it to `0` tells APScheduler to skip the missed run entirely.

```yaml
# Safe to replay: running a purge job late is fine
schedule:
  cron: "0 2 * * *"           # nightly at 02:00 UTC
  # misfire_grace_time not set, so the default of 3600 applies

# Not safe to replay: skip it if the window was missed
schedule:
  cron: "0 15 * * 1-5"        # weekdays at 10:00 ET
  misfire_grace_time: 0        # missed fires are dropped, not replayed
```

| Agent type | Recommended setting | Why |
|---|---|---|
| Purge / housekeeping | omit (default 3600) | Running it late does no harm |
| Reports / summaries | omit or `7200` | Better late than never |
| Trading, billing, notifications | `0` | Running with stale inputs could cause duplicate or unintended actions |

Note: `misfire_grace_time` only controls what happens to missed fires on restart. It does not prevent two runs from overlapping. Overlap is controlled separately by APScheduler's `max_instances` (default: 1 per process).

---

## The ReAct loop

Each iteration:

1. **LLM call** full message history sent to the model, with a timeout
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

## Ontology: domain data

Ontology is Actus's shared store for your domain data (customers, invoices, tickets, whatever your business runs on). It is platform-wide, not specific to any one agent. This is what `semantic_search` and pre-loop RAG search over.

It is defined in `app/ontology/models.py` and is separate from an agent's own `models.py` (see [Three convention files](#registering-tools-at-startup) above). Per-agent `models.py` files are for tables that belong to that agent only (a `doc_qa` session table, for example) and are not searchable by `semantic_search` unless they are also registered as ontology types.

### Defining a new ontology type

```python
# app/ontology/models.py
from sqlmodel import Field
from app.ontology.registry import register
from app.ontology.models import OntologyObjectBase

@register("Invoice")
class Invoice(OntologyObjectBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    number: str = Field(unique=True, index=True)
    client: str
    amount: float
    status: str = "unpaid"
```

`OntologyObjectBase` provides `created_at`, `updated_at`, `created_by`, and soft-delete fields automatically. After adding a type, generate and run a migration:

```bash
make migrations msg="add_invoice"
make migrate
```

### CRUD API

Every registered type gets full CRUD endpoints automatically:

| Endpoint | Notes |
|---|---|
| `GET /v1/ontology/types` | List all registered type names |
| `GET /v1/ontology/objects/{type_name}` | List objects (paginated) |
| `GET /v1/ontology/objects/{type_name}/{id}` | Get one object |
| `POST /v1/ontology/objects/{type_name}` | Create. Auth required |
| `PUT /v1/ontology/objects/{type_name}/{id}` | Update. Auth required |
| `DELETE /v1/ontology/objects/{type_name}/{id}` | Soft delete. Auth required |

Create, update, and delete automatically re-index the object into pgvector in the background, so it is immediately searchable via `semantic_search` and pre-loop RAG.

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
curl -X POST http://localhost:8000/v1/automation/trigger/my_agent \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"extra_context": {"key": "value"}}'
```

Response: `{"status": "queued", "agent_id": "my_agent", "workflow_id": 42}`

Streaming and polling are optional. Many agents do their real work through their tools (sending an email, posting to Slack, updating a record) and the caller doesn't need to wait around for a "result" at all. This is the normal pattern for webhook-triggered and scheduled agents: trigger it and move on, and check `/v1/automation/runs` later if you need to confirm what happened. Use streaming or polling when a user or another system is waiting on the agent's answer.

### Stream the result (preferred)

```bash
curl -N "http://localhost:8000/v1/automation/workflows/42/stream" \
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

The stream is reconnectable. Disconnect and re-open with the same `workflow_id` at any time.

### Poll for result (alternative)

```bash
watch -n 2 curl -s "http://localhost:8000/v1/automation/workflows/42" \
  -H "Authorization: Bearer $TOKEN"
```

Refreshes every 2 seconds. `status` progresses: `pending` → `running` → `completed` / `failed` / `timeout`

`result_json` is the serialised agent result dict, including `result`, `confidence`, `iterations`, `tool_calls`, and token counts.

### Webhook trigger (external systems)

Enable a webhook on the agent by adding a `webhook.secret` to its YAML:

```yaml
# config/agents/lead_qualifier.yaml
id: lead_qualifier
webhook:
  secret: "generate-with-openssl-rand-hex-32"
tools:
  - qualify_lead
system_prompt: |
  You receive a lead from the CRM in context["email"] and context["plan"].
  Score it 1-10 and call qualify_lead with the score.
```

The external system signs the request body and sends it to `POST /v1/automation/webhooks/{agent_id}`:

```bash
BODY='{"email": "alice@example.com", "plan": "enterprise"}'
SECRET="generate-with-openssl-rand-hex-32"
SIG="sha256=$(echo -n "$BODY" | openssl dgst -sha256 -hmac "$SECRET" | awk '{print $2}')"

curl -X POST http://localhost:8000/v1/automation/webhooks/lead_qualifier \
  -H "Content-Type: application/json" \
  -H "X-Actus-Signature: $SIG" \
  -d "$BODY"
# → {"status": "queued", "agent_id": "lead_qualifier", "workflow_id": 7}
```

The JSON body becomes `extra_context` directly: the agent reads it via `context["email"]`, `context["plan"]`, etc. Arrays are wrapped as `{"payload": [...]}`. Non-JSON bodies are wrapped as `{"raw": "..."}`.

**GitHub webhooks** send `X-Hub-Signature-256`. Actus accepts it without any configuration. Point the GitHub webhook URL at `/v1/automation/webhooks/{agent_id}` and use the same secret.

No JWT is needed. The HMAC signature is the authentication mechanism.

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
| **Sequential pipeline** | `invoke_agent` | Agents must run in order, output of one feeds the next |
| **Parallel execution** | `invoke_agents_parallel` | Agents are independent, run concurrently and merge results |
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

**Cycle detection**: `invoke_agent` tracks the invocation stack. If agent A calls agent B which calls agent A, the second call is rejected with `"Circular invocation detected"`. Max depth is 5.

**Parallel cap**: `invoke_agents_parallel` rejects batches larger than 10 with a single error result rather than N identical errors.

---

## Timeouts reference

| Timeout | Value | What it covers |
|---|---|---|
| Total agent run | 10 minutes | The whole agent run, from first iteration to final result |
| Each LLM call | 120 s | A single request to the model |
| Default tool timeout | 30 s | Any tool not marked as long-running |
| Long-running tool timeout | 10 minutes | Tools marked long-running (see [Long-running tools](#long-running-tools)) |

If an agent hits the total run timeout, the workflow is marked as failed with a timeout error.

---

## Audit logging

Every agent run (success, error, timeout) writes a row to the audit log. The record contains:

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

## Building a new agent, step by step

### 1. Create the agent module

```
app/agents/my_agent/
    __init__.py
    tools.py          ← required; auto-discovered
    router.py         ← optional; auto-registered if present
    models.py         ← optional; auto-imported if present
```

```python
# app/agents/my_agent/tools.py
from app.agents.tools import tool

@tool("my_tool", "Does the thing. Returns {result: str}.")
def my_tool(input_text: str) -> dict:
    return {"result": input_text.upper()}
```

### 2. No registration needed — files are discovered automatically

`discover_tools()`, `discover_routers()`, and `discover_models()` scan `app/agents/*/` at startup. No changes to `main.py`.

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

Uvicorn runs with `--reload` in Docker. The tool module import is picked up on reload. The YAML is volume-mounted and re-read on `POST /v1/automation/reload` or on the next server restart.

Exception: if your YAML sets `schedule.cron`, a restart is required for the schedule itself to take effect (see [Scheduling agents](#scheduling-agents)).

### 6. Trigger and verify

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/v1/auth/login \
  -d "username=alice&password=changeme123" | jq -r .access_token)

WF_ID=$(curl -s -X POST http://localhost:8000/v1/automation/trigger/my_agent \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"extra_context": {"input": "hello world"}}' | jq -r .workflow_id)

# Stream live progress (recommended)
curl -N "http://localhost:8000/v1/automation/workflows/$WF_ID/stream" \
  -H "Authorization: Bearer $TOKEN"

# Or watch the poll endpoint (alternative)
watch -n 2 curl -s "http://localhost:8000/v1/automation/workflows/$WF_ID" \
  -H "Authorization: Bearer $TOKEN"
```

---

## Testing

### Unit-test an agent (mock the LLM)

```python
# tests/unit/test_my_agent.py
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
| SQLite in tests | All DB tests use SQLite in-memory; pgvector-backed features (RAG, semantic search) no-op on SQLite |
