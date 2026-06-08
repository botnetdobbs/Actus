import asyncio
import hashlib
import hmac
import uuid
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from sqlmodel import Session
from app.agents.audit import AgentRunLog
from app.agents.builder import AgentConfig, WebhookConfig
from app.context.models import Workflow, WorkflowStatus
from tests.conftest import seed_user, get_token


def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def make_agent_config() -> AgentConfig:
    return AgentConfig(
        id="test-agent",
        name="Test Agent",
        model="ollama/mistral",
        tools=[],
    )


# ── POST /automation/trigger/{agent_id} ───────────────────────────────────────

def test_trigger_requires_auth(client):
    resp = client.post("/automation/trigger/test-agent")
    assert resp.status_code == 401


def test_trigger_viewer_blocked(client, engine):
    seed_user(engine, "viewer1", "viewer")
    token = get_token(client, "viewer1")
    resp = client.post("/automation/trigger/test-agent", headers=auth_header(token))
    assert resp.status_code == 403


def test_trigger_unknown_agent_returns_404(client, engine):
    seed_user(engine, "analyst1", "analyst")
    token = get_token(client, "analyst1")
    with patch("app.automation.router.get_agent", side_effect=KeyError("test-agent")):
        resp = client.post("/automation/trigger/missing-agent", headers=auth_header(token))
    assert resp.status_code == 404


def test_trigger_analyst_queues_agent(client, engine):
    seed_user(engine, "analyst2", "analyst")
    token = get_token(client, "analyst2")
    with patch("app.automation.router.get_agent", return_value=make_agent_config()), \
         patch("app.automation.router._run_workflow", new=AsyncMock(return_value=None)):
        resp = client.post("/automation/trigger/test-agent", headers=auth_header(token))
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert body["agent_id"] == "test-agent"
    assert "workflow_id" in body


def test_outcome_map_completed_becomes_success():
    from app.automation.router import _OUTCOME_MAP
    assert _OUTCOME_MAP["completed"] == "success"
    assert _OUTCOME_MAP["incomplete"] == "incomplete"
    assert _OUTCOME_MAP["error"] == "error"
    assert _OUTCOME_MAP["timeout"] == "timeout"


def test_trigger_admin_can_trigger(client, engine):
    seed_user(engine, "admin1", "admin")
    token = get_token(client, "admin1")
    with patch("app.automation.router.get_agent", return_value=make_agent_config()), \
         patch("app.automation.router._run_workflow", new=AsyncMock(return_value=None)):
        resp = client.post("/automation/trigger/test-agent", headers=auth_header(token))
    assert resp.status_code == 202


# ── SSE streaming helpers ──────────────────────────────────────────────────────

def seed_workflow(engine, agent_id: str, status: str) -> int:
    with Session(engine) as session:
        wf = Workflow(name="Test", agent_id=agent_id,
                      status=WorkflowStatus(status), created_by=None)
        session.add(wf)
        session.commit()
        session.refresh(wf)
        return wf.id


# ── GET /automation/workflows/{id}/stream ─────────────────────────────────────

def test_stream_requires_auth(client):
    resp = client.get("/automation/workflows/999/stream")
    assert resp.status_code == 401


def test_stream_viewer_blocked(client, engine):
    seed_user(engine, "vstream", "viewer")
    token = get_token(client, "vstream")
    resp = client.get("/automation/workflows/999/stream", headers=auth_header(token))
    assert resp.status_code == 403


def test_stream_not_found(client, engine):
    seed_user(engine, "astream_nf", "analyst")
    token = get_token(client, "astream_nf")
    with client.stream("GET", "/automation/workflows/99999/stream",
                       headers=auth_header(token)) as resp:
        body = resp.read().decode()
    assert '"type": "error"' in body


def test_stream_already_completed(client, engine):
    seed_user(engine, "astream_done", "analyst")
    token = get_token(client, "astream_done")
    wf_id = seed_workflow(engine, "test-agent", "completed")
    with client.stream("GET", f"/automation/workflows/{wf_id}/stream",
                       headers=auth_header(token)) as resp:
        body = resp.read().decode()
    assert '"type": "status"' in body
    assert '"status": "completed"' in body
    # full WorkflowResponse emitted immediately — no polling
    assert f'"id":{wf_id}' in body


def test_stream_live_queue_emits_per_iteration_events(client, engine):
    from app.automation.router import _run_queues

    seed_user(engine, "astream_live", "analyst")
    token = get_token(client, "astream_live")
    wf_id = seed_workflow(engine, "test-agent", "running")

    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait({"type": "iteration_start", "run_id": "r1", "iteration": 0})
    queue.put_nowait({"type": "tool_call", "run_id": "r1", "iteration": 0,
                      "tool": "my_tool", "args": {}})
    queue.put_nowait({"type": "tool_result", "run_id": "r1", "iteration": 0,
                      "tool": "my_tool", "success": True, "preview": "{}"})
    queue.put_nowait({"type": "done", "run_id": "r1", "status": "completed",
                      "result": "all good", "iterations": 1, "total_tokens": 100})
    queue.put_nowait(None)  # sentinel
    _run_queues[wf_id] = queue

    try:
        with client.stream("GET", f"/automation/workflows/{wf_id}/stream",
                           headers=auth_header(token)) as resp:
            body = resp.read().decode()
    finally:
        _run_queues.pop(wf_id, None)

    assert "iteration_start" in body
    assert "tool_call" in body
    assert "tool_result" in body
    assert '"type": "done"' in body
    assert '"status": "completed"' in body


def test_stream_db_poll_fallback_detects_transition(client, engine):
    """DB polling fallback: detects running→completed transition and emits full payload."""
    from unittest.mock import patch, AsyncMock
    from app.automation.router import _run_queues

    seed_user(engine, "astream_poll", "analyst")
    token = get_token(client, "astream_poll")
    wf_id = seed_workflow(engine, "test-agent", "running")
    assert wf_id not in _run_queues

    flipped = [False]

    async def flip_on_first_sleep(*_args):
        if not flipped[0]:
            flipped[0] = True
            with Session(engine) as session:
                wf = session.get(Workflow, wf_id)
                wf.status = WorkflowStatus.completed
                session.add(wf)
                session.commit()

    with patch("app.automation.router.asyncio.sleep", side_effect=flip_on_first_sleep):
        with client.stream("GET", f"/automation/workflows/{wf_id}/stream",
                           headers=auth_header(token)) as resp:
            body = resp.read().decode()

    assert '"type": "status"' in body
    assert '"status": "running"' in body          # initial event (f-string template, with spaces)
    assert '"status":"completed"' in body          # WorkflowResponse payload (model_dump_json, no spaces)


# ── GET /automation/runs helpers ──────────────────────────────────────────────

def seed_run_log(
    engine,
    *,
    agent_id: str = "test-agent",
    outcome: str = "success",
    triggered_by: int | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    tool_calls: str = '[{"tool": "my_tool", "success": true}]',
) -> str:
    run_id = str(uuid.uuid4())
    started = started_at or datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    completed = completed_at or datetime(2026, 1, 1, 10, 0, 30, tzinfo=timezone.utc)
    with Session(engine) as session:
        entry = AgentRunLog(
            run_id=run_id,
            agent_id=agent_id,
            triggered_by=triggered_by,
            started_at=started,
            completed_at=completed,
            model="ollama/mistral",
            pii_detected=False,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            tool_calls=tool_calls,
            outcome=outcome,
            result_summary="Test result summary",
        )
        session.add(entry)
        session.commit()
    return run_id


# ── GET /automation/runs ──────────────────────────────────────────────────────

def test_list_runs_requires_auth(client):
    assert client.get("/automation/runs").status_code == 401


def test_list_runs_viewer_blocked(client, engine):
    seed_user(engine, "vr_runs", "viewer")
    token = get_token(client, "vr_runs")
    assert client.get("/automation/runs", headers=auth_header(token)).status_code == 403


def test_list_runs_empty(client, engine):
    seed_user(engine, "ar_empty", "analyst")
    token = get_token(client, "ar_empty")
    resp = client.get("/automation/runs", headers=auth_header(token))
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_runs_returns_seeded_run(client, engine):
    seed_user(engine, "ar_basic", "analyst")
    token = get_token(client, "ar_basic")
    run_id = seed_run_log(engine, agent_id="my-agent", outcome="success")
    resp = client.get("/automation/runs", headers=auth_header(token))
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["run_id"] == run_id
    assert data[0]["agent_id"] == "my-agent"
    assert data[0]["outcome"] == "success"


def test_list_runs_filter_by_agent_id(client, engine):
    seed_user(engine, "ar_agent", "analyst")
    token = get_token(client, "ar_agent")
    seed_run_log(engine, agent_id="agent-a")
    seed_run_log(engine, agent_id="agent-b")
    resp = client.get("/automation/runs?agent_id=agent-a", headers=auth_header(token))
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["agent_id"] == "agent-a"


def test_list_runs_filter_by_outcome(client, engine):
    seed_user(engine, "ar_outcome", "analyst")
    token = get_token(client, "ar_outcome")
    seed_run_log(engine, outcome="success")
    seed_run_log(engine, outcome="error")
    resp = client.get("/automation/runs?outcome=error", headers=auth_header(token))
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["outcome"] == "error"


def test_list_runs_pagination(client, engine):
    seed_user(engine, "ar_page", "analyst")
    token = get_token(client, "ar_page")
    for _ in range(5):
        seed_run_log(engine)
    resp_all = client.get("/automation/runs?limit=5", headers=auth_header(token))
    assert len(resp_all.json()) == 5
    resp_page = client.get("/automation/runs?limit=2&offset=0", headers=auth_header(token))
    assert len(resp_page.json()) == 2
    resp_offset = client.get("/automation/runs?limit=2&offset=4", headers=auth_header(token))
    assert len(resp_offset.json()) == 1


def test_list_runs_tool_calls_parsed_as_list(client, engine):
    seed_user(engine, "ar_tools", "analyst")
    token = get_token(client, "ar_tools")
    seed_run_log(engine, tool_calls='[{"tool": "search_document", "success": true}]')
    resp = client.get("/automation/runs", headers=auth_header(token))
    assert resp.status_code == 200
    tool_calls = resp.json()[0]["tool_calls"]
    assert isinstance(tool_calls, list)
    assert tool_calls[0]["tool"] == "search_document"


def test_list_runs_tool_calls_malformed_json_returns_empty_list(client, engine):
    seed_user(engine, "ar_badtools", "analyst")
    token = get_token(client, "ar_badtools")
    seed_run_log(engine, tool_calls="not-valid-json")
    resp = client.get("/automation/runs", headers=auth_header(token))
    assert resp.status_code == 200
    assert resp.json()[0]["tool_calls"] == []


def test_list_runs_duration_seconds_computed(client, engine):
    seed_user(engine, "ar_dur", "analyst")
    token = get_token(client, "ar_dur")
    started = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    completed = datetime(2026, 1, 1, 10, 0, 45, tzinfo=timezone.utc)
    seed_run_log(engine, started_at=started, completed_at=completed)
    resp = client.get("/automation/runs", headers=auth_header(token))
    assert resp.status_code == 200
    assert resp.json()[0]["duration_seconds"] == 45.0


def test_list_runs_date_range_filter(client, engine):
    seed_user(engine, "ar_date", "analyst")
    token = get_token(client, "ar_date")
    seed_run_log(engine, started_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    seed_run_log(engine, started_at=datetime(2026, 3, 1, tzinfo=timezone.utc))
    resp = client.get(
        "/automation/runs?from_date=2026-02-01T00:00:00Z&to_date=2026-04-01T00:00:00Z",
        headers=auth_header(token),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert "2026-03" in data[0]["started_at"]


# ── GET /automation/runs/{run_id} ─────────────────────────────────────────────

def test_get_run_requires_auth(client):
    assert client.get("/automation/runs/some-run-id").status_code == 401


def test_get_run_viewer_blocked(client, engine):
    seed_user(engine, "vr_run_id", "viewer")
    token = get_token(client, "vr_run_id")
    assert client.get("/automation/runs/x", headers=auth_header(token)).status_code == 403


def test_get_run_not_found(client, engine):
    seed_user(engine, "ar_nf", "analyst")
    token = get_token(client, "ar_nf")
    resp = client.get(f"/automation/runs/{uuid.uuid4()}", headers=auth_header(token))
    assert resp.status_code == 404


def test_get_run_returns_correct_run(client, engine):
    seed_user(engine, "ar_get", "analyst")
    token = get_token(client, "ar_get")
    run_id = seed_run_log(engine, agent_id="target-agent", outcome="timeout")
    resp = client.get(f"/automation/runs/{run_id}", headers=auth_header(token))
    assert resp.status_code == 200
    data = resp.json()
    assert data["run_id"] == run_id
    assert data["agent_id"] == "target-agent"
    assert data["outcome"] == "timeout"
    assert isinstance(data["tool_calls"], list)


# ── POST /automation/webhooks/{agent_id} ──────────────────────────────────────

_WEBHOOK_SECRET = "test-webhook-secret"


def make_agent_with_webhook() -> AgentConfig:
    return AgentConfig(
        id="test-agent", name="Test Agent", model="ollama/mistral",
        tools=[], webhook=WebhookConfig(secret=_WEBHOOK_SECRET),
    )


def make_agent_no_webhook() -> AgentConfig:
    return AgentConfig(id="test-agent", name="Test Agent", model="ollama/mistral", tools=[])


def sign(body: bytes, secret: str = _WEBHOOK_SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_webhook_agent_not_found(client):
    with patch("app.automation.router.get_agent", side_effect=KeyError("no-agent")):
        resp = client.post("/automation/webhooks/no-agent", content=b"{}")
    assert resp.status_code == 404


def test_webhook_not_enabled(client):
    with patch("app.automation.router.get_agent", return_value=make_agent_no_webhook()):
        resp = client.post("/automation/webhooks/test-agent", content=b"{}")
    assert resp.status_code == 403


def test_webhook_missing_signature(client):
    with patch("app.automation.router.get_agent", return_value=make_agent_with_webhook()):
        resp = client.post("/automation/webhooks/test-agent", content=b'{"x": 1}')
    assert resp.status_code == 401
    assert "Missing" in resp.json()["detail"]


def test_webhook_invalid_signature(client):
    body = b'{"x": 1}'
    with patch("app.automation.router.get_agent", return_value=make_agent_with_webhook()):
        resp = client.post(
            "/automation/webhooks/test-agent",
            content=body,
            headers={"X-Actus-Signature": "sha256=badhash"},
        )
    assert resp.status_code == 401
    assert "Invalid" in resp.json()["detail"]


def test_webhook_valid_queues_run(client, engine):
    body = b'{"event": "payment.failed"}'
    sig = sign(body)
    with patch("app.automation.router.get_agent", return_value=make_agent_with_webhook()), \
         patch("app.automation.router._run_workflow", new=AsyncMock(return_value=None)):
        resp = client.post(
            "/automation/webhooks/test-agent",
            content=body,
            headers={"X-Actus-Signature": sig},
        )
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "queued"
    assert data["agent_id"] == "test-agent"
    assert "workflow_id" in data


def test_webhook_json_body_becomes_extra_context(client, engine):
    body = b'{"customer_id": 42, "plan": "enterprise"}'
    sig = sign(body)
    with patch("app.automation.router.get_agent", return_value=make_agent_with_webhook()), \
         patch("app.automation.router._run_workflow", new=AsyncMock(return_value=None)):
        resp = client.post(
            "/automation/webhooks/test-agent",
            content=body,
            headers={"X-Actus-Signature": sig},
        )
    assert resp.status_code == 202
    wf_id = resp.json()["workflow_id"]
    with Session(engine) as session:
        wf = session.get(Workflow, wf_id)
        import json as _json
        ctx = _json.loads(wf.extra_context_json)
    assert ctx == {"customer_id": 42, "plan": "enterprise"}


def test_webhook_non_dict_json_wrapped(client, engine):
    body = b'[1, 2, 3]'
    sig = sign(body)
    with patch("app.automation.router.get_agent", return_value=make_agent_with_webhook()), \
         patch("app.automation.router._run_workflow", new=AsyncMock(return_value=None)):
        resp = client.post(
            "/automation/webhooks/test-agent",
            content=body,
            headers={"X-Actus-Signature": sig},
        )
    assert resp.status_code == 202
    wf_id = resp.json()["workflow_id"]
    with Session(engine) as session:
        wf = session.get(Workflow, wf_id)
        import json as _json
        ctx = _json.loads(wf.extra_context_json)
    assert ctx == {"payload": [1, 2, 3]}


def test_webhook_non_json_body_wrapped(client, engine):
    body = b'plain text payload'
    sig = sign(body)
    with patch("app.automation.router.get_agent", return_value=make_agent_with_webhook()), \
         patch("app.automation.router._run_workflow", new=AsyncMock(return_value=None)):
        resp = client.post(
            "/automation/webhooks/test-agent",
            content=body,
            headers={"X-Actus-Signature": sig},
        )
    assert resp.status_code == 202
    wf_id = resp.json()["workflow_id"]
    with Session(engine) as session:
        wf = session.get(Workflow, wf_id)
        import json as _json
        ctx = _json.loads(wf.extra_context_json)
    assert ctx == {"raw": "plain text payload"}


def test_webhook_oversized_body_rejected(client):
    body = b"x" * (1024 * 1024 + 1)
    sig = sign(body)
    with patch("app.automation.router.get_agent", return_value=make_agent_with_webhook()):
        resp = client.post(
            "/automation/webhooks/test-agent",
            content=body,
            headers={"X-Actus-Signature": sig},
        )
    assert resp.status_code == 413


def test_webhook_github_header_accepted(client, engine):
    body = b'{"action": "opened"}'
    sig = sign(body)
    with patch("app.automation.router.get_agent", return_value=make_agent_with_webhook()), \
         patch("app.automation.router._run_workflow", new=AsyncMock(return_value=None)):
        resp = client.post(
            "/automation/webhooks/test-agent",
            content=body,
            headers={"X-Hub-Signature-256": sig},   # GitHub's header name
        )
    assert resp.status_code == 202
