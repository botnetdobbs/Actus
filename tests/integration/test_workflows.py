from app.agents.builder import AgentConfig
from app.context.models import Workflow, WorkflowStatus
from sqlmodel import Session
from tests.conftest import seed_user, get_token


def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def make_agent_config() -> AgentConfig:
    return AgentConfig(id="test-agent", name="Test Agent", model="ollama/mistral", tools=[])


def seed_workflow(engine, agent_id: str = "test-agent", status: str = "completed") -> Workflow:
    with Session(engine) as session:
        wf = Workflow(name="Test Agent", agent_id=agent_id, status=WorkflowStatus(status))
        session.add(wf)
        session.commit()
        session.refresh(wf)
        return wf


# ── GET /automation/workflows ─────────────────────────────────────────────────

def test_list_workflows_requires_auth(client):
    resp = client.get("/v1/automation/workflows")
    assert resp.status_code == 401


def test_list_workflows_viewer_blocked(client, engine):
    seed_user(engine, "viewer1", "viewer")
    token = get_token(client, "viewer1")
    resp = client.get("/v1/automation/workflows", headers=auth_header(token))
    assert resp.status_code == 403


def test_list_workflows_empty(client, engine):
    seed_user(engine, "analyst1", "analyst")
    token = get_token(client, "analyst1")
    resp = client.get("/v1/automation/workflows", headers=auth_header(token))
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_workflows_returns_all(client, engine):
    seed_user(engine, "analyst2", "analyst")
    seed_workflow(engine, "agent-a", "completed")
    seed_workflow(engine, "agent-b", "failed")
    token = get_token(client, "analyst2")
    resp = client.get("/v1/automation/workflows", headers=auth_header(token))
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_list_workflows_filter_by_agent_id(client, engine):
    seed_user(engine, "analyst3", "analyst")
    seed_workflow(engine, "agent-x", "completed")
    seed_workflow(engine, "agent-y", "completed")
    token = get_token(client, "analyst3")
    resp = client.get("/v1/automation/workflows?agent_id=agent-x", headers=auth_header(token))
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["agent_id"] == "agent-x"


def test_list_workflows_filter_by_status(client, engine):
    seed_user(engine, "analyst4", "analyst")
    seed_workflow(engine, "agent-a", "completed")
    seed_workflow(engine, "agent-a", "failed")
    token = get_token(client, "analyst4")
    resp = client.get("/v1/automation/workflows?status=failed", headers=auth_header(token))
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["status"] == "failed"


# ── GET /automation/workflows/{id} ───────────────────────────────────────────

def test_get_workflow_by_id(client, engine):
    seed_user(engine, "analyst5", "analyst")
    wf = seed_workflow(engine, "agent-z", "running")
    token = get_token(client, "analyst5")
    resp = client.get(f"/v1/automation/workflows/{wf.id}", headers=auth_header(token))
    assert resp.status_code == 200
    assert resp.json()["id"] == wf.id
    assert resp.json()["status"] == "running"


def test_get_workflow_not_found(client, engine):
    seed_user(engine, "analyst6", "analyst")
    token = get_token(client, "analyst6")
    resp = client.get("/v1/automation/workflows/99999", headers=auth_header(token))
    assert resp.status_code == 404


# ── POST /automation/reload ───────────────────────────────────────────────────

def test_reload_requires_admin(client, engine):
    seed_user(engine, "analyst7", "analyst")
    token = get_token(client, "analyst7")
    resp = client.post("/v1/automation/reload", headers=auth_header(token))
    assert resp.status_code == 403


def test_reload_admin_succeeds(client, engine):
    seed_user(engine, "admin1", "admin")
    token = get_token(client, "admin1")
    resp = client.post("/v1/automation/reload", headers=auth_header(token))
    assert resp.status_code == 200
    assert "loaded" in resp.json()
