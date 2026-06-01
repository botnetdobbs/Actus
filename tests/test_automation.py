import pytest
from unittest.mock import AsyncMock, patch
from app.agents.builder import AgentConfig
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


def test_trigger_admin_can_trigger(client, engine):
    seed_user(engine, "admin1", "admin")
    token = get_token(client, "admin1")
    with patch("app.automation.router.get_agent", return_value=make_agent_config()), \
         patch("app.automation.router._run_workflow", new=AsyncMock(return_value=None)):
        resp = client.post("/automation/trigger/test-agent", headers=auth_header(token))
    assert resp.status_code == 202
