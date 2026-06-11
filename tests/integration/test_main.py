"""
Tests for app/main.py: scheduler startup wiring and the /healthz endpoint.
"""
from unittest.mock import AsyncMock, MagicMock, patch
from apscheduler.schedulers.base import STATE_RUNNING
from fastapi.testclient import TestClient
from sqlmodel import Session
from app.main import create_app
from app.database import get_session


# ── fixtures ──────────────────────────────────────────────────────────────────
# `engine` and `client` fixtures are provided by tests/conftest.py


def test_scheduler_not_started_when_disabled(engine):
    def override_session():
        with Session(engine) as session:
            yield session

    with (
        patch("app.main.instrument_app"),
        patch("app.main._settings") as mock_settings,
    ):
        mock_settings.scheduler_enabled = False
        mock_settings.debug = True
        mock_settings.cors_origins = ["*"]
        mock_settings.app_name = "Actus"
        mock_settings.app_version = "0.1.0"
        mock_settings.redis_url = ""
        mock_settings.ollama_base_url = "http://localhost:11434"
        with patch("app.automation.scheduler.start_scheduler") as mock_start:
            application = create_app()
            application.dependency_overrides[get_session] = override_session
            with TestClient(application, raise_server_exceptions=False):
                pass
            mock_start.assert_not_called()


def test_healthz_returns_200_when_ollama_unreachable(client):
    with (
        patch("httpx.AsyncClient") as mock_client_class,
        patch("app.automation.scheduler.scheduler.state", new=STATE_RUNNING),
    ):
        mock_ac = AsyncMock()
        mock_ac.__aenter__ = AsyncMock(return_value=mock_ac)
        mock_ac.__aexit__ = AsyncMock(return_value=False)
        mock_ac.get = AsyncMock(side_effect=Exception("connection refused"))
        mock_client_class.return_value = mock_ac

        resp = client.get("/healthz")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["core"]["database"] == "ok"
    assert body["info"]["ollama"] == "unreachable"


def test_healthz_returns_503_when_db_fails(client):
    with (
        patch("httpx.AsyncClient") as mock_client_class,
        patch("app.main.select", side_effect=Exception("DB down")),
    ):
        mock_ac = AsyncMock()
        mock_ac.__aenter__ = AsyncMock(return_value=mock_ac)
        mock_ac.__aexit__ = AsyncMock(return_value=False)
        mock_r = MagicMock()
        mock_r.status_code = 200
        mock_ac.get = AsyncMock(return_value=mock_r)
        mock_client_class.return_value = mock_ac

        resp = client.get("/healthz")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert "error" in body["core"]["database"]


def test_healthz_response_has_core_and_info_keys(client):
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_ac = AsyncMock()
        mock_ac.__aenter__ = AsyncMock(return_value=mock_ac)
        mock_ac.__aexit__ = AsyncMock(return_value=False)
        mock_r = MagicMock()
        mock_r.status_code = 200
        mock_ac.get = AsyncMock(return_value=mock_r)
        mock_client_class.return_value = mock_ac

        resp = client.get("/healthz")

    body = resp.json()
    assert "core" in body
    assert "info" in body
    assert "status" in body


def test_healthz_reports_redis_not_configured_when_absent(client):
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_ac = AsyncMock()
        mock_ac.__aenter__ = AsyncMock(return_value=mock_ac)
        mock_ac.__aexit__ = AsyncMock(return_value=False)
        mock_ac.get = AsyncMock(side_effect=Exception("ollama not running"))
        mock_client_class.return_value = mock_ac

        with patch("app.pubsub._redis", None):
            resp = client.get("/healthz")

    assert resp.json()["info"]["redis"] == "not_configured"


def test_healthz_reports_redis_ok_when_available(client):
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_ac = AsyncMock()
        mock_ac.__aenter__ = AsyncMock(return_value=mock_ac)
        mock_ac.__aexit__ = AsyncMock(return_value=False)
        mock_ac.get = AsyncMock(side_effect=Exception("ollama not running"))
        mock_client_class.return_value = mock_ac

        with patch("app.pubsub._redis", mock_redis):
            resp = client.get("/healthz")

    assert resp.json()["info"]["redis"] == "ok"


def test_healthz_reports_redis_error_on_ping_failure(client):
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(side_effect=Exception("redis timeout"))

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_ac = AsyncMock()
        mock_ac.__aenter__ = AsyncMock(return_value=mock_ac)
        mock_ac.__aexit__ = AsyncMock(return_value=False)
        mock_ac.get = AsyncMock(side_effect=Exception("ollama not running"))
        mock_client_class.return_value = mock_ac

        with patch("app.pubsub._redis", mock_redis):
            resp = client.get("/healthz")

    assert resp.json()["info"]["redis"] == "error"
