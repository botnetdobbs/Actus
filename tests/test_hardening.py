"""
Tests for production hardening fixes:
  Fix B  — rate limiter Redis storage
  Fix C  — scheduler_enabled knob
  Fix E  — /healthz: Ollama down does not cause 503
  Fix F  — /healthz: Redis status reported in info block
  Fix I  — JWT refresh token flow
  Fix J  — token revocation via jti + Redis blocklist
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool
import app.database as db_module
from app.main import create_app
from app.database import get_session
from tests.conftest import seed_user, get_token


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def engine():
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    import app.auth.models
    import app.ontology.models
    import app.context.store
    import app.context.models
    import app.agents.audit
    import app.rag.models
    SQLModel.metadata.create_all(test_engine)

    original = db_module._engine
    db_module._engine = test_engine
    yield test_engine
    db_module._engine = original
    SQLModel.metadata.drop_all(test_engine)


@pytest.fixture()
def client(engine):
    def override_session():
        with Session(engine) as session:
            yield session

    with patch("app.main.instrument_app"):
        application = create_app()
    application.dependency_overrides[get_session] = override_session

    with TestClient(application, raise_server_exceptions=False) as c:
        yield c

    application.dependency_overrides.clear()


def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── Fix B: Rate limiter storage ───────────────────────────────────────────────

def test_rate_limiter_falls_back_to_memory_when_no_redis():
    from limits.storage import MemoryStorage
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    l = Limiter(key_func=get_remote_address, storage_uri="memory://")
    assert isinstance(l._storage, MemoryStorage)


def test_rate_limiter_uses_redis_storage_when_url_set():
    from limits.storage import MemoryStorage
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    l = Limiter(key_func=get_remote_address, storage_uri="redis://localhost:6379")
    assert not isinstance(l._storage, MemoryStorage)


# ── Fix C: scheduler_enabled ──────────────────────────────────────────────────

def test_scheduler_enabled_defaults_to_true():
    from app.config import get_settings
    get_settings.cache_clear()
    try:
        s = get_settings()
        assert s.scheduler_enabled is True
    finally:
        get_settings.cache_clear()


def test_scheduler_enabled_can_be_disabled():
    from app.config import get_settings, Settings
    get_settings.cache_clear()
    try:
        with patch.dict("os.environ", {"SCHEDULER_ENABLED": "false", "DEBUG": "true"}):
            s = Settings()
            assert s.scheduler_enabled is False
    finally:
        get_settings.cache_clear()


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


# ── Fix E: /healthz — Ollama does not block core health ──────────────────────

def test_healthz_returns_200_when_ollama_unreachable(client):
    with patch("httpx.AsyncClient") as mock_client_class:
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


# ── Fix F: /healthz — Redis reported in info ──────────────────────────────────

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


# ── Fix I: JWT refresh token flow ─────────────────────────────────────────────

def test_login_returns_refresh_token(client, engine):
    seed_user(engine, "alice_refresh", "viewer")
    resp = client.post("/v1/auth/login", data={"username": "alice_refresh", "password": "testpass"})
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert "refresh_token" in body
    assert body["token_type"] == "bearer"


def test_refresh_endpoint_returns_new_tokens(client, engine):
    seed_user(engine, "bob_refresh", "viewer")
    login_resp = client.post("/v1/auth/login", data={"username": "bob_refresh", "password": "testpass"})
    refresh_token = login_resp.json()["refresh_token"]

    resp = client.post("/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert "refresh_token" in body


def test_refresh_new_access_token_is_valid(client, engine):
    seed_user(engine, "carol_refresh", "viewer")
    login_resp = client.post("/v1/auth/login", data={"username": "carol_refresh", "password": "testpass"})
    refresh_token = login_resp.json()["refresh_token"]

    refresh_resp = client.post("/v1/auth/refresh", json={"refresh_token": refresh_token})
    new_access = refresh_resp.json()["access_token"]

    me_resp = client.get("/v1/auth/me", headers=auth_header(new_access))
    assert me_resp.status_code == 200
    assert me_resp.json()["username"] == "carol_refresh"


def test_refresh_rejects_access_token_as_refresh(client, engine):
    seed_user(engine, "dave_refresh", "viewer")
    login_resp = client.post("/v1/auth/login", data={"username": "dave_refresh", "password": "testpass"})
    access_token = login_resp.json()["access_token"]

    resp = client.post("/v1/auth/refresh", json={"refresh_token": access_token})
    assert resp.status_code == 401


def test_refresh_rejects_expired_or_invalid_token(client):
    resp = client.post("/v1/auth/refresh", json={"refresh_token": "not.a.real.token"})
    assert resp.status_code == 401


def test_refresh_rejects_inactive_user(client, engine):
    from app.auth.models import User
    from sqlmodel import select

    seed_user(engine, "eve_refresh", "viewer")
    login_resp = client.post("/v1/auth/login", data={"username": "eve_refresh", "password": "testpass"})
    refresh_token = login_resp.json()["refresh_token"]

    with Session(engine) as session:
        u = session.exec(select(User).where(User.username == "eve_refresh")).first()
        u.is_active = False
        session.add(u)
        session.commit()

    resp = client.post("/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert resp.status_code == 401


# ── Fix J: Token revocation via jti + Redis blocklist ────────────────────────

def _make_mock_redis() -> tuple[MagicMock, set]:
    """Return (mock_redis, revoked_set) — setex adds to set, exists checks it."""
    revoked: set[str] = set()

    mock_redis = MagicMock()

    async def _setex(key: str, ttl: int, val: str) -> None:
        revoked.add(key)

    async def _exists(key: str) -> int:
        return 1 if key in revoked else 0

    mock_redis.setex = _setex
    mock_redis.exists = _exists
    return mock_redis, revoked


def test_logout_requires_auth(client):
    resp = client.post("/v1/auth/logout")
    assert resp.status_code == 401


def test_logout_returns_204(client, engine):
    seed_user(engine, "frank_logout", "viewer")
    token = get_token(client, "frank_logout")

    resp = client.post("/v1/auth/logout", headers=auth_header(token))
    assert resp.status_code == 204


def test_logout_revokes_token_when_redis_available(client, engine):
    seed_user(engine, "grace_logout", "viewer")
    token = get_token(client, "grace_logout")

    mock_redis, revoked = _make_mock_redis()
    with patch("app.pubsub._redis", mock_redis):
        resp = client.post("/v1/auth/logout", headers=auth_header(token))
        assert resp.status_code == 204

        resp = client.get("/v1/auth/me", headers=auth_header(token))
        assert resp.status_code == 401


def test_logout_no_redis_returns_204_without_revoking(client, engine):
    seed_user(engine, "henry_logout", "viewer")
    token = get_token(client, "henry_logout")

    with patch("app.pubsub._redis", None):
        resp = client.post("/v1/auth/logout", headers=auth_header(token))
        assert resp.status_code == 204

    resp = client.get("/v1/auth/me", headers=auth_header(token))
    assert resp.status_code == 200


def test_refresh_rotation_revokes_old_refresh_token(client, engine):
    seed_user(engine, "ivan_refresh", "viewer")
    login_resp = client.post("/v1/auth/login", data={"username": "ivan_refresh", "password": "testpass"})
    old_refresh = login_resp.json()["refresh_token"]

    mock_redis, revoked = _make_mock_redis()
    with patch("app.pubsub._redis", mock_redis):
        # Use old refresh token to get new tokens — rotation revokes old token
        resp = client.post("/v1/auth/refresh", json={"refresh_token": old_refresh})
        assert resp.status_code == 200

        # Old refresh token should now be revoked
        resp = client.post("/v1/auth/refresh", json={"refresh_token": old_refresh})
        assert resp.status_code == 401


def test_new_refresh_token_still_works_after_rotation(client, engine):
    seed_user(engine, "judy_refresh", "viewer")
    login_resp = client.post("/v1/auth/login", data={"username": "judy_refresh", "password": "testpass"})
    old_refresh = login_resp.json()["refresh_token"]

    mock_redis, revoked = _make_mock_redis()
    with patch("app.pubsub._redis", mock_redis):
        rotate_resp = client.post("/v1/auth/refresh", json={"refresh_token": old_refresh})
        new_refresh = rotate_resp.json()["refresh_token"]

        # New refresh token must still work
        resp = client.post("/v1/auth/refresh", json={"refresh_token": new_refresh})
        assert resp.status_code == 200
