"""
Role hierarchy tests.

roles_hierarchy = {"viewer": 0, "analyst": 1, "admin": 2}

require_role("admin")  → only admin passes
require_role("analyst") → analyst and admin pass, viewer blocked
require_role("viewer")  → all roles pass

The assign_role endpoint (PATCH /auth/users/{id}/role) is the only admin-gated
endpoint currently in the codebase, so it is used as the fixture for hierarchy tests.
An analyst-gated endpoint will be added with the automation module.
"""
from unittest.mock import patch, MagicMock
from sqlmodel import Session
from tests.conftest import seed_user, get_token


# ── helpers ──────────────────────────────────────────────────────────────────

def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── registration ──────────────────────────────────────────────────────────────

def test_register_creates_viewer(client):
    resp = client.post("/v1/auth/register", json={"username": "newuser", "password": "pass1234"})
    assert resp.status_code == 201
    assert resp.json()["role"] == "viewer"


def test_register_rejects_role_field(client):
    resp = client.post(
        "/v1/auth/register",
        json={"username": "hacker", "password": "pass1234", "role": "admin"},
    )
    assert resp.status_code == 422


def test_register_duplicate_username(client, engine):
    seed_user(engine, "alice", "viewer")
    resp = client.post("/v1/auth/register", json={"username": "alice", "password": "pass1234"})
    assert resp.status_code == 409


# ── login ─────────────────────────────────────────────────────────────────────

def test_login_returns_token(client, engine):
    seed_user(engine, "bob", "viewer")
    resp = client.post("/v1/auth/login", data={"username": "bob", "password": "testpass"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_login_wrong_password(client, engine):
    seed_user(engine, "carol", "viewer")
    resp = client.post("/v1/auth/login", data={"username": "carol", "password": "wrong"})
    assert resp.status_code == 401


def test_login_unknown_user(client):
    resp = client.post("/v1/auth/login", data={"username": "ghost", "password": "pass"})
    assert resp.status_code == 401


def test_account_lockout_after_max_failures(client, engine):
    seed_user(engine, "dave", "viewer")
    for _ in range(5):
        client.post("/v1/auth/login", data={"username": "dave", "password": "wrong"})
    resp = client.post("/v1/auth/login", data={"username": "dave", "password": "testpass"})
    assert resp.status_code == 403


# ── /me ───────────────────────────────────────────────────────────────────────

def test_me_requires_auth(client):
    resp = client.get("/v1/auth/me")
    assert resp.status_code == 401


def test_me_returns_current_user(client, engine):
    seed_user(engine, "eve", "analyst")
    token = get_token(client, "eve")
    resp = client.get("/v1/auth/me", headers=auth_header(token))
    assert resp.status_code == 200
    assert resp.json()["username"] == "eve"
    assert resp.json()["role"] == "analyst"


# ── role hierarchy ────────────────────────────────────────────────────────────

def test_unauthenticated_blocked_from_admin_endpoint(client, engine):
    target = seed_user(engine, "target1", "viewer")
    resp = client.patch(f"/v1/auth/users/{target.id}/role", json={"role": "analyst"})
    assert resp.status_code == 401


def test_viewer_blocked_from_admin_endpoint(client, engine):
    seed_user(engine, "viewer1", "viewer")
    target = seed_user(engine, "target2", "viewer")
    token = get_token(client, "viewer1")
    resp = client.patch(
        f"/v1/auth/users/{target.id}/role",
        json={"role": "analyst"},
        headers=auth_header(token),
    )
    assert resp.status_code == 403


def test_analyst_blocked_from_admin_endpoint(client, engine):
    seed_user(engine, "analyst1", "analyst")
    target = seed_user(engine, "target3", "viewer")
    token = get_token(client, "analyst1")
    resp = client.patch(
        f"/v1/auth/users/{target.id}/role",
        json={"role": "analyst"},
        headers=auth_header(token),
    )
    assert resp.status_code == 403


def test_admin_can_access_admin_endpoint(client, engine):
    seed_user(engine, "admin1", "admin")
    target = seed_user(engine, "target4", "viewer")
    token = get_token(client, "admin1")
    resp = client.patch(
        f"/v1/auth/users/{target.id}/role",
        json={"role": "analyst"},
        headers=auth_header(token),
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "analyst"


def test_invalid_role_rejected(client, engine):
    seed_user(engine, "admin2", "admin")
    target = seed_user(engine, "target5", "viewer")
    token = get_token(client, "admin2")
    resp = client.patch(
        f"/v1/auth/users/{target.id}/role",
        json={"role": "superuser"},
        headers=auth_header(token),
    )
    assert resp.status_code == 422


# ── refresh tokens ────────────────────────────────────────────────────────────

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


# ── logout / token revocation ─────────────────────────────────────────────────

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
