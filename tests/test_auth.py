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
import pytest
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
    viewer = seed_user(engine, "viewer1", "viewer")
    target = seed_user(engine, "target2", "viewer")
    token = get_token(client, "viewer1")
    resp = client.patch(
        f"/v1/auth/users/{target.id}/role",
        json={"role": "analyst"},
        headers=auth_header(token),
    )
    assert resp.status_code == 403


def test_analyst_blocked_from_admin_endpoint(client, engine):
    analyst = seed_user(engine, "analyst1", "analyst")
    target = seed_user(engine, "target3", "viewer")
    token = get_token(client, "analyst1")
    resp = client.patch(
        f"/v1/auth/users/{target.id}/role",
        json={"role": "analyst"},
        headers=auth_header(token),
    )
    assert resp.status_code == 403


def test_admin_can_access_admin_endpoint(client, engine):
    admin = seed_user(engine, "admin1", "admin")
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
    admin = seed_user(engine, "admin2", "admin")
    target = seed_user(engine, "target5", "viewer")
    token = get_token(client, "admin2")
    resp = client.patch(
        f"/v1/auth/users/{target.id}/role",
        json={"role": "superuser"},
        headers=auth_header(token),
    )
    assert resp.status_code == 422
