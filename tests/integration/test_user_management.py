from tests.conftest import seed_user, get_token


def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── GET /auth/users ───────────────────────────────────────────────────────────

def test_list_users_requires_admin(client, engine):
    seed_user(engine, "analyst1", "analyst")
    token = get_token(client, "analyst1")
    resp = client.get("/v1/auth/users", headers=auth_header(token))
    assert resp.status_code == 403


def test_list_users_returns_all_active(client, engine):
    seed_user(engine, "admin1", "admin")
    seed_user(engine, "user1", "viewer")
    seed_user(engine, "user2", "analyst")
    token = get_token(client, "admin1")
    resp = client.get("/v1/auth/users", headers=auth_header(token))
    assert resp.status_code == 200
    usernames = [u["username"] for u in resp.json()]
    assert "user1" in usernames
    assert "user2" in usernames


def test_list_users_pagination(client, engine):
    seed_user(engine, "admin2", "admin")
    for i in range(5):
        seed_user(engine, f"bulk{i}", "viewer")
    token = get_token(client, "admin2")
    resp = client.get("/v1/auth/users?limit=2&offset=0", headers=auth_header(token))
    assert resp.status_code == 200
    assert len(resp.json()) == 2


# ── DELETE /auth/users/{id} ───────────────────────────────────────────────────

def test_delete_user_requires_admin(client, engine):
    seed_user(engine, "analyst2", "analyst")
    target = seed_user(engine, "target1", "viewer")
    token = get_token(client, "analyst2")
    resp = client.delete(f"/v1/auth/users/{target.id}", headers=auth_header(token))
    assert resp.status_code == 403


def test_delete_user_soft_deletes(client, engine):
    seed_user(engine, "admin3", "admin")
    target = seed_user(engine, "target2", "viewer")
    token = get_token(client, "admin3")
    resp = client.delete(f"/v1/auth/users/{target.id}", headers=auth_header(token))
    assert resp.status_code == 204
    # Deleted user cannot log in
    resp = client.post("/v1/auth/login", data={"username": "target2", "password": "testpass"})
    assert resp.status_code == 401


def test_delete_user_not_found(client, engine):
    seed_user(engine, "admin4", "admin")
    token = get_token(client, "admin4")
    resp = client.delete("/v1/auth/users/99999", headers=auth_header(token))
    assert resp.status_code == 404


def test_admin_cannot_delete_self(client, engine):
    admin = seed_user(engine, "admin5", "admin")
    token = get_token(client, "admin5")
    resp = client.delete(f"/v1/auth/users/{admin.id}", headers=auth_header(token))
    assert resp.status_code == 400


# ── PATCH /auth/users/{id}/password ──────────────────────────────────────────

def test_password_reset_requires_admin(client, engine):
    seed_user(engine, "analyst3", "analyst")
    target = seed_user(engine, "target3", "viewer")
    token = get_token(client, "analyst3")
    resp = client.patch(
        f"/v1/auth/users/{target.id}/password",
        json={"new_password": "newpass"},
        headers=auth_header(token),
    )
    assert resp.status_code == 403


def test_password_reset_allows_new_login(client, engine):
    seed_user(engine, "admin6", "admin")
    target = seed_user(engine, "target4", "viewer")
    token = get_token(client, "admin6")
    resp = client.patch(
        f"/v1/auth/users/{target.id}/password",
        json={"new_password": "newpassword123"},
        headers=auth_header(token),
    )
    assert resp.status_code == 200
    # Old password no longer works
    resp = client.post("/v1/auth/login", data={"username": "target4", "password": "testpass"})
    assert resp.status_code == 401
    # New password works
    resp = client.post("/v1/auth/login", data={"username": "target4", "password": "newpassword123"})
    assert resp.status_code == 200


def test_password_reset_not_found(client, engine):
    seed_user(engine, "admin7", "admin")
    token = get_token(client, "admin7")
    resp = client.patch(
        "/v1/auth/users/99999/password",
        json={"new_password": "newpassword"},
        headers=auth_header(token),
    )
    assert resp.status_code == 404


# ── PATCH /auth/users/{id}/role — last-admin protection ──────────────────────

def test_cannot_demote_last_admin(client, engine):
    admin = seed_user(engine, "lastadmin", "admin")
    token = get_token(client, "lastadmin")
    resp = client.patch(
        f"/v1/auth/users/{admin.id}/role",
        json={"role": "viewer"},
        headers=auth_header(token),
    )
    assert resp.status_code == 400


def test_can_demote_admin_when_another_admin_remains(client, engine):
    seed_user(engine, "admin8", "admin")
    target = seed_user(engine, "admin9", "admin")
    token = get_token(client, "admin8")
    resp = client.patch(
        f"/v1/auth/users/{target.id}/role",
        json={"role": "viewer"},
        headers=auth_header(token),
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "viewer"
