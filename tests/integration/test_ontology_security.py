from tests.conftest import seed_user, get_token


def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── auth required ──────────────────────────────────────────────────────────────

def test_ontology_types_requires_auth(client):
    resp = client.get("/v1/ontology/types")
    assert resp.status_code == 401


def test_ontology_list_objects_requires_auth(client):
    resp = client.get("/v1/ontology/objects/Customer")
    assert resp.status_code == 401


def test_ontology_get_object_requires_auth(client):
    resp = client.get("/v1/ontology/objects/Customer/1")
    assert resp.status_code == 401


def test_ontology_create_requires_auth(client):
    resp = client.post("/v1/ontology/objects/Customer", json={"name": "x", "email": "x@y.com"})
    assert resp.status_code == 401


def test_ontology_update_requires_auth(client):
    resp = client.put("/v1/ontology/objects/Customer/1", json={"name": "x"})
    assert resp.status_code == 401


def test_ontology_delete_requires_auth(client):
    resp = client.delete("/v1/ontology/objects/Customer/1")
    assert resp.status_code == 401


# ── role checks on writes ────────────────────────────────────────────────────

def test_viewer_cannot_create_ontology_object(client, engine):
    seed_user(engine, "viewer_onto", "viewer")
    token = get_token(client, "viewer_onto")
    resp = client.post(
        "/v1/ontology/objects/Customer",
        json={"name": "Acme", "email": "acme@example.com"},
        headers=auth_header(token),
    )
    assert resp.status_code == 403


def test_analyst_can_create_and_update_ontology_object(client, engine):
    seed_user(engine, "analyst_onto", "analyst")
    token = get_token(client, "analyst_onto")

    create_resp = client.post(
        "/v1/ontology/objects/Customer",
        json={"name": "Acme", "email": "acme@example.com"},
        headers=auth_header(token),
    )
    assert create_resp.status_code == 201
    obj_id = create_resp.json()["id"]

    update_resp = client.put(
        f"/v1/ontology/objects/Customer/{obj_id}",
        json={"segment": "enterprise"},
        headers=auth_header(token),
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["segment"] == "enterprise"


# ── mass assignment protection ───────────────────────────────────────────────

def test_update_object_cannot_set_protected_fields(client, engine):
    seed_user(engine, "analyst_onto2", "analyst")
    token = get_token(client, "analyst_onto2")

    create_resp = client.post(
        "/v1/ontology/objects/Customer",
        json={"name": "Beta", "email": "beta@example.com"},
        headers=auth_header(token),
    )
    obj_id = create_resp.json()["id"]

    resp = client.put(
        f"/v1/ontology/objects/Customer/{obj_id}",
        json={"is_deleted": False, "created_by": 999},
        headers=auth_header(token),
    )
    assert resp.status_code == 422


# ── team scoping ──────────────────────────────────────────────────────────────

def test_create_object_stamps_team_id(client, engine):
    seed_user(engine, "team_a_creator", "analyst", team_id=1)
    token = get_token(client, "team_a_creator")

    resp = client.post(
        "/v1/ontology/objects/Customer",
        json={"name": "TeamCo", "email": "teamco@example.com"},
        headers=auth_header(token),
    )
    assert resp.status_code == 201
    assert resp.json()["team_id"] == 1


def test_team_user_cannot_see_other_teams_object(client, engine):
    seed_user(engine, "team_a_user", "analyst", team_id=1)
    seed_user(engine, "team_b_user", "analyst", team_id=2)
    token_a = get_token(client, "team_a_user")
    token_b = get_token(client, "team_b_user")

    create_resp = client.post(
        "/v1/ontology/objects/Customer",
        json={"name": "TeamA Co", "email": "teama@example.com"},
        headers=auth_header(token_a),
    )
    obj_id = create_resp.json()["id"]

    # Team B cannot read, update, or delete team A's object
    get_resp = client.get(f"/v1/ontology/objects/Customer/{obj_id}", headers=auth_header(token_b))
    assert get_resp.status_code == 404

    put_resp = client.put(
        f"/v1/ontology/objects/Customer/{obj_id}",
        json={"segment": "enterprise"},
        headers=auth_header(token_b),
    )
    assert put_resp.status_code == 404

    delete_resp = client.delete(f"/v1/ontology/objects/Customer/{obj_id}", headers=auth_header(token_b))
    assert delete_resp.status_code == 404

    # Team A can read its own object
    own_resp = client.get(f"/v1/ontology/objects/Customer/{obj_id}", headers=auth_header(token_a))
    assert own_resp.status_code == 200


def test_team_user_list_excludes_other_teams_objects(client, engine):
    seed_user(engine, "team_c_user", "analyst", team_id=10)
    seed_user(engine, "team_d_user", "analyst", team_id=20)
    token_c = get_token(client, "team_c_user")
    token_d = get_token(client, "team_d_user")

    client.post(
        "/v1/ontology/objects/Customer",
        json={"name": "TeamC Co", "email": "teamc@example.com"},
        headers=auth_header(token_c),
    )
    client.post(
        "/v1/ontology/objects/Customer",
        json={"name": "TeamD Co", "email": "teamd@example.com"},
        headers=auth_header(token_d),
    )

    resp = client.get("/v1/ontology/objects/Customer", headers=auth_header(token_c))
    assert resp.status_code == 200
    names = [item["name"] for item in resp.json()["items"]]
    assert "TeamC Co" in names
    assert "TeamD Co" not in names


def test_global_admin_sees_all_team_objects(client, engine):
    seed_user(engine, "team_e_user", "analyst", team_id=30)
    seed_user(engine, "global_admin", "admin")
    token_e = get_token(client, "team_e_user")
    admin_token = get_token(client, "global_admin")

    create_resp = client.post(
        "/v1/ontology/objects/Customer",
        json={"name": "TeamE Co", "email": "teame@example.com"},
        headers=auth_header(token_e),
    )
    obj_id = create_resp.json()["id"]

    resp = client.get(f"/v1/ontology/objects/Customer/{obj_id}", headers=auth_header(admin_token))
    assert resp.status_code == 200
    assert resp.json()["name"] == "TeamE Co"
