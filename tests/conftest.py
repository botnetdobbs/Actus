import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool
from unittest.mock import patch
import app.database as db_module
from app.main import create_app
from app.database import get_session
from app.auth.models import User, hash_password


@pytest.fixture()
def engine():
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # Import all table models before create_all so metadata is populated
    import app.auth.models
    import app.ontology.models
    import app.context.store
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

    # instrument_app registers Prometheus metrics into the global registry.
    # Re-registering on each test raises ValueError, so we suppress it in tests.
    with patch("app.main.instrument_app"):
        application = create_app()
    application.dependency_overrides[get_session] = override_session

    with TestClient(application, raise_server_exceptions=False) as c:
        yield c

    application.dependency_overrides.clear()


def seed_user(engine, username: str, role: str, password: str = "testpass") -> User:
    with Session(engine) as session:
        user = User(username=username, hashed_password=hash_password(password), role=role)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


def get_token(client: TestClient, username: str, password: str = "testpass") -> str:
    resp = client.post("/auth/login", data={"username": username, "password": password})
    assert resp.status_code == 200, f"Login failed for {username!r}: {resp.json()}"
    return resp.json()["access_token"]
