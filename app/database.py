import threading
from typing import Any
from sqlmodel import SQLModel, Session, create_engine

from app.config import get_settings

_engine = None
_engine_lock = threading.Lock()


def get_engine():
    global _engine

    if _engine is None:
        with _engine_lock:
            if _engine is None:
                settings = get_settings()

                kwargs: dict[str, Any] = {"echo": settings.debug}

                if "sqlite" in settings.database_url:
                    kwargs["connect_args"] = {"check_same_thread": False}

                if "postgresql" in settings.database_url:
                    kwargs.update({
                        "pool_size": 10,
                        "max_overflow": 20,
                        "pool_pre_ping": True,
                        "pool_recycle": 3600,
                    })

                _engine = create_engine(settings.database_url, **kwargs)

    return _engine


def create_db_and_tables():
    SQLModel.metadata.create_all(get_engine())


def get_session():
    with Session(get_engine()) as session:
        yield session