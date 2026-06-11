from datetime import datetime, timezone
from typing import Any, cast

import structlog
from sqlalchemy import CursorResult, delete
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlmodel import Session, col, select

from app.database import get_engine
from app.rag.embedder import embed
from app.rag.models import VectorIndex

_PG_DRIVER = "postgresql"


def _is_postgres() -> bool:
    return _PG_DRIVER in get_engine().url.drivername

log = structlog.get_logger()

_METADATA_FIELDS = {
    "id", "created_at", "updated_at", "created_by",
    "is_deleted", "deleted_at", "deleted_by",
}


def _object_to_text(type_name: str, obj: Any) -> str:
    if hasattr(obj, "rag_document") and callable(obj.rag_document):
        return str(obj.rag_document())

    parts = [type_name]
    for field, value in obj.model_dump().items():
        if field in _METADATA_FIELDS:
            continue
        if value is None:
            continue
        if field.endswith("_id") and isinstance(value, int):  # skip foreign keys
            continue
        parts.append(f"{field}: {value}")
    return "; ".join(parts)


def index_object(type_name: str, object_id: int, obj) -> None:
    if not _is_postgres():
        return  # pgvector requires PostgreSQL; no-op in dev with SQLite
    try:
        text = _object_to_text(type_name, obj)
        embedding = embed(text)
        team_id = getattr(obj, "team_id", None)
        stmt = pg_insert(VectorIndex).values(
            object_type=type_name,
            object_id=object_id,
            team_id=team_id,
            document=text,
            embedding=embedding,
            created_at=datetime.now(timezone.utc),
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_vector_indexes_type_id",
            set_={
                "team_id": stmt.excluded.team_id,
                "document": stmt.excluded.document,
                "embedding": stmt.excluded.embedding,
                "created_at": stmt.excluded.created_at,
            },
        )
        with Session(get_engine()) as session:
            session.execute(stmt)
            session.commit()
        log.info("rag_indexed", type=type_name, id=object_id)
    except Exception as e:
        log.error("rag_index_failed", type=type_name, id=object_id, error=str(e))


def index_text(type_name: str, object_id: int, text: str, team_id: int | None = None) -> None:
    """Index a raw text string into VectorIndex. No-op on SQLite. Re-raises on failure."""
    if not _is_postgres():
        return
    embedding = embed(text)
    stmt = pg_insert(VectorIndex).values(
        object_type=type_name,
        object_id=object_id,
        team_id=team_id,
        document=text,
        embedding=embedding,
        created_at=datetime.now(timezone.utc),
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_vector_indexes_type_id",
        set_={
            "team_id": stmt.excluded.team_id,
            "document": stmt.excluded.document,
            "embedding": stmt.excluded.embedding,
            "created_at": stmt.excluded.created_at,
        },
    )
    with Session(get_engine()) as session:
        session.execute(stmt)
        session.commit()
    log.info("rag_text_indexed", type=type_name, id=object_id)


def delete_by_type(type_name: str) -> int:
    """Delete all VectorIndex rows for a type_name. Returns count deleted. No-op on SQLite."""
    if not _is_postgres():
        return 0
    with Session(get_engine()) as session:
        result = cast(CursorResult, session.execute(delete(VectorIndex).where(col(VectorIndex.object_type) == type_name)))
        session.commit()
    log.info("rag_type_deleted", type=type_name, count=result.rowcount)
    return result.rowcount


def delete_from_index(type_name: str, object_id: int) -> None:
    if not _is_postgres():
        return  # pgvector requires PostgreSQL; no-op in dev with SQLite
    try:
        with Session(get_engine()) as session:
            row = session.exec(
                select(VectorIndex)
                .where(VectorIndex.object_type == type_name)
                .where(VectorIndex.object_id == object_id)
            ).first()
            if row:
                session.delete(row)
                session.commit()
        log.info("rag_deleted", type=type_name, id=object_id)
    except Exception as e:
        log.error("rag_delete_failed", type=type_name, id=object_id, error=str(e))
