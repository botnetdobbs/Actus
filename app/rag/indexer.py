from datetime import datetime, timezone

import structlog
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlmodel import Session, select

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


def _object_to_text(type_name: str, obj) -> str:
    if hasattr(obj, "rag_document") and callable(obj.rag_document):
        return obj.rag_document()

    parts = [type_name]
    for field, value in obj.model_dump().items():
        if field in _METADATA_FIELDS:
            continue
        if value is None:
            continue
        if isinstance(value, bool):
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
        stmt = pg_insert(VectorIndex).values(
            object_type=type_name,
            object_id=object_id,
            document=text,
            embedding=embedding,
            created_at=datetime.now(timezone.utc),
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_vectorindex_type_id",
            set_={
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
