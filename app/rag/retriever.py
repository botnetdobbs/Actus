import structlog
from sqlalchemy import func
from sqlmodel import Session, select

from app.database import get_engine
from app.rag.embedder import embed
from app.rag.models import VectorIndex

log = structlog.get_logger()

_RRF_K = 60  # standard RRF parameter — higher reduces influence of top ranks


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank + 1)


def retrieve(query: str, type_name: str | None = None, top_k: int = 5) -> list[dict]:
    if not query.strip():
        return []

    embedding = embed(query)

    with Session(get_engine()) as session:
        # Semantic search — cosine distance via pgvector
        sem_q = (
            select(VectorIndex)
            .order_by(VectorIndex.embedding.cosine_distance(embedding))
            .limit(top_k * 3)
        )
        if type_name:
            sem_q = sem_q.where(VectorIndex.object_type == type_name)
        semantic_rows = session.exec(sem_q).all()

        # Full-text search — PostgreSQL tsvector
        tsquery = func.plainto_tsquery("english", query)
        fts_filter = func.to_tsvector("english", VectorIndex.document).op("@@")(tsquery)
        fts_q = select(VectorIndex).where(fts_filter).limit(top_k)
        if type_name:
            fts_q = fts_q.where(VectorIndex.object_type == type_name)
        try:
            fts_rows = session.exec(fts_q).all()
        except Exception as e:
            log.warning("rag_fts_failed", error=str(e))
            fts_rows = []

    # Reciprocal Rank Fusion merge
    scores: dict[tuple, float] = {}
    row_map: dict[tuple, VectorIndex] = {}

    for rank, row in enumerate(semantic_rows):
        key = (row.object_type, row.object_id)
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
        row_map[key] = row

    for rank, row in enumerate(fts_rows):
        key = (row.object_type, row.object_id)
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
        row_map.setdefault(key, row)

    # Session is closed here; row attributes are safe to access because
    # .all() eagerly loads all column values into detached Python objects.
    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [
        {
            "document": row_map[k].document,
            "metadata": {"type": k[0], "object_id": k[1]},
            "score": round(scores[k], 4),
        }
        for k in sorted_keys[:top_k]
    ]
