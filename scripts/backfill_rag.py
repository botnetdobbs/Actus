"""
Backfill RAG index for all existing ontology objects.

Run once after enabling RAG on a database that already has ontology objects.
Safe to run multiple times — pgvector upserts (ON CONFLICT DO UPDATE) are idempotent.

Usage:
    uv run python scripts/backfill_rag.py
"""
import sys
from pathlib import Path

# Add project root to path so app imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlmodel import Session, select
from app.database import get_engine
from app.ontology.registry import list_types, get_type
from app.rag.embedder import warmup
from app.rag.indexer import index_object

# Side-effect imports to register all ontology models
import app.ontology.models  # noqa: F401


def main() -> None:
    print("Loading embedding model...")
    warmup()

    total = 0
    errors = 0

    with Session(get_engine()) as session:
        for type_name in list_types():
            cls = get_type(type_name)
            objects = session.exec(
                select(cls).where(cls.is_deleted == False)
            ).all()
            print(f"{type_name}: {len(objects)} objects")
            for obj in objects:
                try:
                    index_object(type_name, obj.id, obj)
                    total += 1
                    print(f"  ✓ {type_name}:{obj.id}")
                except Exception as e:
                    errors += 1
                    print(f"  ✗ {type_name}:{obj.id} — {e}", file=sys.stderr)

    print(f"\nDone. Indexed: {total}, errors: {errors}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
