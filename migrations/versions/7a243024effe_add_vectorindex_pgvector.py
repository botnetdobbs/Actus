"""add_vectorindex_pgvector

Revision ID: 7a243024effe
Revises: c2db21e1f86c
Create Date: 2026-06-05

"""
from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

revision: str = '7a243024effe'
down_revision: Union[str, Sequence[str], None] = 'c2db21e1f86c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    is_pg = op.get_context().connection.dialect.name == "postgresql"

    if is_pg:
        # pgvector extension — required before the vector column can be created.
        # Requires pgvector/pgvector:pg16 Docker image (postgres:16-alpine lacks it).
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "vectorindex",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("object_type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("object_id", sa.Integer(), nullable=False),
        sa.Column("document", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("object_type", "object_id", name="uq_vectorindex_type_id"),
    )
    op.create_index("ix_vectorindex_object_type", "vectorindex", ["object_type"])
    op.create_index("ix_vectorindex_object_id", "vectorindex", ["object_id"])

    if is_pg:
        # Add vector column separately — requires the extension to already exist.
        # 384 dimensions = all-MiniLM-L6-v2 output size. Changing EMBEDDING_MODEL
        # to a different dimension requires a new migration + full re-index.
        op.execute("ALTER TABLE vectorindex ADD COLUMN embedding vector(384)")

        # HNSW index for fast approximate nearest-neighbour cosine search.
        # m=16 and ef_construction=64 are the recommended defaults from pgvector docs.
        op.execute("""
            CREATE INDEX ix_vectorindex_embedding
            ON vectorindex USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)

        # GIN index for fast PostgreSQL full-text search on the document field.
        op.execute("""
            CREATE INDEX ix_vectorindex_fts
            ON vectorindex USING gin (to_tsvector('english', document))
        """)


def downgrade() -> None:
    is_pg = op.get_context().connection.dialect.name == "postgresql"
    op.drop_table("vectorindex")
    if is_pg:
        op.execute("DROP EXTENSION IF EXISTS vector")
