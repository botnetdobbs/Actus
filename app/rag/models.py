from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, UniqueConstraint
from sqlmodel import Field, SQLModel

VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimensions


class VectorIndex(SQLModel, table=True):
    __tablename__ = "vectorindex"
    __table_args__ = (
        UniqueConstraint("object_type", "object_id", name="uq_vectorindex_type_id"),
    )

    id: int | None = Field(default=None, primary_key=True)
    object_type: str = Field(index=True)
    object_id: int = Field(index=True)
    document: str
    embedding: Any = Field(default=None, sa_column=Column(Vector(VECTOR_DIM)))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
