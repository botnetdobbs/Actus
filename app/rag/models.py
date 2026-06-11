from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, ClassVar

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, UniqueConstraint
from sqlmodel import Field, SQLModel

VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimensions


class VectorIndex(SQLModel, table=True):
    __tablename__: ClassVar[str] = "vector_indexes"  # pyright: ignore[reportIncompatibleVariableOverride]
    __table_args__ = (
        UniqueConstraint("object_type", "object_id", name="uq_vector_indexes_type_id"),
    )

    id: int | None = Field(default=None, primary_key=True)
    object_type: str = Field(index=True)
    object_id: int = Field(index=True)
    team_id: int | None = Field(default=None, index=True)
    document: str
    embedding: Any = Field(default=None, sa_column=Column(Vector(VECTOR_DIM)))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
