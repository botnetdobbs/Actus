from datetime import datetime, timezone
from typing import ClassVar
from sqlmodel import SQLModel, Field
from app.ontology.registry import register


class OntologyObjectBase(SQLModel):
    """
    Base class for all ontology types. Provides audit fields, soft delete,
    and RAG indexing (automatic on create/update via the ontology router).

    NOTE: `team_id` (nullable) scopes records to a team, following the same
    convention as `User.team_id`/`Workflow.team_id`: NULL means global/unscoped
    (visible to everyone), non-null restricts visibility.

    To define a new type:

        @register("Invoice")
        class Invoice(OntologyObjectBase, table=True):
            id: int | None = Field(default=None, primary_key=True)
            number: str = Field(unique=True, index=True)
            client: str
            amount: float
            status: str = "unpaid"

    Then run:
        make migrations msg="add_invoice"
        make migrate
    """
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: int | None = None
    team_id: int | None = Field(default=None, index=True)
    is_deleted: bool = Field(default=False, index=True)
    deleted_at: datetime | None = None
    deleted_by: int | None = None

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def soft_delete(self, deleted_by: int | None = None) -> None:
        self.is_deleted = True
        self.deleted_at = datetime.now(timezone.utc)
        self.deleted_by = deleted_by


# ── Example types — replace with your own domain models ──────────────────────
#
# These are illustrative. Delete them and define types that match your domain.
# Each type gets full CRUD endpoints at /ontology/objects/{type_name} and is
# automatically indexed into RAG on create/update.

@register("Customer")
class Customer(OntologyObjectBase, table=True):
    __tablename__: ClassVar[str] = "customers"  # pyright: ignore[reportIncompatibleVariableOverride]

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    email: str = Field(unique=True)
    segment: str | None = None
    is_active: bool = True
