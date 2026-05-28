from datetime import datetime, timezone
from sqlmodel import SQLModel, Field
from app.ontology.registry import register


class OntologyObjectBase(SQLModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: int | None = None
    is_deleted: bool = Field(default=False, index=True)
    deleted_at: datetime | None = None
    deleted_by: int | None = None

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def soft_delete(self, deleted_by: int | None = None) -> None:
        self.is_deleted = True
        self.deleted_at = datetime.now(timezone.utc)
        self.deleted_by = deleted_by


@register("Customer")
class Customer(OntologyObjectBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    email: str = Field(unique=True)
    segment: str | None = None
    is_active: bool = True


@register("Machine")
class Machine(OntologyObjectBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    serial_number: str = Field(unique=True)
    location: str
    model_name: str


@register("MachineReading")
class MachineReading(OntologyObjectBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    machine_id: int = Field(foreign_key="machine.id", index=True)
    temperature: float
    pressure: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@register("AnalysisReport")
class AnalysisReport(OntologyObjectBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    title: str
    content: str
    source_type: str
    source_ids: str
    status: str = "draft"
