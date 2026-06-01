from datetime import datetime, timezone
from typing import Any
from enum import Enum
from pydantic import BaseModel
from sqlmodel import SQLModel, Field
import uuid


class ContextualData(BaseModel):
    type: str
    object_ids: list[int]
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: list[dict] = []


class ContextualLogic(BaseModel):
    name: str
    description: str
    priority: int = 0


class ContextualAction(BaseModel):
    tool_name: str
    description: str
    parameters: dict[str, Any] = {}


class AgentContext(BaseModel):
    agent_id: str
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    data: list[ContextualData] = []
    logic: list[ContextualLogic] = []
    actions: list[ContextualAction] = []
    metadata: dict[str, Any] = {}
    assembled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: int = 3600


class WorkflowStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    timeout = "timeout"


class Workflow(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    agent_id: str = Field(index=True)
    status: WorkflowStatus = WorkflowStatus.pending
    run_id: str | None = None
    result_json: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_by: int | None = None
