import json
from datetime import datetime, timezone
from typing import ClassVar
from pydantic import BaseModel, ConfigDict, computed_field, field_validator
from sqlmodel import SQLModel, Field, Session
from app.database import get_engine
import structlog

log = structlog.get_logger()

_TRACE_SIZE_LIMIT = 64 * 1024  # 64 KB


class AgentRunLog(SQLModel, table=True):
    __tablename__: ClassVar[str] = "agent_run_logs"  # pyright: ignore[reportIncompatibleVariableOverride]

    id: int | None = Field(default=None, primary_key=True)
    agent_id: str | None = Field(default=None, index=True)
    run_id: str = Field(index=True)
    triggered_by: int | None = Field(default=None, index=True)
    team_id: int | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    model: str | None = None
    pii_detected: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tool_calls: str = Field(default="[]")  # JSON: [{tool, success, detail}]
    outcome: str = "success"              # "success" | "incomplete" | "error" | "timeout"
    result_summary: str | None = None     # first 500 chars only, never raw output
    ip_address: str | None = None
    trace_json: str | None = None         # iteration-level trace; capped at 64 KB


class AgentRunLogResponse(BaseModel):
    """API representation of an AgentRunLog row.

    tool_calls is stored as a JSON string in the DB; this model deserialises it
    automatically. duration_seconds is computed from started_at / completed_at.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: str
    agent_id: str | None
    triggered_by: int | None
    team_id: int | None
    started_at: datetime
    completed_at: datetime | None
    model: str | None
    pii_detected: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tool_calls: list[dict]
    outcome: str
    result_summary: str | None

    @field_validator("tool_calls", mode="before")
    @classmethod
    def _parse_tool_calls(cls, v) -> list[dict]:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, ValueError):
                return []
        return v or []

    @computed_field
    @property
    def duration_seconds(self) -> float | None:
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


def _cap_trace(trace: list[dict]) -> str:
    """Serialise trace to a JSON object with embedded truncated flag.

    If the serialised size exceeds 64 KB, tool call and result data is stripped
    from all entries and truncated is set to True in the stored object.
    """
    body = json.dumps({"truncated": False, "iterations": trace})
    if len(body.encode()) <= _TRACE_SIZE_LIMIT:
        return body

    trimmed = [dict(entry) for entry in trace]
    for entry in trimmed:
        entry.pop("tool_result", None)
        entry.pop("tool_calls", None)
    return json.dumps({"truncated": True, "iterations": trimmed})


def log_agent_run(
    run_id: str,
    started_at: datetime,
    model: str | None,
    pii_detected: bool,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    outcome: str,
    tool_calls: list[dict] | None = None,
    agent_id: str | None = None,
    triggered_by: int | None = None,
    team_id: int | None = None,
    result_summary: str | None = None,
    ip_address: str | None = None,
    trace: list[dict] | None = None,
) -> None:
    trace_json: str | None = None
    if trace:
        trace_json = _cap_trace(trace)
        try:
            stored = json.loads(trace_json)
            if isinstance(stored, dict) and stored.get("truncated"):
                log.warning("agent_run_trace_truncated", run_id=run_id)
        except Exception:
            pass

    entry = AgentRunLog(
        run_id=run_id,
        agent_id=agent_id,
        triggered_by=triggered_by,
        team_id=team_id,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
        model=model,
        pii_detected=pii_detected,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        tool_calls=json.dumps(tool_calls or []),
        outcome=outcome,
        result_summary=result_summary,
        ip_address=ip_address,
        trace_json=trace_json,
    )
    with Session(get_engine()) as session:
        try:
            session.add(entry)
            session.commit()
        except Exception as e:
            session.rollback()
            log.error("agent_run_log_write_failed", run_id=run_id, error=str(e))
