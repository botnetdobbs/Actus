import asyncio
import hashlib
import hmac as _hmac
import json
from datetime import datetime, timezone
from typing import Annotated
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from sqlmodel import Session, select
from app.agents.audit import AgentRunLog, AgentRunLogResponse, log_agent_run
from app.agents.builder import get_agent, list_agents, reload_agents
from app.agents.orchestrator import run_agent_with_timeout
from app.auth.models import User
from app.auth.jwt import require_role, write_audit_log
from app.context.models import Workflow, WorkflowStatus
from app.database import get_engine, get_session
import structlog

log = structlog.get_logger()
router = APIRouter()

MAX_WEBHOOK_BODY_BYTES = 1 * 1024 * 1024
_SIG_HEADERS = ("x-actus-signature", "x-hub-signature-256")


def _verify_webhook_signature(body: bytes, secret: str, signature: str) -> bool:
    expected = "sha256=" + _hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return _hmac.compare_digest(expected, signature)


_OUTCOME_MAP = {
    "completed": "success",
    "incomplete": "incomplete",
    "error": "error",
    "timeout": "timeout",
}


class TriggerRequest(BaseModel):
    extra_context: dict | None = None


class WorkflowResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    agent_id: str
    status: str
    run_id: str | None
    result_json: str | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    created_by: int | None
    extra_context_json: str | None


# keyed by workflow_id; holds the live event queue for an in-progress run on this process
_run_queues: dict[int, asyncio.Queue] = {}


async def _run_workflow(workflow_id: int, triggered_by: int | None, ip_address: str | None) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    _run_queues[workflow_id] = queue

    started_at = datetime.now(timezone.utc)
    with Session(get_engine()) as session:
        wf = session.get(Workflow, workflow_id)
        wf.status = WorkflowStatus.running
        wf.started_at = started_at
        session.add(wf)
        session.commit()
        session.refresh(wf)
        agent_id = wf.agent_id
        extra_context_json = wf.extra_context_json

    status = WorkflowStatus.failed
    outcome = "error"
    result = None
    error = None
    config = None
    try:
        config = get_agent(agent_id)
        extra_context = json.loads(extra_context_json) if extra_context_json else None
        result = await run_agent_with_timeout(config, extra_context=extra_context, event_queue=queue)
        status = WorkflowStatus.completed
        outcome = _OUTCOME_MAP.get(result.get("status", ""), "error")
    except Exception as e:
        error = str(e)
        log.error("workflow_failed", workflow_id=workflow_id, agent_id=agent_id, error=error)
    finally:
        _run_queues.pop(workflow_id, None)
        with Session(get_engine()) as session:
            wf = session.get(Workflow, workflow_id)
            wf.status = status
            wf.completed_at = datetime.now(timezone.utc)
            wf.result_json = json.dumps(result) if result else None
            wf.error = error
            if result:
                wf.run_id = result.get("run_id")
            session.add(wf)
            session.commit()

        raw_result = result.get("result") if result else None
        summary = str(raw_result)[:500] if raw_result else None
        log_agent_run(
            run_id=result.get("run_id", "") if result else "",
            started_at=started_at,
            model=config.model if config else None,
            pii_detected=result.get("pii_detected", False) if result else False,
            prompt_tokens=result.get("prompt_tokens", 0) if result else 0,
            completion_tokens=result.get("completion_tokens", 0) if result else 0,
            total_tokens=result.get("total_tokens", 0) if result else 0,
            outcome=outcome,
            tool_calls=result.get("tool_calls") if result else None,
            agent_id=agent_id,
            triggered_by=triggered_by,
            result_summary=summary,
            ip_address=ip_address,
        )


# ── Trigger ───────────────────────────────────────────────────────────────────

@router.post("/trigger/{agent_id}", status_code=202)
async def trigger_agent(
    agent_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    body: TriggerRequest | None = None,
    user: User = Depends(require_role("analyst")),
):
    try:
        config = get_agent(agent_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Agent not found: '{agent_id}'")

    with Session(get_engine()) as session:
        wf = Workflow(
            name=config.name,
            agent_id=agent_id,
            created_by=user.id,
            extra_context_json=json.dumps(body.extra_context) if body and body.extra_context else None,
        )
        session.add(wf)
        session.commit()
        session.refresh(wf)
        workflow_id = wf.id

    ip = request.client.host if request.client else None
    background_tasks.add_task(_run_workflow, workflow_id, user.id, ip)

    write_audit_log(
        username=user.username,
        action="agent_trigger",
        resource=agent_id,
        ip=ip,
        detail=f"workflow_id={workflow_id}",
    )

    return {"status": "queued", "agent_id": agent_id, "workflow_id": workflow_id}


# ── Webhook trigger ───────────────────────────────────────────────────────────

@router.post("/webhooks/{agent_id}", status_code=202)
async def webhook_trigger(
    agent_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    try:
        config = get_agent(agent_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Agent not found: '{agent_id}'")

    if not config.webhook or not config.webhook.secret:
        raise HTTPException(status_code=403, detail="Webhook not enabled for this agent")

    body = await request.body()
    if len(body) > MAX_WEBHOOK_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Webhook payload exceeds 1 MB limit")

    signature = next(
        (request.headers.get(h) for h in _SIG_HEADERS if request.headers.get(h)),
        None,
    )
    if not signature:
        raise HTTPException(status_code=401, detail="Missing webhook signature")
    if not _verify_webhook_signature(body, config.webhook.secret, signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    extra_context: dict | None = None
    if body:
        try:
            parsed = json.loads(body)
            extra_context = parsed if isinstance(parsed, dict) else {"payload": parsed}
        except (json.JSONDecodeError, ValueError):
            extra_context = {"raw": body.decode("utf-8", errors="replace")}

    with Session(get_engine()) as session:
        wf = Workflow(
            name=config.name,
            agent_id=agent_id,
            created_by=None,
            extra_context_json=json.dumps(extra_context) if extra_context else None,
        )
        session.add(wf)
        session.commit()
        session.refresh(wf)
        workflow_id = wf.id

    ip = request.client.host if request.client else None
    background_tasks.add_task(_run_workflow, workflow_id, None, ip)

    write_audit_log(
        username="webhook",
        action="webhook_trigger",
        resource=agent_id,
        ip=ip,
        detail=f"workflow_id={workflow_id}",
    )

    return {"status": "queued", "agent_id": agent_id, "workflow_id": workflow_id}


# ── Workflow polling ───────────────────────────────────────────────────────────

@router.get("/workflows", response_model=list[WorkflowResponse])
async def list_workflows(
    agent_id: Annotated[str | None, Query()] = None,
    status: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    session: Session = Depends(get_session),
    _: User = Depends(require_role("analyst")),
):
    query = select(Workflow)
    if agent_id:
        query = query.where(Workflow.agent_id == agent_id)
    if status:
        query = query.where(Workflow.status == status)
    query = query.order_by(Workflow.created_at.desc()).offset(offset).limit(limit)
    return session.exec(query).all()


@router.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: int,
    session: Session = Depends(get_session),
    _: User = Depends(require_role("analyst")),
):
    wf = session.get(Workflow, workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    return wf


@router.get("/workflows/{workflow_id}/stream")
async def stream_workflow(
    workflow_id: int,
    request: Request,
    _: User = Depends(require_role("analyst")),
):
    _TERMINAL = {WorkflowStatus.completed, WorkflowStatus.failed, WorkflowStatus.timeout}

    async def sse_generator():
        # Emit current status immediately from DB — fast first event, captures values inside session
        with Session(get_engine()) as session:
            db_wf = session.get(Workflow, workflow_id)
            if db_wf is None:
                found = False
                wf_status = None
                terminal_payload = None
            else:
                found = True
                wf_status = db_wf.status
                terminal_payload = (WorkflowResponse.model_validate(db_wf).model_dump_json()
                                    if wf_status in _TERMINAL else None)

        if not found:
            yield 'data: {"type": "error", "error": "Not found"}\n\n'
            return

        yield f'data: {{"type": "status", "status": "{wf_status.value}", "workflow_id": {workflow_id}}}\n\n'

        if wf_status in _TERMINAL:
            yield f"data: {terminal_payload}\n\n"
            return

        # Try live queue (per-iteration events) — falls back to DB polling if absent
        queue = _run_queues.get(workflow_id)
        if queue is not None:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if event is None:  # sentinel — run_agent finally block guarantees this fires
                    break
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "done":
                    break
            # Emit final durable result from DB
            with Session(get_engine()) as session:
                db_wf = session.get(Workflow, workflow_id)
                final_payload = (WorkflowResponse.model_validate(db_wf).model_dump_json()
                                 if db_wf is not None else None)
            if final_payload:
                yield f"data: {final_payload}\n\n"
        else:
            # Fallback: DB polling for reconnect / LB failover / already-done without queue
            last_status = wf_status
            while True:
                if await request.is_disconnected():
                    break
                await asyncio.sleep(1)
                with Session(get_engine()) as session:
                    db_wf = session.get(Workflow, workflow_id)
                    if db_wf is None:
                        break
                    curr_status = db_wf.status
                    curr_payload = (WorkflowResponse.model_validate(db_wf).model_dump_json()
                                    if curr_status in _TERMINAL else None)
                if curr_status != last_status:
                    last_status = curr_status
                    if curr_status in _TERMINAL:
                        yield f"data: {curr_payload}\n\n"
                        break
                    yield (f'data: {{"type": "status", "status": "{curr_status.value}",'
                           f' "workflow_id": {workflow_id}}}\n\n')

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Agent run history ─────────────────────────────────────────────────────────

@router.get("/runs", response_model=list[AgentRunLogResponse])
async def list_runs(
    agent_id: Annotated[str | None, Query()] = None,
    outcome: Annotated[str | None, Query()] = None,
    triggered_by: Annotated[int | None, Query()] = None,
    from_date: Annotated[datetime | None, Query()] = None,
    to_date: Annotated[datetime | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    session: Session = Depends(get_session),
    _: User = Depends(require_role("analyst")),
):
    query = select(AgentRunLog)
    if agent_id:
        query = query.where(AgentRunLog.agent_id == agent_id)
    if outcome:
        query = query.where(AgentRunLog.outcome == outcome)
    if triggered_by is not None:
        query = query.where(AgentRunLog.triggered_by == triggered_by)
    if from_date:
        query = query.where(AgentRunLog.started_at >= from_date)
    if to_date:
        query = query.where(AgentRunLog.started_at <= to_date)
    query = query.order_by(AgentRunLog.started_at.desc()).offset(offset).limit(limit)
    return session.exec(query).all()


@router.get("/runs/{run_id}", response_model=AgentRunLogResponse)
async def get_run(
    run_id: str,
    session: Session = Depends(get_session),
    _: User = Depends(require_role("analyst")),
):
    run = session.exec(select(AgentRunLog).where(AgentRunLog.run_id == run_id)).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run


# ── Reload agents ─────────────────────────────────────────────────────────────

@router.post("/reload")
async def reload(user: User = Depends(require_role("admin"))):
    reload_agents()
    agents = list_agents()
    log.info("agents_reloaded", count=len(agents), triggered_by=user.username)
    return {"loaded": len(agents), "agents": [a.id for a in agents]}
