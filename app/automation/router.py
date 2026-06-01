import asyncio
import json
from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlmodel import Session
from app.agents.builder import get_agent
from app.agents.orchestrator import run_agent
from app.auth.models import User
from app.auth.jwt import require_role, write_audit_log
from app.context.models import Workflow, WorkflowStatus
from app.database import get_engine
import structlog

log = structlog.get_logger()
router = APIRouter()


async def _run_workflow(workflow_id: int) -> None:
    with Session(get_engine()) as session:
        wf = session.get(Workflow, workflow_id)
        wf.status = WorkflowStatus.running
        wf.started_at = datetime.now(timezone.utc)
        session.add(wf)
        session.commit()

    status = WorkflowStatus.failed
    result = None
    error = None
    try:
        config = get_agent(wf.agent_id)
        result = await run_agent(config)
        status = WorkflowStatus.completed
    except asyncio.TimeoutError:
        status = WorkflowStatus.timeout
        error = "Agent run timed out"
        log.error("workflow_timeout", workflow_id=workflow_id, agent_id=wf.agent_id)
    except Exception as e:
        error = str(e)
        log.error("workflow_failed", workflow_id=workflow_id, agent_id=wf.agent_id, error=error)
    finally:
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


@router.post("/trigger/{agent_id}", status_code=202)
async def trigger_agent(
    agent_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    user: User = Depends(require_role("analyst")),
):
    try:
        config = get_agent(agent_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Agent not found: '{agent_id}'")

    with Session(get_engine()) as session:
        wf = Workflow(name=config.name, agent_id=agent_id, created_by=user.id)
        session.add(wf)
        session.commit()
        session.refresh(wf)
        workflow_id = wf.id

    background_tasks.add_task(_run_workflow, workflow_id)

    ip = request.client.host if request.client else None
    write_audit_log(
        username=user.username,
        action="agent_trigger",
        resource=agent_id,
        ip=ip,
        detail=f"workflow_id={workflow_id}",
    )

    return {"status": "queued", "agent_id": agent_id, "workflow_id": workflow_id}
