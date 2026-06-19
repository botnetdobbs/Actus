from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import cast
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import CursorResult, delete
from sqlmodel import Session, select, col
from app.config import get_settings
from app.database import get_engine
from app.context.store import ContextSnapshot
from app.context.models import Workflow, WorkflowStatus
from app.auth.models import AuditLog
from app.agents.builder import get_agent
from app.agents.orchestrator import run_agent_with_timeout, AGENT_TOTAL_TIMEOUT
import structlog

log = structlog.get_logger()

PURGE_AFTER_DAYS = 30
_UPLOAD_DIR = Path("/tmp/actus-uploads")
_DOC_CHUNK_TTL_HOURS = 2


async def purge_old_context_snapshots() -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=PURGE_AFTER_DAYS)
    with Session(get_engine()) as session:
        old = session.exec(
            select(ContextSnapshot).where(
                col(ContextSnapshot.is_deleted).is_(True),
                col(ContextSnapshot.deleted_at) < cutoff,
            )
        ).all()
        try:
            for snap in old:
                session.delete(snap)
            session.commit()
        except Exception as e:
            session.rollback()
            log.error("context_purge_failed", error=str(e))
            raise
        log.info("context_purge_complete", deleted=len(old))
        return len(old)


async def purge_old_audit_logs() -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=get_settings().audit_log_retention_days)
    with Session(get_engine()) as session:
        result = cast(CursorResult, session.execute(delete(AuditLog).where(col(AuditLog.timestamp) < cutoff)))
        session.commit()
    log.info("audit_logs_purged", deleted=result.rowcount)
    return result.rowcount


async def purge_orphan_doc_chunks() -> int:
    """Delete VectorIndex rows with object_type LIKE 'doc:%' older than TTL."""
    from app.rag.indexer import _is_postgres, delete_by_type
    from app.rag.models import VectorIndex

    if not _is_postgres():
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(hours=_DOC_CHUNK_TTL_HOURS)
    total_deleted = 0
    try:
        with Session(get_engine()) as session:
            stale_types = session.exec(
                select(VectorIndex.object_type)
                .where(col(VectorIndex.object_type).like("doc:%"))
                .where(VectorIndex.created_at < cutoff)
                .distinct()
            ).all()

        for type_name in stale_types:
            try:
                deleted = delete_by_type(type_name)
                total_deleted += deleted
            except Exception as e:
                log.error("doc_chunk_purge_failed", type_name=type_name, error=str(e))
    except Exception as e:
        log.error("doc_chunk_purge_query_failed", error=str(e))

    _purge_temp_uploads(cutoff)
    log.info("orphan_doc_chunks_purged", deleted=total_deleted)
    return total_deleted


def _purge_temp_uploads(cutoff: datetime) -> None:
    if not _UPLOAD_DIR.exists():
        return
    cutoff_ts = cutoff.timestamp()
    for f in _UPLOAD_DIR.iterdir():
        try:
            if f.is_file() and f.stat().st_mtime < cutoff_ts:
                f.unlink()
        except Exception as e:
            log.warning("temp_upload_unlink_failed", path=str(f), error=str(e))


async def _run_scheduled_agent(agent_id: str) -> None:
    """Run a named agent. Defined at module level so APScheduler can serialize it to the database."""
    try:
        config = get_agent(agent_id)
        result = await run_agent_with_timeout(config)
        log.info("scheduled_agent_complete", agent_id=agent_id, run_id=result.get("run_id"))
    except KeyError:
        log.error("scheduled_agent_not_found", agent_id=agent_id)
    except Exception as e:
        log.error("scheduled_agent_failed", agent_id=agent_id, error=str(e), exc_info=True)


_STUCK_WORKFLOW_BUFFER_SECONDS = 60
STUCK_WORKFLOW_TIMEOUT_SECONDS = AGENT_TOTAL_TIMEOUT + _STUCK_WORKFLOW_BUFFER_SECONDS


async def reap_stuck_workflows() -> int:
    """Mark workflows stuck in 'running' longer than the total agent timeout as failed.

    Targets crashes and process restarts that leave rows in running state permanently.
    Safe to run from multiple processes concurrently. Idempotent due to status filter.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=STUCK_WORKFLOW_TIMEOUT_SECONDS)
    with Session(get_engine()) as session:
        stuck = session.exec(
            select(Workflow).where(
                Workflow.status == WorkflowStatus.running,
                col(Workflow.started_at) < cutoff,
            )
        ).all()
        if not stuck:
            return 0
        now = datetime.now(timezone.utc)
        for wf in stuck:
            wf.status = WorkflowStatus.failed
            wf.completed_at = now
            wf.error = "Workflow timed out. Process restarted or crashed during run"
            session.add(wf)
            log.error(
                "stuck_workflow_reaped",
                workflow_id=wf.id,
                agent_id=wf.agent_id,
                started_at=str(wf.started_at),
            )
        try:
            session.commit()
        except Exception as e:
            session.rollback()
            log.error("stuck_workflow_reap_failed", error=str(e))
            raise
        log.info("stuck_workflows_reaped", count=len(stuck))
        return len(stuck)


async def heartbeat() -> None:
    try:
        with Session(get_engine()) as session:
            session.exec(select(1))
        log.info("heartbeat", db="ok")
    except Exception as e:
        log.error("heartbeat_db_failed", error=str(e), exc_info=True)


def register_all_jobs(scheduler: AsyncIOScheduler) -> None:
    scheduler.add_job(
        purge_old_context_snapshots,
        CronTrigger(hour=2, minute=0),
        id="purge_context_snapshots",
        replace_existing=True,
        misfire_grace_time=3600,
        coalesce=True,
    )
    scheduler.add_job(
        purge_old_audit_logs,
        CronTrigger(hour=3, minute=0),
        id="purge_old_audit_logs",
        replace_existing=True,
        misfire_grace_time=3600,
        coalesce=True,
    )
    scheduler.add_job(
        purge_orphan_doc_chunks,
        IntervalTrigger(hours=1),
        id="purge_orphan_doc_chunks",
        replace_existing=True,
        misfire_grace_time=3600,
        coalesce=True,
    )
    scheduler.add_job(
        reap_stuck_workflows,
        IntervalTrigger(minutes=5),
        id="reap_stuck_workflows",
        replace_existing=True,
        misfire_grace_time=120,
        coalesce=True,
    )
    scheduler.add_job(
        heartbeat,
        IntervalTrigger(seconds=60),
        id="heartbeat",
        replace_existing=True,
    )
