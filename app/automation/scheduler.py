import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from apscheduler.triggers.cron import CronTrigger
from app.config import get_settings
from app.agents.builder import list_agents
from app.automation.jobs import register_all_jobs, _make_agent_job
import structlog

log = structlog.get_logger()

_settings = get_settings()

scheduler = AsyncIOScheduler(
    jobstores={"default": SQLAlchemyJobStore(url=_settings.database_url)},
    timezone="UTC",
)


def job_listener(event) -> None:
    if event.exception:
        log.error("scheduled_job_failed",
                  job_id=event.job_id,
                  error=str(event.exception),
                  traceback=str(event.traceback))
    else:
        log.info("scheduled_job_complete",
                 job_id=event.job_id,
                 retval=str(event.retval))


def start_scheduler() -> None:
    scheduler.add_listener(job_listener, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)
    register_all_jobs(scheduler)
    for config in list_agents():
        if config.schedule and config.schedule.cron:
            scheduler.add_job(
                _make_agent_job(config.id),
                CronTrigger.from_crontab(config.schedule.cron),
                id=f"agent_{config.id}",
                replace_existing=True,
                misfire_grace_time=3600,
                coalesce=True,
            )
            log.info("agent_job_registered", agent_id=config.id, cron=config.schedule.cron)
    scheduler.start()
    log.info("scheduler_started", job_count=len(scheduler.get_jobs()))


async def stop_scheduler(timeout: float = 30.0) -> None:
    if scheduler.running:
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(loop.run_in_executor(None, scheduler.shutdown, True), timeout=timeout)
        except asyncio.TimeoutError:
            log.warning("scheduler_shutdown_timeout", timeout=timeout)
        log.info("scheduler_stopped")
