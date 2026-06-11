from contextlib import asynccontextmanager
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlmodel import Session, select
from app.auth.jwt import get_current_user
from app.config import get_settings
from app.database import get_session
from app.limiter import limiter
from app.llm.router import router as llm_router
from app.ontology.router import router as ontology_router
from app.auth.router import router as auth_router
from app.agents.builder import load_agents
from app.agents.discovery import discover_models, discover_routers, discover_tools
from app.automation.router import router as automation_router
from app.automation.scheduler import scheduler, start_scheduler, stop_scheduler
from app.observability.logging import configure_logging
from app.observability.metrics import instrument_app
from app import pubsub
from starlette.middleware.base import BaseHTTPMiddleware
import httpx
import structlog
import uuid

# Side-effect imports: register SQLModel table metadata before create_db_and_tables()
import app.ontology.models  # noqa: F401
import app.auth.models      # noqa: F401
import app.agents.audit     # noqa: F401
discover_models()            # registers app/agents/*/models.py table metadata

_settings = get_settings()

log = structlog.get_logger()


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            path=request.url.path,
            method=request.method,
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(debug=_settings.debug)
    from app.llm import pii  # noqa: F401 — Presidio NLP model load, controls when the 2-3s cost is paid
    from app.rag.embedder import warmup as warmup_embedder
    warmup_embedder()
    await pubsub.init_redis(_settings.redis_url)
    discover_tools()
    try:
        load_agents()
    except Exception as e:
        log.error("agent_load_failed_at_startup", error=str(e))
        raise

    if _settings.scheduler_enabled:
        start_scheduler()

    try:
        yield
    finally:
        if _settings.scheduler_enabled:
            await stop_scheduler()
        await pubsub.close_redis()


def create_app() -> FastAPI:
    app = FastAPI(
        title=_settings.app_name,
        version=_settings.app_version,
        lifespan=lifespan,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    instrument_app(app)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_settings.cors_origins,
        allow_credentials=_settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    v1 = APIRouter(prefix="/v1")
    v1.include_router(auth_router, prefix="/auth", tags=["Auth"])
    v1.include_router(automation_router, prefix="/automation", tags=["Automation"])
    v1.include_router(llm_router, prefix="/llm", tags=["LLM"],
                      dependencies=[Depends(get_current_user)])
    v1.include_router(ontology_router, prefix="/ontology", tags=["Ontology"],
                      dependencies=[Depends(get_current_user)])
    discover_routers(v1)    # registers app/agents/*/router.py
    app.include_router(v1)

    @app.get("/healthz", tags=["Health"])
    async def health(session: Session = Depends(get_session)):
        core: dict[str, str] = {}
        info: dict[str, str] = {}

        # Core checks — failures here return 503
        try:
            session.exec(select(1))
            core["database"] = "ok"
        except Exception as e:
            core["database"] = f"error: {e}"

        if _settings.scheduler_enabled:
            core["scheduler"] = "ok" if scheduler.running else "stopped"

        # Informational checks — reported but never cause 503
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{_settings.ollama_base_url}/api/tags")
                info["ollama"] = "ok" if r.status_code == 200 else "unreachable"
        except Exception:
            info["ollama"] = "unreachable"

        redis_result = await pubsub.ping()
        if redis_result is None:
            info["redis"] = "not_configured"
        elif redis_result:
            info["redis"] = "ok"
        else:
            info["redis"] = "error"

        healthy = all(v == "ok" for v in core.values())
        status = "ok" if healthy else "degraded"
        return JSONResponse(
            {"status": status, "core": core, "info": info},
            status_code=200 if healthy else 503,
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        log.error(
            "unhandled_exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    return app


app = create_app()  # type: ignore[assignment]
