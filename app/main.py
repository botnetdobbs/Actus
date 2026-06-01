from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlmodel import Session, select
from app.config import get_settings
from app.database import create_db_and_tables, get_session
from app.llm.router import router as llm_router
from app.ontology.router import router as ontology_router
from app.auth.router import router as auth_router
from app.agents.builder import load_agents
from app.automation.router import router as automation_router
from app.automation.scheduler import start_scheduler, stop_scheduler
from app.observability.logging import configure_logging
from app.observability.metrics import instrument_app
from starlette.middleware.base import BaseHTTPMiddleware
import httpx
import structlog
import uuid

# Side-effect imports: register SQLModel table metadata before create_db_and_tables()
import app.ontology.models  # noqa: F401
import app.auth.models      # noqa: F401

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
    create_db_and_tables()
    try:
        load_agents()
    except Exception as e:
        log.error("agent_load_failed_at_startup", error=str(e))
        raise

    start_scheduler()

    yield

    stop_scheduler()


def create_app() -> FastAPI:
    app = FastAPI(
        title=_settings.app_name,
        version=_settings.app_version,
        lifespan=lifespan,
    )
    instrument_app(app)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_settings.cors_origins,
        allow_credentials=_settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth_router, prefix="/auth", tags=["Auth"])
    app.include_router(automation_router, prefix="/automation", tags=["Automation"])
    app.include_router(llm_router, prefix="/llm", tags=["LLM"])
    app.include_router(ontology_router, prefix="/ontology", tags=["Ontology"])

    @app.get("/healthz", tags=["Health"])
    async def health(session: Session = Depends(get_session)):
        checks = {}
        try:
            session.exec(select(1))
            checks["database"] = "ok"
        except Exception as e:
            checks["database"] = f"error: {e}"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{_settings.ollama_base_url}/api/tags")
                checks["ollama"] = "ok" if r.status_code == 200 else "unreachable"
        except Exception:
            checks["ollama"] = "unreachable"
        status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
        code = 200 if status == "ok" else 503
        return JSONResponse({"status": status, "checks": checks}, status_code=code)

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


app = create_app()
