from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlmodel import Session, select
from app.config import get_settings
from app.database import create_db_and_tables, get_session
from app.llm.router import router as llm_router
from app.ontology.router import router as ontology_router
from starlette.middleware.base import BaseHTTPMiddleware
import httpx
import structlog
import uuid

log = structlog.get_logger()


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    import app.llm.pii       # triggers Presidio NLP model load at startup
    import app.ontology.models  # registers all ontology types before table creation
    create_db_and_tables()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
                r = await client.get(f"{settings.ollama_base_url}/api/tags")
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
