import importlib
from pathlib import Path

import structlog

log = structlog.get_logger()


def _agents_root() -> Path:
    import app as _app_pkg
    return Path(_app_pkg.__file__).parent / "agents"


def discover_tools() -> None:
    """Auto-import every app/agents/*/tools.py to register @tool decorators."""
    for subdir in sorted(_agents_root().iterdir()):
        if not subdir.is_dir() or not (subdir / "tools.py").exists():
            continue
        module_name = f"app.agents.{subdir.name}.tools"
        try:
            importlib.import_module(module_name)
            log.debug("tools_discovered", module=module_name)
        except Exception as e:
            log.error("tools_discovery_failed", module=module_name, error=str(e))
            raise


def discover_routers(app) -> None:
    """Auto-register every app/agents/*/router.py that exports an APIRouter.

    Each router.py must declare prefix and tags on the APIRouter constructor:
        router = APIRouter(prefix="/my-agent", tags=["My Agent"])
    """
    for subdir in sorted(_agents_root().iterdir()):
        if not subdir.is_dir() or not (subdir / "router.py").exists():
            continue
        module_name = f"app.agents.{subdir.name}.router"
        try:
            module = importlib.import_module(module_name)
            router = getattr(module, "router", None)
            if router is None:
                log.warning("router_missing_export", module=module_name)
                continue
            app.include_router(router)
            log.debug("router_registered", module=module_name)
        except Exception as e:
            log.error("router_discovery_failed", module=module_name, error=str(e))
            raise


def discover_models() -> None:
    """Auto-import every app/agents/*/models.py for SQLModel table registration."""
    for subdir in sorted(_agents_root().iterdir()):
        if not subdir.is_dir() or not (subdir / "models.py").exists():
            continue
        module_name = f"app.agents.{subdir.name}.models"
        try:
            importlib.import_module(module_name)
            log.debug("models_discovered", module=module_name)
        except Exception as e:
            log.error("models_discovery_failed", module=module_name, error=str(e))
            raise
