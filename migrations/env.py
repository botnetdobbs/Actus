from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from sqlmodel import SQLModel
from alembic import context

# Register all table models so Alembic can detect the full schema
import app.auth.models       # noqa: F401
import app.ontology.models   # noqa: F401
import app.agents.audit      # noqa: F401
import app.context.store     # noqa: F401
import app.context.models    # noqa: F401
import app.rag.models         # noqa: F401

from app.config import get_settings

alembic_config = context.config

if alembic_config.config_file_name is not None:
    fileConfig(alembic_config.config_file_name)

# Use DATABASE_URL from app settings so alembic.ini doesn't need it hardcoded
alembic_config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = SQLModel.metadata

# Tables managed outside of Alembic (e.g. by APScheduler) must be excluded
# from autogenerate so Alembic doesn't try to drop them.
_EXCLUDED_TABLES = {"apscheduler_jobs"}


def include_object(object, name, type_, reflected, compare_to):
    if type_ == "table" and name in _EXCLUDED_TABLES:
        return False
    return True


def run_migrations_offline() -> None:
    url = alembic_config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        alembic_config.get_section(alembic_config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
