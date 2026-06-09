"""add team_id indexes

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-06-09

Adds indexes on team_id columns that are missing from the previous migration.
Without these, team-scoped visibility queries do full table scans.
"""
from alembic import op

revision = "b2c3d4e5f6a7"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_workflows_team_id", "workflows", ["team_id"])
    op.create_index("ix_agent_run_logs_team_id", "agent_run_logs", ["team_id"])


def downgrade() -> None:
    op.drop_index("ix_agent_run_logs_team_id", "agent_run_logs")
    op.drop_index("ix_workflows_team_id", "workflows")
