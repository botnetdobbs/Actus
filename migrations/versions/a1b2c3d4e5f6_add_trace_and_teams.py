"""add trace and teams

Revision ID: a1b2c3d4e5f6
Revises: 8d66929aeff9
Create Date: 2026-06-09

Adds:
  - teams table
  - users.team_id (FK to teams)
  - workflows.team_id
  - agent_run_logs.team_id
  - agent_run_logs.trace_json
"""
from alembic import op
import sqlalchemy as sa

revision = "a1b2c3d4e5f6"
down_revision = "8d66929aeff9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "teams",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String, nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.Integer, nullable=True),
        sa.Column("is_deleted", sa.Boolean, nullable=False, server_default="false"),
    )
    op.create_index("ix_teams_name", "teams", ["name"])

    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(sa.Column("team_id", sa.Integer, sa.ForeignKey("teams.id"), nullable=True))

    with op.batch_alter_table("workflows") as batch_op:
        batch_op.add_column(sa.Column("team_id", sa.Integer, nullable=True))

    with op.batch_alter_table("agent_run_logs") as batch_op:
        batch_op.add_column(sa.Column("team_id", sa.Integer, nullable=True))
        batch_op.add_column(sa.Column("trace_json", sa.Text, nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("agent_run_logs") as batch_op:
        batch_op.drop_column("trace_json")
        batch_op.drop_column("team_id")

    with op.batch_alter_table("workflows") as batch_op:
        batch_op.drop_column("team_id")

    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_column("team_id")

    op.drop_index("ix_teams_name", "teams")
    op.drop_table("teams")
