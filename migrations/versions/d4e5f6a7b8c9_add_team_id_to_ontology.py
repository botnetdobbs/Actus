"""add team_id to ontology objects and vector index

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-06-11

Adds:
  - customers.team_id (nullable, indexed)
  - vector_indexes.team_id (nullable, indexed)

NULL means global/unscoped, following the same convention as
users.team_id / workflows.team_id / agent_run_logs.team_id.
"""
from alembic import op
import sqlalchemy as sa

revision = "d4e5f6a7b8c9"
down_revision = "c3d4e5f6a7b8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("customers") as batch_op:
        batch_op.add_column(sa.Column("team_id", sa.Integer, nullable=True))
        batch_op.create_index("ix_customers_team_id", ["team_id"])

    with op.batch_alter_table("vector_indexes") as batch_op:
        batch_op.add_column(sa.Column("team_id", sa.Integer, nullable=True))
        batch_op.create_index("ix_vector_indexes_team_id", ["team_id"])


def downgrade() -> None:
    with op.batch_alter_table("vector_indexes") as batch_op:
        batch_op.drop_index("ix_vector_indexes_team_id")
        batch_op.drop_column("team_id")

    with op.batch_alter_table("customers") as batch_op:
        batch_op.drop_index("ix_customers_team_id")
        batch_op.drop_column("team_id")
