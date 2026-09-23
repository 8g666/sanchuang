"""Persist /v1/support/chat conversation history.

CustomerServiceService kept history in a process-local defaultdict of deques, so it was
lost on restart, invisible to every other worker, and grew without bound by session count.
Two tables replace it: support_sessions carries the tenant ownership and lifecycle that a
flat message log cannot express, support_messages carries the turns.

Revision ID: 0003_support_sessions
Revises: 0002_corpus_created_at_index
"""

import sqlalchemy as sa
from alembic import op

revision = "0003_support_sessions"
down_revision = "0002_corpus_created_at_index"
branch_labels = None
depends_on = None

SESSION_INDEXES = (
    ("ix_support_sessions_session_id", ["session_id"]),
    ("ix_support_sessions_tenant_id", ["tenant_id"]),
    ("ix_support_sessions_status", ["status"]),
    ("ix_support_sessions_last_message_at", ["last_message_at"]),
)
MESSAGE_INDEXES = (
    ("ix_support_messages_session_id", ["session_id"]),
    ("ix_support_messages_tenant_id", ["tenant_id"]),
    ("ix_support_messages_session_seq", ["tenant_id", "session_id", "id"]),
)


def upgrade() -> None:
    # 0001 builds the baseline from current SQLAlchemy metadata, so a database migrated
    # from a clean slate already carries both tables. Skipping when present keeps that
    # path and a genuine incremental upgrade working from one revision.
    existing = set(sa.inspect(op.get_bind()).get_table_names())

    if "support_sessions" not in existing:
        op.create_table(
            "support_sessions",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("session_id", sa.String(length=128), nullable=False),
            sa.Column("tenant_id", sa.String(length=64), nullable=False),
            sa.Column("status", sa.String(length=16), nullable=False),
            sa.Column("message_count", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("last_message_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("tenant_id", "session_id", name="uq_support_session"),
        )
    for name, columns in SESSION_INDEXES:
        op.create_index(name, "support_sessions", columns, if_not_exists=True)

    if "support_messages" not in existing:
        op.create_table(
            "support_messages",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("session_id", sa.String(length=128), nullable=False),
            sa.Column("tenant_id", sa.String(length=64), nullable=False),
            sa.Column("role", sa.String(length=16), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("intent", sa.String(length=32), nullable=True),
            sa.Column("status", sa.String(length=32), nullable=True),
            sa.Column("request_id", sa.String(length=64), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
    for name, columns in MESSAGE_INDEXES:
        op.create_index(name, "support_messages", columns, if_not_exists=True)


def downgrade() -> None:
    for name, _ in reversed(MESSAGE_INDEXES):
        op.drop_index(name, table_name="support_messages", if_exists=True)
    op.drop_table("support_messages", if_exists=True)
    for name, _ in reversed(SESSION_INDEXES):
        op.drop_index(name, table_name="support_sessions", if_exists=True)
    op.drop_table("support_sessions", if_exists=True)
