"""Create the DocGuard baseline schema.

Revision ID: 0001_initial
Revises: None
"""

from alembic import op

from docguard_rag.server.database import models  # noqa: F401
from docguard_rag.server.database.connection import Base

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The tracked SQLAlchemy metadata is the schema contract for the baseline.
    # Later revisions must use explicit Alembic operations for incremental changes.
    Base.metadata.create_all(bind=op.get_bind(), checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    for table in reversed(Base.metadata.sorted_tables):
        table.drop(bind=bind, checkfirst=True)
