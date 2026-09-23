"""Index created_at on the corpus tables.

The retrieval cache validates itself on every search with MAX(created_at) over
document_chunks and document_vectors. Without an index that probe is a full scan, and
document_vectors rows are wide enough (serialized vector plus chunk text) for it to
dominate search latency.

Revision ID: 0002_corpus_created_at_index
Revises: 0001_initial
"""

from alembic import op

revision = "0002_corpus_created_at_index"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

# Names must match what SQLAlchemy derives from index=True on the model columns.
INDEXES = (
    ("ix_document_chunks_created_at", "document_chunks"),
    ("ix_document_vectors_created_at", "document_vectors"),
)


def upgrade() -> None:
    # 0001 builds the schema from SQLAlchemy metadata, so a database migrated from a
    # clean slate already carries these indexes. if_not_exists keeps both paths working.
    for name, table in INDEXES:
        op.create_index(name, table, ["created_at"], if_not_exists=True)


def downgrade() -> None:
    for name, table in reversed(INDEXES):
        op.drop_index(name, table_name=table, if_exists=True)
