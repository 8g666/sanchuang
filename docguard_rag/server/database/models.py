from datetime import datetime

from sqlalchemy import DateTime, Float, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from docguard_rag.server.database.connection import Base


class DocumentTable(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default", index=True)
    owner_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    collection_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    file_type: Mapped[str] = mapped_column(String(32))
    file_hash: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32))
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    storage_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class DocumentBlockTable(Base):
    __tablename__ = "document_blocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    block_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    document_id: Mapped[str] = mapped_column(String(64), index=True)
    block_type: Mapped[str] = mapped_column(String(32))
    text: Mapped[str] = mapped_column(Text)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    paragraph_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sentence_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    start_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class DocumentChunkTable(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    document_id: Mapped[str] = mapped_column(String(64), index=True)
    block_id: Mapped[str] = mapped_column(String(64), index=True)
    text: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    split_method: Mapped[str] = mapped_column(String(32))
    # Indexed: the retrieval cache probes MAX(created_at) on every search.
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class DocumentVectorTable(Base):
    __tablename__ = "document_vectors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    vector: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Indexed: rows are wide, so an unindexed MAX(created_at) probe dominates search time.
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class ReviewJobTable(Base):
    __tablename__ = "review_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    review_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default", index=True)
    document_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="PROCESSING")
    mode: Mapped[str] = mapped_column(String(32), default="BALANCED")
    check_types: Mapped[str] = mapped_column(Text)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class DocumentIssueTable(Base):
    __tablename__ = "document_issues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    issue_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    document_id: Mapped[str] = mapped_column(String(64), index=True)
    review_id: Mapped[str] = mapped_column(String(64), index=True)
    issue_type: Mapped[str] = mapped_column(String(32), index=True)
    severity: Mapped[str] = mapped_column(String(16))
    original_text: Mapped[str] = mapped_column(Text)
    problem_text: Mapped[str] = mapped_column(Text)
    suggestion: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    paragraph_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sentence_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    start_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="PENDING", index=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class JobTable(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default", index=True)
    job_type: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(32), default="PENDING", index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    current_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class QueryLogTable(Base):
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default", index=True)
    query_text: Mapped[str] = mapped_column(Text)
    document_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    retrieved_chunk_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    token_usage_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class SupportSessionTable(Base):
    """One /v1/support/chat conversation.

    The logical key is (tenant_id, session_id), not session_id alone: callers may supply
    their own session_id, and treating a colliding one as globally unique would let one
    tenant read another tenant's history.
    """

    __tablename__ = "support_sessions"
    __table_args__ = (UniqueConstraint("tenant_id", "session_id", name="uq_support_session"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # 128, not the String(64) used elsewhere: CustomerServiceRequest.session_id already
    # allows 128, and narrowing the column would silently truncate or reject valid input.
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default", index=True)
    status: Mapped[str] = mapped_column(String(16), default="ACTIVE", index=True)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    last_message_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class SupportMessageTable(Base):
    """One turn of a support conversation, stored as separate user and assistant rows."""

    __tablename__ = "support_messages"
    # Covers the hot read path. Ordered by id, not created_at: the latter has only
    # millisecond precision and a turn's two rows are written within the same request.
    __table_args__ = (Index("ix_support_messages_session_seq", "tenant_id", "session_id", "id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # No ForeignKey, matching every other table here; scoping is enforced in the
    # repository by always filtering on tenant_id together with session_id.
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default", index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    intent: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
