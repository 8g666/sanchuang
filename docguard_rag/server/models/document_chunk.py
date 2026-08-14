from datetime import UTC, datetime

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    block_id: str
    block_ids: list[str] = Field(default_factory=list)
    text: str
    char_count: int
    token_count: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    chunk_index: int
    split_method: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
