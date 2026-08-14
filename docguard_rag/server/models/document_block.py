from datetime import UTC, datetime

from pydantic import BaseModel, Field


class DocumentBlock(BaseModel):
    block_id: str
    document_id: str
    block_type: str
    text: str
    page_number: int | None = None
    section_path: list[str] = Field(default_factory=list)
    paragraph_index: int | None = None
    sentence_index: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
