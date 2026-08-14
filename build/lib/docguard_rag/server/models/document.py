from datetime import UTC, datetime

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size: int
    file_hash: str
    status: str
    created_at: datetime
    tenant_id: str = "default"
    owner_id: str | None = None
    collection_id: str | None = None
    storage_path: str | None = None
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def new_timestamp(cls) -> datetime:
        return datetime.now(UTC)
