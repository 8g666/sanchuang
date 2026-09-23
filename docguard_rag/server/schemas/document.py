from pydantic import BaseModel, Field


class DocumentUploadResult(BaseModel):
    document_id: str
    job_id: str | None = None
    review_id: str | None = None
    status: str
    blocks_count: int
    chunks_count: int
    vectors_count: int


class DocumentDeleteResult(BaseModel):
    document_id: str
    status: str = "DELETED"


class ReembedRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list)
    batch_size: int | None = None
    async_mode: bool = Field(default=False, alias="async")
    model_config = {"populate_by_name": True}


class ReembedResult(BaseModel):
    reembedded: int = 0
    # Chunks with no previous vector row: embedded, but start_offset/end_offset are
    # derived from a chunk's position inside its block and never persisted, so they are
    # unrecoverable from SQL. `POST /documents/{id}/reindex` reproduces them exactly.
    embedded_without_offsets: int = 0
    document_ids_needing_reindex: list[str] = Field(default_factory=list)
    batches: int = 0
    dimension: int = 0
    job_id: str | None = None
    status: str = "COMPLETED"


class DocumentStatus(BaseModel):
    document_id: str
    filename: str
    file_type: str
    status: str
    page_count: int | None = None
    chunk_count: int
    sentence_count: int


class DocumentBlockItem(BaseModel):
    block_id: str
    block_type: str
    text: str
    page_number: int | None = None
    section_path: list[str] = Field(default_factory=list)
    paragraph_index: int | None = None
    sentence_index: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None


class DocumentBlocksResult(BaseModel):
    document_id: str
    blocks: list[DocumentBlockItem] = Field(default_factory=list)
    blocks_count: int = 0


class DocumentChunkItem(BaseModel):
    chunk_id: str
    document_id: str
    block_id: str
    text: str
    chunk_index: int


class DocumentChunksResult(BaseModel):
    chunks: list[DocumentChunkItem] = Field(default_factory=list)
