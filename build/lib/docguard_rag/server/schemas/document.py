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
