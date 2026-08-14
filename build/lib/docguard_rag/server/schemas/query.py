from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] = Field(default_factory=list)
    collection_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    generate_answer: bool = True
    include_sources: bool = True
    filters: dict = Field(default_factory=dict)
    corrective: bool | None = None
    max_retries: int | None = Field(default=None, ge=0, le=3)


class QuerySource(BaseModel):
    chunk_id: str
    document_id: str
    filename: str | None = None
    page: int | None = None
    section_path: list[str] = Field(default_factory=list)
    paragraph_index: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    text: str
    score: float
    relevance_score: float | None = None


class QueryAttempt(BaseModel):
    attempt: int
    query: str
    retrieved_count: int = 0
    relevant_count: int = 0
    evidence_score: float = 0.0
    evidence_sufficient: bool = False
    answer_verified: bool | None = None
    verification_score: float | None = None
    reason: str | None = None


class QueryResult(BaseModel):
    answer: str | None
    confidence: float
    sources: list[QuerySource] = Field(default_factory=list)
    status: str = "ANSWERED"
    original_query: str | None = None
    effective_query: str | None = None
    query_rewritten: bool = False
    attempt_count: int = 1
    verified: bool | None = None
    attempts: list[QueryAttempt] = Field(default_factory=list)
