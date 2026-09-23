from pydantic import BaseModel, Field


class DocumentResult(BaseModel):
    document_id: str
    job_id: str | None = None
    review_id: str | None = None
    status: str
    blocks_count: int = 0
    chunks_count: int = 0
    vectors_count: int = 0


class DocumentStatus(BaseModel):
    document_id: str
    filename: str
    file_type: str
    status: str
    page_count: int | None = None
    chunk_count: int = 0
    sentence_count: int = 0


class DocumentDeleteResult(BaseModel):
    document_id: str
    status: str = "DELETED"


class ReembedResult(BaseModel):
    reembedded: int = 0
    embedded_without_offsets: int = 0
    document_ids_needing_reindex: list[str] = Field(default_factory=list)
    batches: int = 0
    dimension: int = 0
    job_id: str | None = None
    status: str = "COMPLETED"


class CustomerServiceResponse(BaseModel):
    session_id: str
    message: str
    intent: str
    intent_confidence: float
    status: str
    handoff_required: bool = False
    suggested_actions: list[str] = Field(default_factory=list)
    sources: list["Source"] = Field(default_factory=list)
    query_status: str | None = None
    query_verified: bool | None = None


class Source(BaseModel):
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
    answer: str | None = None
    confidence: float = 0.0
    sources: list[Source] = Field(default_factory=list)
    status: str = "ANSWERED"
    original_query: str | None = None
    effective_query: str | None = None
    query_rewritten: bool = False
    attempt_count: int = 1
    verified: bool | None = None
    attempts: list[QueryAttempt] = Field(default_factory=list)


class ReviewResult(BaseModel):
    review_id: str
    job_id: str | None = None
    status: str
    issue_count: int = 0


class IssueLocation(BaseModel):
    page: int | None = None
    section_path: list[str] = Field(default_factory=list)
    paragraph_index: int | None = None
    sentence_index: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None


class Issue(BaseModel):
    issue_id: str
    document_id: str | None = None
    issue_type: str
    severity: str
    original_text: str
    problem_text: str
    suggestion: str
    reason: str
    confidence: float
    location: IssueLocation
    status: str
    comment: str | None = None


class IssuePage(BaseModel):
    total: int
    items: list[Issue] = Field(default_factory=list)


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    current_stage: str | None = None
    error: str | None = None


class CorrectionLocation(BaseModel):
    paragraph_index: int | None = None
    sentence_index: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None


class CorrectionIssue(BaseModel):
    issue_id: str
    issue_type: str
    severity: str
    original_text: str
    problem_text: str
    suggestion: str
    reason: str
    confidence: float
    location: CorrectionLocation


class CorrectionResult(BaseModel):
    original_text: str
    corrected_text: str
    changed: bool
    issues: list[CorrectionIssue] = Field(default_factory=list)
