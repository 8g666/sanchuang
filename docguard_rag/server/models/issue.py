from datetime import UTC, datetime

from pydantic import BaseModel, Field


class IssueLocation(BaseModel):
    page: int | None = None
    section_path: list[str] = Field(default_factory=list)
    paragraph_index: int | None = None
    sentence_index: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None


class DocumentIssue(BaseModel):
    issue_id: str
    document_id: str
    review_id: str
    issue_type: str
    severity: str
    original_text: str
    problem_text: str
    suggestion: str
    reason: str
    confidence: float
    location: IssueLocation
    status: str = "PENDING"
    comment: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
