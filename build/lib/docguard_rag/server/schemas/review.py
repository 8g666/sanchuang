from typing import Literal

from pydantic import BaseModel, Field

ReviewMode = Literal["FAST", "BALANCED", "STRICT"]
IssueStatus = Literal["PENDING", "ACCEPTED", "REJECTED", "IGNORED", "FIXED"]


class ReviewRequest(BaseModel):
    document_ids: list[str] = Field(min_length=1)
    check_types: list[str] = Field(default_factory=lambda: ["TYPO", "SEMANTIC_UNCLEAR"])
    scope: str = "FULL_DOCUMENT"
    mode: ReviewMode = "BALANCED"
    async_mode: bool = Field(default=False, alias="async")


class ReviewResult(BaseModel):
    review_id: str
    job_id: str | None = None
    status: str
    issue_count: int


class IssueStatusUpdate(BaseModel):
    status: IssueStatus
    comment: str | None = Field(default=None, max_length=1000)


class IssueLocation(BaseModel):
    page: int | None = None
    section_path: list[str] = Field(default_factory=list)
    paragraph_index: int | None = None
    sentence_index: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None


class IssueResult(BaseModel):
    issue_id: str
    document_id: str
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


class IssuePageResult(BaseModel):
    total: int
    items: list[IssueResult] = Field(default_factory=list)
