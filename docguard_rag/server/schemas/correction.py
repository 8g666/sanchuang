from pydantic import BaseModel, Field


class CorrectionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200_000)
    check_types: list[str] = Field(default_factory=lambda: ["TYPO", "GRAMMAR", "SEMANTIC_UNCLEAR"])
    mode: str = Field(default="BALANCED", pattern="^(FAST|BALANCED|STRICT)$")
    apply_changes: bool = True


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
