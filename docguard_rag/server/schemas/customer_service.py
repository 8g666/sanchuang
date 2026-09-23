from pydantic import BaseModel, Field

from docguard_rag.server.schemas.query import QuerySource


class CustomerServiceRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=128)
    document_ids: list[str] = Field(default_factory=list)
    collection_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    corrective: bool | None = None
    max_retries: int | None = Field(default=None, ge=0, le=3)
    handoff_if_unresolved: bool = True


class CustomerServiceResponse(BaseModel):
    session_id: str
    message: str
    intent: str
    intent_confidence: float
    status: str
    handoff_required: bool = False
    suggested_actions: list[str] = Field(default_factory=list)
    sources: list[QuerySource] = Field(default_factory=list)
    query_status: str | None = None
    query_verified: bool | None = None
