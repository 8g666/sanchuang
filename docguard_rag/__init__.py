from docguard_rag.client import DocGuardClient
from docguard_rag.errors import DocGuardAPIError, DocGuardError
from docguard_rag.local import DocGuard
from docguard_rag.types import (
    CorrectionIssue,
    CorrectionLocation,
    CorrectionResult,
    CustomerServiceResponse,
    DocumentDeleteResult,
    DocumentResult,
    DocumentStatus,
    Issue,
    IssueLocation,
    IssuePage,
    JobStatus,
    QueryAttempt,
    QueryResult,
    ReembedResult,
    ReviewResult,
    Source,
)

__version__ = "0.6.0"

__all__ = [
    "DocGuard",
    "DocGuardAPIError",
    "DocGuardClient",
    "DocGuardError",
    "DocumentResult",
    "DocumentDeleteResult",
    "DocumentStatus",
    "ReembedResult",
    "Issue",
    "IssueLocation",
    "IssuePage",
    "JobStatus",
    "QueryResult",
    "QueryAttempt",
    "ReviewResult",
    "Source",
    "CorrectionIssue",
    "CorrectionLocation",
    "CorrectionResult",
    "CustomerServiceResponse",
]
