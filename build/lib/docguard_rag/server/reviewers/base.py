from abc import ABC, abstractmethod

from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.models.issue import DocumentIssue


class DocumentReviewer(ABC):
    issue_types: set[str]

    @abstractmethod
    def review(
        self,
        block: DocumentBlock,
        review_id: str,
        requested_type: str,
    ) -> list[DocumentIssue]:
        raise NotImplementedError
