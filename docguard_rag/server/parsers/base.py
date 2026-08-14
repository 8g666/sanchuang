from abc import ABC, abstractmethod

from docguard_rag.server.models.document_block import DocumentBlock


class DocumentParser(ABC):
    @abstractmethod
    def supports(self, file_type: str) -> bool:
        """Return whether this parser supports the normalized file type."""

        raise NotImplementedError

    @abstractmethod
    def parse(self, file_path: str, document_id: str) -> list[DocumentBlock]:
        """Parse a file into normalized document blocks."""

        raise NotImplementedError
