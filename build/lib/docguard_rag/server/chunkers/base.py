from abc import ABC, abstractmethod

from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.models.document_chunk import DocumentChunk


class Chunker(ABC):
    @abstractmethod
    def split(self, block: DocumentBlock) -> list[DocumentChunk]:
        """Split one document block into indexable chunks."""

        raise NotImplementedError
