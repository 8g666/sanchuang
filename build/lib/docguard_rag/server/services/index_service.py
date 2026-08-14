from sqlalchemy.orm import Session

from docguard_rag.server.models.document_chunk import DocumentChunk
from docguard_rag.server.models.embedding import EmbeddingRecord
from docguard_rag.server.services.embedding_service import EmbeddingService
from docguard_rag.server.services.vector_service import VectorService


class IndexService:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_service: VectorService | None = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_service = vector_service or VectorService()

    def build_index(
        self,
        db: Session,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddingRecord]:
        embeddings = self.embedding_service.embed_chunks(chunks)
        self.vector_service.add_vectors(embeddings, db=db)
        return embeddings
