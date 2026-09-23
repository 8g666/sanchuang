from docguard_rag.server.core.config import get_settings
from docguard_rag.server.core.embedding_cache import query_embedding_cache
from docguard_rag.server.core.provider_errors import ProviderError
from docguard_rag.server.core.provider_factory import create_embedding_provider
from docguard_rag.server.models.document_chunk import DocumentChunk
from docguard_rag.server.models.embedding import EmbeddingRecord


class EmbeddingService:
    def __init__(self, provider=None):
        self.provider = provider or create_embedding_provider()

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[EmbeddingRecord]:
        vectors = self._validated(self.provider.embed_texts([chunk.text for chunk in chunks]))
        return [
            EmbeddingRecord(
                chunk_id=chunk.chunk_id,
                vector=vector,
                metadata={
                    "document_id": chunk.document_id,
                    "tenant_id": chunk.metadata.get("tenant_id", "default"),
                    "owner_id": chunk.metadata.get("owner_id"),
                    "block_id": chunk.block_id,
                    "text": chunk.text,
                    "page": chunk.page_start,
                    "section_path": chunk.section_path,
                    "paragraph_index": chunk.metadata.get("paragraph_index"),
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                    "chunk_index": chunk.chunk_index,
                    "collection_id": chunk.metadata.get("collection_id"),
                    "custom_metadata": chunk.metadata.get("custom_metadata", {}),
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    def embed_query(self, query: str) -> list[float]:
        fingerprint = getattr(self.provider, "fingerprint", None)
        if fingerprint is None:
            # Third-party providers need only implement EmbeddingProvider; making
            # `fingerprint` an abstract method would break every existing one.
            return self._validated([self.provider.embed_query(query)])[0]
        # Validation sits inside the computation, so a ProviderError propagates before the
        # cache stores anything and the next call retries instead of replaying the failure.
        return query_embedding_cache().get_or_compute(
            (fingerprint, query),
            lambda: self._validated([self.provider.embed_query(query)])[0],
        )

    @staticmethod
    def _validated(vectors: list[list[float]]) -> list[list[float]]:
        """Reject a width mismatch at the write and read boundary, not at startup.

        A wrong DOCGUARD_EMBEDDING_DIMENSION is deterministic and cannot self-heal, so
        the first ingest or query catches it within milliseconds of real use. Probing at
        boot instead would make startup depend on provider availability.
        """

        expected = get_settings().embedding_dimension
        for vector in vectors:
            if len(vector) != expected:
                raise ProviderError(
                    "Embedding dimension mismatch: "
                    f"expected {expected}, provider returned {len(vector)}"
                )
        return vectors
