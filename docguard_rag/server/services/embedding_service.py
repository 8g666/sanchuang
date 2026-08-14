from docguard_rag.server.core.provider_factory import create_embedding_provider
from docguard_rag.server.models.document_chunk import DocumentChunk
from docguard_rag.server.models.embedding import EmbeddingRecord


class EmbeddingService:
    def __init__(self, provider=None):
        self.provider = provider or create_embedding_provider()

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[EmbeddingRecord]:
        vectors = self.provider.embed_texts([chunk.text for chunk in chunks])
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
        return self.provider.embed_query(query)
