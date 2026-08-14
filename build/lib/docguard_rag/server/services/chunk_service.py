from docguard_rag.server.chunkers.text_chunker import TextChunker
from docguard_rag.server.core.config import get_settings
from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.models.document_chunk import DocumentChunk


class ChunkService:
    indexable_types = {"paragraph", "list_item", "table", "table_row", "caption"}

    def __init__(self, chunker: TextChunker | None = None):
        settings = get_settings()
        self.chunker = chunker or TextChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

    def create_chunks(self, blocks: list[DocumentBlock]) -> list[DocumentChunk]:
        chunks = [
            chunk
            for block in blocks
            if block.block_type in self.indexable_types
            for chunk in self.chunker.split(block)
        ]
        for index, chunk in enumerate(chunks):
            chunk.chunk_index = index
        return chunks
