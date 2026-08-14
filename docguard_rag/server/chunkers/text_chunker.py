from uuid import uuid4

from docguard_rag.server.chunkers.base import Chunker
from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.models.document_chunk import DocumentChunk


class TextChunker(Chunker):
    def __init__(self, chunk_size: int = 500, overlap: int = 100):

        self.chunk_size = chunk_size
        self.overlap = overlap

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be between zero and chunk_size - 1")

    def split(self, block: DocumentBlock) -> list[DocumentChunk]:

        text = block.text

        chunks = []

        start = 0

        index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                minimum_boundary = start + int(self.chunk_size * 0.6)
                candidates = [text.rfind(mark, minimum_boundary, end) for mark in "。！？!?；;\n"]
                natural_end = max(candidates, default=-1)
                if natural_end >= minimum_boundary:
                    end = natural_end + 1

            chunk_text = text[start:end]
            absolute_start = block.start_offset + start if block.start_offset is not None else start

            chunk = DocumentChunk(
                chunk_id=f"chunk_{uuid4().hex[:8]}",
                document_id=block.document_id,
                block_id=block.block_id,
                block_ids=[block.block_id],
                text=chunk_text,
                char_count=len(chunk_text),
                token_count=max(1, len(chunk_text) // 2),
                start_offset=absolute_start,
                end_offset=absolute_start + len(chunk_text),
                page_start=block.page_number,
                page_end=block.page_number,
                section_path=list(block.section_path),
                metadata={
                    **block.metadata,
                    "block_type": block.block_type,
                    "paragraph_index": block.paragraph_index,
                },
                chunk_index=index,
                split_method="character",
            )

            chunks.append(chunk)

            index += 1

            if end >= len(text):
                break

            start = end - self.overlap

        return chunks
