from collections.abc import Iterator

from docguard_rag.server.chunkers.text_chunker import TextChunker
from docguard_rag.server.core.config import get_settings
from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.models.document_chunk import DocumentChunk

# Joining with a newline keeps the merged text readable and gives TextChunker a boundary it
# already treats as a natural cut point.
SEPARATOR = "\n"


class ChunkService:
    indexable_types = {"paragraph", "list_item", "table", "table_row", "caption"}

    def __init__(self, chunker: TextChunker | None = None):
        settings = get_settings()
        self.chunker = chunker or TextChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

    def create_chunks(self, blocks: list[DocumentBlock]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for group in self._group(blocks):
            segment, spans = self._merge(group)
            for chunk in self.chunker.split(segment):
                chunk.block_ids = [block.block_id for block in group]
                self._restore_offsets(chunk, segment, spans)
                chunks.append(chunk)
        for index, chunk in enumerate(chunks):
            chunk.chunk_index = index
        return chunks

    def _group(self, blocks: list[DocumentBlock]) -> Iterator[list[DocumentBlock]]:
        """Pack consecutive indexable blocks into segments of about chunk_size characters.

        TextChunker.split only ever cuts a single block apart, so without this a document
        parsed into many short blocks yields one fragment chunk per block and the configured
        chunk_size is ignored. Real corpora are dominated by short blocks — a slide-deck PDF
        averages 11 characters per block, a line-oriented Markdown file 27 — which turned a
        360k-character corpus into 9639 chunks instead of roughly 720.
        """

        current: list[DocumentBlock] = []
        size = 0
        for block in blocks:
            if block.block_type not in self.indexable_types:
                continue
            if current and not self._extends(current[-1], block, size):
                yield current
                current, size = [], 0
            current.append(block)
            size += len(block.text) + (len(SEPARATOR) if len(current) > 1 else 0)
        if current:
            yield current

    def _extends(self, previous: DocumentBlock, block: DocumentBlock, size: int) -> bool:
        joined = size + len(SEPARATOR) + len(block.text)
        return joined <= self.chunker.chunk_size and self._same_context(previous, block)

    @staticmethod
    def _same_context(previous: DocumentBlock, block: DocumentBlock) -> bool:
        """Boundaries a segment must not cross.

        Merging across a heading or a page break would leave the chunk's section_path and page
        citation describing only part of its own content. Headings are not indexable, so they
        reach this check indirectly: every block below one carries it in section_path.
        """

        return (
            previous.document_id == block.document_id
            and previous.section_path == block.section_path
            and previous.page_number == block.page_number
        )

    def _merge(
        self, group: list[DocumentBlock]
    ) -> tuple[DocumentBlock, list[tuple[int, int]] | None]:
        """Build the block handed to the chunker, plus a merged-position to offset map.

        A single block is passed through untouched, so documents whose blocks are already
        chunk-sized keep byte-identical output. The map is None unless every member carries a
        real start_offset; without one there is no ground truth to restore, and the
        segment-relative offsets TextChunker computes are left as they are.
        """

        first = group[0]
        if len(group) == 1:
            return first, None

        spans: list[tuple[int, int]] = []
        position = 0
        for block in group:
            if block.start_offset is None:
                spans = []
                break
            spans.append((position, block.start_offset))
            position += len(block.text) + len(SEPARATOR)

        return (
            DocumentBlock(
                block_id=first.block_id,
                document_id=first.document_id,
                block_type=first.block_type,
                text=SEPARATOR.join(block.text for block in group),
                page_number=first.page_number,
                section_path=list(first.section_path),
                paragraph_index=first.paragraph_index,
                start_offset=first.start_offset,
                metadata=dict(first.metadata),
            ),
            spans or None,
        )

    @staticmethod
    def _restore_offsets(
        chunk: DocumentChunk,
        segment: DocumentBlock,
        spans: list[tuple[int, int]] | None,
    ) -> None:
        """Translate offsets back from merged-segment positions to document positions.

        TextChunker computes them as segment.start_offset + position-within-segment, which is
        only a document offset while the segment is one block. QuerySource publishes both
        numbers as citations, so a merged segment has to undo the join.
        """

        if spans is None or chunk.start_offset is None:
            return

        base = segment.start_offset or 0
        start = chunk.start_offset - base
        end = start + len(chunk.text)
        chunk.start_offset = _document_offset(spans, start)
        chunk.end_offset = _document_offset(spans, end)


def _document_offset(spans: list[tuple[int, int]], position: int) -> int:
    """Map a position in the merged text to the corresponding document offset."""

    for merged_start, document_start in reversed(spans):
        if position >= merged_start:
            return document_start + (position - merged_start)
    return position
