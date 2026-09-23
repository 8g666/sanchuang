"""Re-embed committed chunks in place, without re-parsing the source files.

Switching embedding models (or fixing a wrong DOCGUARD_EMBEDDING_DIMENSION) invalidates
every stored vector, but the chunks and blocks in SQL are still good. This rebuilds the
vectors from them, in batches, so an interruption leaves counted progress rather than a
half-written transaction nobody can describe.
"""

import json
import logging
from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from docguard_rag.server.core.config import get_settings
from docguard_rag.server.database.models import (
    DocumentBlockTable,
    DocumentChunkTable,
    DocumentTable,
    DocumentVectorTable,
)
from docguard_rag.server.models.document_chunk import DocumentChunk
from docguard_rag.server.schemas.document import ReembedResult
from docguard_rag.server.services.embedding_service import EmbeddingService
from docguard_rag.server.services.vector_service import VectorService

logger = logging.getLogger(__name__)


class ReembedService:
    def __init__(self, embedding_service=None, vector_service=None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_service = vector_service or VectorService()

    def reembed(
        self,
        db: Session,
        *,
        tenant_id: str,
        document_ids: list[str] | None = None,
        batch_size: int | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> ReembedResult:
        settings = get_settings()
        size = max(1, batch_size or settings.embedding_batch_size)
        rows = self._candidates(db, tenant_id, document_ids)
        result = ReembedResult(dimension=settings.embedding_dimension)
        if not rows:
            return result

        needing_reindex: dict[str, None] = {}
        for start in range(0, len(rows), size):
            batch = rows[start : start + size]
            chunks = []
            for row in batch:
                existing = json.loads(row.vector_metadata) if row.vector_metadata else None
                if existing is None:
                    # Offsets are derived from a chunk's position inside its block and are
                    # never persisted, so with no previous vector row they cannot be
                    # recovered from SQL. The vector is still usable for retrieval; only
                    # citation offsets degrade.
                    result.embedded_without_offsets += 1
                    needing_reindex[row.document_id] = None
                chunks.append(self._chunk(row, existing))
            embeddings = self.embedding_service.embed_chunks(chunks)
            self.vector_service.add_vectors(embeddings, db=db)
            # Per batch, not once at the end: a 2048-wide vector serialises to ~40 KB of
            # JSON, so one transaction over a large corpus would balloon the SQLite
            # journal, and committing makes partial progress durable and reportable.
            db.commit()
            result.batches += 1
            result.reembedded += len(batch)
            if progress is not None:
                progress(min(start + size, len(rows)), len(rows))

        result.document_ids_needing_reindex = sorted(needing_reindex)
        if needing_reindex:
            logger.warning(
                "Re-embedded %d chunk(s) across %d document(s) without stored offsets; "
                "POST /v1/documents/{document_id}/reindex reproduces them exactly.",
                result.embedded_without_offsets,
                len(needing_reindex),
            )
        return result

    @staticmethod
    def _candidates(db: Session, tenant_id: str, document_ids: list[str] | None):
        statement = (
            select(
                DocumentChunkTable.chunk_id,
                DocumentChunkTable.document_id,
                DocumentChunkTable.block_id,
                DocumentChunkTable.text,
                DocumentChunkTable.char_count,
                DocumentChunkTable.token_count,
                DocumentChunkTable.chunk_index,
                DocumentChunkTable.split_method,
                DocumentTable.tenant_id,
                DocumentTable.owner_id,
                DocumentTable.collection_id,
                DocumentTable.metadata_json.label("document_metadata"),
                DocumentBlockTable.page_number,
                DocumentBlockTable.section_path,
                DocumentBlockTable.paragraph_index,
                DocumentVectorTable.metadata_json.label("vector_metadata"),
            )
            .join(DocumentTable, DocumentTable.document_id == DocumentChunkTable.document_id)
            # Outer joins: an inner one would silently drop a chunk whose block or vector
            # row is missing, and the caller would be told it had been re-embedded.
            .outerjoin(
                DocumentBlockTable, DocumentBlockTable.block_id == DocumentChunkTable.block_id
            )
            .outerjoin(
                DocumentVectorTable, DocumentVectorTable.chunk_id == DocumentChunkTable.chunk_id
            )
            .where(DocumentTable.tenant_id == tenant_id)
            .order_by(DocumentChunkTable.id)
        )
        if document_ids:
            statement = statement.where(DocumentChunkTable.document_id.in_(document_ids))
        return db.execute(statement).all()

    @staticmethod
    def _chunk(row, existing: dict | None) -> DocumentChunk:
        """Rebuild a chunk so embed_chunks reproduces the metadata already stored.

        The chunk_id is reused verbatim: generating a fresh one would orphan the previous
        vector row and leave the corpus with duplicates.
        """

        if existing is None:
            metadata = {
                "tenant_id": row.tenant_id,
                "owner_id": row.owner_id,
                "collection_id": row.collection_id,
                "paragraph_index": row.paragraph_index,
                "custom_metadata": json.loads(row.document_metadata or "{}"),
            }
            page_start = row.page_number
            section_path = json.loads(row.section_path or "[]")
            start_offset = end_offset = None
        else:
            metadata = {
                "tenant_id": existing.get("tenant_id", row.tenant_id),
                "owner_id": existing.get("owner_id"),
                "collection_id": existing.get("collection_id"),
                "paragraph_index": existing.get("paragraph_index"),
                "custom_metadata": existing.get("custom_metadata", {}),
            }
            page_start = existing.get("page")
            section_path = existing.get("section_path") or []
            start_offset = existing.get("start_offset")
            end_offset = existing.get("end_offset")

        return DocumentChunk(
            chunk_id=row.chunk_id,
            document_id=row.document_id,
            block_id=row.block_id,
            text=row.text,
            char_count=row.char_count,
            token_count=row.token_count,
            chunk_index=row.chunk_index,
            split_method=row.split_method,
            start_offset=start_offset,
            end_offset=end_offset,
            page_start=page_start,
            section_path=section_path,
            metadata=metadata,
        )
