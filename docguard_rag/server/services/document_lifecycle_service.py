import json
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from docguard_rag.server.repositories.chunk_repository import ChunkRepository
from docguard_rag.server.repositories.document_repository import DocumentRepository
from docguard_rag.server.repositories.review_repository import ReviewRepository
from docguard_rag.server.schemas.document import DocumentDeleteResult, DocumentUploadResult
from docguard_rag.server.services.ingestion_service import IngestionError, IngestionService

logger = logging.getLogger(__name__)


class DocumentLifecycleService:
    """Delete, replace and rebuild documents while preserving tenant isolation."""

    def __init__(self, ingestion_service: IngestionService | None = None):
        self.ingestion = ingestion_service or IngestionService()
        self.documents = DocumentRepository()
        self.chunks = ChunkRepository()
        self.reviews = ReviewRepository()
        self.vectors = self.ingestion.index_service.vector_service

    def delete(
        self,
        db: Session,
        document_id: str,
        *,
        tenant_id: str = "default",
    ) -> DocumentDeleteResult:
        document = self._require_document(db, document_id, tenant_id)
        storage_path = document.storage_path
        try:
            chunk_ids = self._delete_database_records(db, document_id)
            self.vectors.delete_vectors(db, chunk_ids)
            db.commit()
        except Exception:
            db.rollback()
            raise
        self._delete_file(storage_path)
        return DocumentDeleteResult(document_id=document_id)

    def replace_bytes(
        self,
        db: Session,
        document_id: str,
        *,
        filename: str,
        content: bytes,
        collection_id: str | None = None,
        enable_review: bool = False,
        metadata: dict | None = None,
        tenant_id: str = "default",
        owner_id: str | None = None,
    ) -> DocumentUploadResult:
        document = self._require_document(db, document_id, tenant_id)
        old_path = document.storage_path
        old_chunk_ids = [item.chunk_id for item in self.chunks.get_chunks(db, document_id)]
        old_collection = document.collection_id
        old_owner = document.owner_id
        old_metadata = json.loads(document.metadata_json or "{}")
        db.expunge(document)

        self._delete_database_records(db, document_id)
        try:
            result = self.ingestion.ingest_bytes(
                db,
                filename=filename,
                content=content,
                collection_id=collection_id if collection_id is not None else old_collection,
                enable_review=enable_review,
                metadata=metadata if metadata is not None else old_metadata,
                tenant_id=tenant_id,
                owner_id=owner_id if owner_id is not None else old_owner,
                document_id=document_id,
            )
        except Exception:
            db.rollback()
            raise

        try:
            self.vectors.forget(old_chunk_ids)
        except Exception:
            logger.exception("Old vectors for replacement %s could not be removed", document_id)
        self._delete_file(old_path)
        return result

    def reindex(
        self,
        db: Session,
        document_id: str,
        *,
        tenant_id: str = "default",
        owner_id: str | None = None,
    ) -> DocumentUploadResult:
        document = self._require_document(db, document_id, tenant_id)
        if not document.storage_path or not Path(document.storage_path).is_file():
            raise IngestionError("Stored document file not found", status_code=404)
        return self.replace_bytes(
            db,
            document_id,
            filename=document.filename,
            content=Path(document.storage_path).read_bytes(),
            tenant_id=tenant_id,
            owner_id=owner_id,
        )

    def _delete_database_records(self, db: Session, document_id: str) -> list[str]:
        self.reviews.detach_document(db, document_id)
        chunk_ids = self.chunks.delete_by_document(db, document_id)
        self.ingestion.index_service.vector_service.repository.delete_by_chunk_ids(db, chunk_ids)
        self.documents.delete_blocks(db, document_id)
        self.documents.delete_document(db, document_id)
        return chunk_ids

    def _require_document(self, db: Session, document_id: str, tenant_id: str):
        document = self.documents.get_document(db, document_id, tenant_id)
        if document is None:
            raise IngestionError("Document not found", status_code=404)
        return document

    def _delete_file(self, storage_path: str | None) -> None:
        if not storage_path:
            return
        try:
            self.ingestion.storage.delete(storage_path)
        except (OSError, ValueError):
            logger.exception("Stored document file could not be removed: %s", storage_path)
