from pathlib import Path

from sqlalchemy.orm import Session

from docguard_rag.server.core.config import get_settings
from docguard_rag.server.repositories.chunk_repository import ChunkRepository
from docguard_rag.server.repositories.document_repository import DocumentRepository
from docguard_rag.server.schemas.document import DocumentUploadResult
from docguard_rag.server.services.chunk_service import ChunkService
from docguard_rag.server.services.document_service import DocumentService
from docguard_rag.server.services.index_service import IndexService
from docguard_rag.server.services.parser_service import ParserService
from docguard_rag.server.services.review_service import ReviewService
from docguard_rag.server.storage.file_storage import FileStorage


class IngestionError(ValueError):
    """A safe, caller-facing document ingestion error."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


class IngestionService:
    """Run the complete parse, chunk, persist and index pipeline."""

    def __init__(
        self,
        *,
        storage: FileStorage | None = None,
        parser_service: ParserService | None = None,
        chunk_service: ChunkService | None = None,
        index_service: IndexService | None = None,
        review_service: ReviewService | None = None,
    ):
        self.storage = storage or FileStorage()
        self.parser_service = parser_service or ParserService()
        self.chunk_service = chunk_service or ChunkService()
        self.index_service = index_service or IndexService()
        self.review_service = review_service or ReviewService()
        self.document_service = DocumentService()
        self.document_repository = DocumentRepository()
        self.chunk_repository = ChunkRepository()

    def ingest_bytes(
        self,
        db: Session,
        *,
        filename: str,
        content: bytes,
        collection_id: str | None = None,
        enable_review: bool = False,
        metadata: dict | None = None,
        tenant_id: str = "default",
        owner_id: str | None = None,
        document_id: str | None = None,
    ) -> DocumentUploadResult:
        settings = get_settings()
        normalized_filename = filename or "document"
        extension = Path(normalized_filename).suffix.lower().lstrip(".")
        if extension not in settings.allowed_extensions:
            raise IngestionError(
                f"Unsupported document type: {extension or 'unknown'}",
                status_code=415,
            )
        if not content:
            raise IngestionError("Uploaded document is empty", status_code=400)
        if len(content) > settings.max_upload_size:
            raise IngestionError("Uploaded document is too large", status_code=413)
        if extension == "pdf" and not content.startswith(b"%PDF"):
            raise IngestionError("File content is not a valid PDF", status_code=400)
        if extension == "docx" and not content.startswith(b"PK"):
            raise IngestionError("File content is not a valid DOCX", status_code=400)

        saved_path = self.storage.save_bytes(normalized_filename, content)
        document = self.document_service.create_document(
            filename=normalized_filename,
            file_hash=self.storage.calculate_hash(saved_path),
            file_size=len(content),
            tenant_id=tenant_id,
            owner_id=owner_id,
            collection_id=collection_id,
            storage_path=saved_path,
            metadata=metadata,
            document_id=document_id,
        )
        try:
            blocks = self.parser_service.parse(
                saved_path,
                document.file_type,
                document.document_id,
            )
        except (ValueError, RuntimeError) as exc:
            self.storage.delete(saved_path)
            raise IngestionError(str(exc)) from exc
        if not blocks:
            self.storage.delete(saved_path)
            raise IngestionError("No readable text was found in the document")

        shared_metadata = {
            "tenant_id": tenant_id,
            "owner_id": owner_id,
            "collection_id": collection_id,
            "custom_metadata": metadata or {},
        }
        for block in blocks:
            block.metadata = {**block.metadata, **shared_metadata}

        try:
            document.status = "READY"
            self.document_repository.save_document(db, document)
            self.document_repository.save_blocks(db, blocks)
            chunks = self.chunk_service.create_chunks(blocks)
            self.chunk_repository.save_chunks(db, chunks)
            embeddings = self.index_service.build_index(db, chunks)
            db.commit()
        except Exception:
            db.rollback()
            self.storage.delete(saved_path)
            raise

        review_id = None
        if enable_review:
            review_id, _ = self.review_service.create_review(
                db,
                [document.document_id],
                ["TYPO", "SEMANTIC_UNCLEAR"],
                "BALANCED",
                tenant_id=tenant_id,
            )

        return DocumentUploadResult(
            document_id=document.document_id,
            job_id=review_id,
            review_id=review_id,
            status=document.status,
            blocks_count=len(blocks),
            chunks_count=len(chunks),
            vectors_count=len(embeddings),
        )

    def ingest_file(
        self,
        db: Session,
        file_path: str | Path,
        **kwargs,
    ) -> DocumentUploadResult:
        path = Path(file_path)
        if not path.is_file():
            raise IngestionError(f"Document file not found: {path}", status_code=404)
        return self.ingest_bytes(
            db,
            filename=path.name,
            content=path.read_bytes(),
            **kwargs,
        )
