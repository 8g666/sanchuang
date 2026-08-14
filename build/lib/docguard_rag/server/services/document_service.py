from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from docguard_rag.server.models.document import DocumentInfo


class DocumentService:
    """Create document domain objects without retaining duplicate in-memory state."""

    def create_document(
        self,
        filename: str,
        file_hash: str,
        file_size: int,
        tenant_id: str = "default",
        owner_id: str | None = None,
        collection_id: str | None = None,
        storage_path: str | None = None,
        metadata: dict | None = None,
        document_id: str | None = None,
    ) -> DocumentInfo:
        return DocumentInfo(
            document_id=document_id or f"doc_{uuid4().hex[:12]}",
            filename=filename,
            file_type=Path(filename).suffix.lower().lstrip("."),
            file_size=file_size,
            file_hash=file_hash,
            status="PROCESSING",
            created_at=datetime.now(UTC),
            tenant_id=tenant_id,
            owner_id=owner_id,
            collection_id=collection_id,
            storage_path=storage_path,
            metadata=metadata or {},
        )
