import json

from sqlalchemy.orm import Session

from docguard_rag.server.database.models import DocumentBlockTable, DocumentTable
from docguard_rag.server.models.document import DocumentInfo
from docguard_rag.server.models.document_block import DocumentBlock


class DocumentRepository:
    def save_document(self, db: Session, document: DocumentInfo) -> DocumentTable:
        record = DocumentTable(
            document_id=document.document_id,
            tenant_id=document.tenant_id,
            owner_id=document.owner_id,
            collection_id=document.collection_id,
            filename=document.filename,
            file_type=document.file_type,
            file_hash=document.file_hash,
            status=document.status,
            file_size=document.file_size,
            storage_path=document.storage_path,
            metadata_json=json.dumps(document.metadata, ensure_ascii=False),
            created_at=document.created_at,
        )

        db.add(record)
        db.flush()
        db.refresh(record)

        return record

    def save_blocks(self, db: Session, blocks: list[DocumentBlock]) -> list[DocumentBlockTable]:
        records = [
            DocumentBlockTable(
                block_id=block.block_id,
                document_id=block.document_id,
                block_type=block.block_type,
                text=block.text,
                page_number=block.page_number,
                paragraph_index=block.paragraph_index,
                sentence_index=block.sentence_index,
                start_offset=block.start_offset,
                end_offset=block.end_offset,
                section_path=json.dumps(block.section_path, ensure_ascii=False),
                metadata_json=json.dumps(block.metadata, ensure_ascii=False),
            )
            for block in blocks
        ]
        db.add_all(records)
        db.flush()
        return records

    def get_blocks(self, db: Session, document_id: str) -> list[DocumentBlockTable]:
        return (
            db.query(DocumentBlockTable).filter(DocumentBlockTable.document_id == document_id).all()
        )

    def get_document(
        self,
        db: Session,
        document_id: str,
        tenant_id: str | None = None,
    ) -> DocumentTable | None:
        query = db.query(DocumentTable).filter(DocumentTable.document_id == document_id)
        if tenant_id is not None:
            query = query.filter(DocumentTable.tenant_id == tenant_id)
        return query.one_or_none()

    def get_documents(
        self,
        db: Session,
        document_ids: list[str],
        tenant_id: str | None = None,
    ) -> list[DocumentTable]:
        if not document_ids:
            return []
        query = db.query(DocumentTable).filter(DocumentTable.document_id.in_(document_ids))
        if tenant_id is not None:
            query = query.filter(DocumentTable.tenant_id == tenant_id)
        return query.all()

    def update_status(self, db: Session, document_id: str, status: str) -> DocumentTable | None:
        document = self.get_document(db, document_id)
        if document is None:
            return None
        document.status = status
        db.commit()
        db.refresh(document)
        return document

    def delete_blocks(self, db: Session, document_id: str) -> int:
        deleted = (
            db.query(DocumentBlockTable)
            .filter(DocumentBlockTable.document_id == document_id)
            .delete(synchronize_session=False)
        )
        db.flush()
        return deleted

    def delete_document(self, db: Session, document_id: str) -> int:
        deleted = (
            db.query(DocumentTable)
            .filter(DocumentTable.document_id == document_id)
            .delete(synchronize_session=False)
        )
        db.flush()
        return deleted
