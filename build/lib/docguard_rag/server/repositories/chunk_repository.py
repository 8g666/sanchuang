from sqlalchemy.orm import Session

from docguard_rag.server.database.models import DocumentChunkTable
from docguard_rag.server.models.document_chunk import DocumentChunk


class ChunkRepository:
    def save_chunks(self, db: Session, chunks: list[DocumentChunk]) -> list[DocumentChunkTable]:
        records = [
            DocumentChunkTable(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                block_id=chunk.block_id,
                text=chunk.text,
                char_count=chunk.char_count,
                token_count=chunk.token_count,
                chunk_index=chunk.chunk_index,
                split_method=chunk.split_method,
            )
            for chunk in chunks
        ]
        db.add_all(records)
        db.flush()
        return records

    def get_chunks(self, db: Session, document_id: str) -> list[DocumentChunkTable]:
        return (
            db.query(DocumentChunkTable).filter(DocumentChunkTable.document_id == document_id).all()
        )

    def count_chunks(self, db: Session, document_id: str) -> int:
        return (
            db.query(DocumentChunkTable)
            .filter(DocumentChunkTable.document_id == document_id)
            .count()
        )

    def delete_by_document(self, db: Session, document_id: str) -> list[str]:
        chunk_ids = [item.chunk_id for item in self.get_chunks(db, document_id)]
        if chunk_ids:
            (
                db.query(DocumentChunkTable)
                .filter(DocumentChunkTable.document_id == document_id)
                .delete(synchronize_session=False)
            )
            db.flush()
        return chunk_ids
