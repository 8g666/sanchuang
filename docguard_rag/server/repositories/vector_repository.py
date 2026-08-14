import json

from sqlalchemy.orm import Session

from docguard_rag.server.database.models import DocumentVectorTable
from docguard_rag.server.models.embedding import EmbeddingRecord


class VectorRepository:
    def save_vector(
        self,
        db: Session,
        chunk_id: str,
        vector: list[float],
        metadata: dict,
    ) -> DocumentVectorTable:
        record = (
            db.query(DocumentVectorTable)
            .filter(DocumentVectorTable.chunk_id == chunk_id)
            .one_or_none()
        )
        if record is None:
            record = DocumentVectorTable(chunk_id=chunk_id)
            db.add(record)

        record.vector = json.dumps(vector)
        record.metadata_json = json.dumps(metadata, ensure_ascii=False)
        db.flush()

        return record

    def save_vectors(
        self,
        db: Session,
        embeddings: list[EmbeddingRecord],
    ) -> list[DocumentVectorTable]:
        records = [
            self.save_vector(
                db,
                embedding["chunk_id"],
                embedding["vector"],
                embedding.get("metadata", {}),
            )
            for embedding in embeddings
        ]
        db.flush()
        return records

    def get_all_vectors(self, db: Session) -> list[DocumentVectorTable]:
        return db.query(DocumentVectorTable).all()

    def delete_by_chunk_ids(self, db: Session, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        deleted = (
            db.query(DocumentVectorTable)
            .filter(DocumentVectorTable.chunk_id.in_(chunk_ids))
            .delete(synchronize_session=False)
        )
        db.flush()
        return deleted
