import json

from sqlalchemy.orm import Session

from docguard_rag.server.database.connection import SessionLocal
from docguard_rag.server.models.embedding import EmbeddingRecord
from docguard_rag.server.repositories.vector_repository import VectorRepository
from docguard_rag.server.vectorstores.factory import create_vector_store
from docguard_rag.server.vectorstores.simple_vector_store import SimpleVectorStore


class VectorService:
    def __init__(self, session_factory=None, store=None):
        self.store = store or create_vector_store()
        self.repository = VectorRepository()
        self.session_factory = session_factory

    def load_vectors(self, db: Session | None = None) -> None:
        owns_session = db is None
        db = db or (self.session_factory or SessionLocal)()
        try:
            records = self.repository.get_all_vectors(db)
            for record in records:
                metadata = json.loads(record.metadata_json or "{}")
                metadata.setdefault("tenant_id", "default")
                self.store.add(
                    chunk_id=record.chunk_id, vector=json.loads(record.vector), metadata=metadata
                )
        finally:
            if owns_session:
                db.close()

    def add_vectors(
        self,
        embeddings: list[EmbeddingRecord],
        db: Session | None = None,
    ) -> None:
        owns_session = db is None
        db = db or (self.session_factory or SessionLocal)()
        try:
            self.repository.save_vectors(db, embeddings)
            if owns_session:
                db.commit()
            add_many = getattr(self.store, "add_many", None)
            if add_many is not None:
                add_many(embeddings)
            else:
                for embedding in embeddings:
                    self.store.add(
                        chunk_id=embedding["chunk_id"],
                        vector=embedding["vector"],
                        metadata=embedding.get("metadata", {}),
                    )
        finally:
            if owns_session:
                db.close()

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        if isinstance(self.store, SimpleVectorStore):
            # Rebuild from committed storage to avoid returning stale rolled-back vectors.
            self.store = SimpleVectorStore()
            self.load_vectors()
        return self.store.search(query_vector, top_k=top_k, filters=filters)

    def delete_vectors(self, db: Session, chunk_ids: list[str]) -> None:
        self.repository.delete_by_chunk_ids(db, chunk_ids)
        self.store.delete(chunk_ids)

    def close(self) -> None:
        close = getattr(self.store, "close", None)
        if close is not None:
            close()
