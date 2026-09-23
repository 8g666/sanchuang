from sqlalchemy.orm import Session

from docguard_rag.server.core.retrieval_index import snapshot_for
from docguard_rag.server.database.connection import SessionLocal
from docguard_rag.server.models.embedding import EmbeddingRecord
from docguard_rag.server.repositories.vector_repository import VectorRepository
from docguard_rag.server.vectorstores.factory import create_vector_store
from docguard_rag.server.vectorstores.simple_vector_store import SimpleVectorStore


class VectorService:
    """Persist vectors and search them through the configured backend.

    The in-process store is a read-through cache: SQL is its only source of truth, so
    writes land in the database and become visible after commit. Nothing mutates the
    shared snapshot directly, which is what keeps rolled-back writes out of results.
    """

    def __init__(self, session_factory=None, store=None):
        self.store = store or create_vector_store()
        self.repository = VectorRepository()
        self.session_factory = session_factory

    @property
    def _store_is_cache(self) -> bool:
        return isinstance(self.store, SimpleVectorStore)

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
            if self._store_is_cache:
                return
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
        if self._store_is_cache:
            # Rebind every search: the snapshot is swapped wholesale on corpus changes,
            # and bind moves the vectors and their matrices together.
            self.store.bind(snapshot_for(self.session_factory or SessionLocal))
        return self.store.search(query_vector, top_k=top_k, filters=filters)

    def delete_vectors(self, db: Session, chunk_ids: list[str]) -> None:
        self.repository.delete_by_chunk_ids(db, chunk_ids)
        self.forget(chunk_ids)

    def forget(self, chunk_ids: list[str]) -> None:
        """Drop vectors from backends that are their own source of truth."""

        if self._store_is_cache or not chunk_ids:
            return
        self.store.delete(chunk_ids)

    def close(self) -> None:
        close = getattr(self.store, "close", None)
        if close is not None:
            close()
