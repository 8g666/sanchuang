from docguard_rag.server.retrievers.base import Retriever
from docguard_rag.server.services.embedding_service import EmbeddingService
from docguard_rag.server.services.vector_service import VectorService


class VectorRetriever(Retriever):
    def __init__(self, embedding_service=None, vector_service=None):
        self.embedding_service = embedding_service or EmbeddingService()

        self.vector_service = vector_service or VectorService()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ):
        query_vector = self.embedding_service.embed_query(query)

        results = self.vector_service.search(
            query_vector,
            top_k,
            filters=filters,
        )

        return results
