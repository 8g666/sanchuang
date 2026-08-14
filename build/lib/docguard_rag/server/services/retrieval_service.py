from docguard_rag.server.retrievers.hybrid_retriever import HybridRetriever


class RetrievalService:
    def __init__(self, retriever=None):

        self.retriever = retriever or HybridRetriever()

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
        filters: dict | None = None,
    ):
        combined_filters = dict(filters or {})
        if document_ids:
            combined_filters["document_id"] = document_ids
        return self.retriever.retrieve(
            query,
            top_k,
            combined_filters,
        )
