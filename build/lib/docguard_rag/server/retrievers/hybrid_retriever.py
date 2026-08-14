from docguard_rag.server.core.provider_factory import create_reranker
from docguard_rag.server.retrievers.base import Retriever
from docguard_rag.server.retrievers.lexical_retriever import LexicalRetriever, tokenize
from docguard_rag.server.retrievers.vector_retriever import VectorRetriever


class HybridRetriever(Retriever):
    """Fuse lexical and dense rankings with reciprocal rank fusion."""

    def __init__(self, lexical=None, dense=None, reranker=None, rrf_k: int = 60):
        self.lexical = lexical or LexicalRetriever()
        self.dense = dense or VectorRetriever()
        self.reranker = reranker if reranker is not None else create_reranker()
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        if not query.strip() or top_k <= 0:
            return []

        candidate_count = max(top_k * 4, 20)
        rankings = [
            self.lexical.retrieve(query, candidate_count, filters),
            self.dense.retrieve(query, candidate_count, filters),
        ]
        fused: dict[str, dict] = {}
        for ranking in rankings:
            for rank, item in enumerate(ranking, start=1):
                candidate = fused.setdefault(
                    item["chunk_id"],
                    {
                        "chunk_id": item["chunk_id"],
                        "score": 0.0,
                        "metadata": item.get("metadata", {}),
                    },
                )
                candidate["score"] += 1.0 / (self.rrf_k + rank)

        query_terms = set(tokenize(query))
        for candidate in fused.values():
            text_terms = set(tokenize(candidate["metadata"].get("text", "")))
            overlap = len(query_terms & text_terms) / max(1, len(query_terms))
            candidate["score"] += 0.01 * overlap

        results = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
        if results:
            best_score = results[0]["score"] or 1.0
            for result in results:
                result["score"] = round(result["score"] / best_score, 6)
        if self.reranker:
            return self.reranker.rerank(query, results, top_k)
        return results[:top_k]
