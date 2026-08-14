import re

from docguard_rag.server.rerankers.base import Reranker


def _terms(text: str) -> set[str]:
    normalized = "".join(text.lower().split())
    values = set(re.findall(r"[a-z0-9_./-]+", normalized))
    values.update(normalized)
    values.update(normalized[index : index + 2] for index in range(len(normalized) - 1))
    return values


class OverlapReranker(Reranker):
    """Deterministic offline reranker used when no model endpoint is configured."""

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        query_terms = _terms(query)
        rescored = []
        for candidate in candidates:
            text_terms = _terms(candidate.get("metadata", {}).get("text", ""))
            overlap = len(query_terms & text_terms) / max(1, len(query_terms))
            item = dict(candidate)
            item["score"] = 0.7 * float(candidate.get("score", 0.0)) + 0.3 * overlap
            rescored.append(item)
        rescored.sort(key=lambda item: item["score"], reverse=True)
        return rescored[:top_k]
