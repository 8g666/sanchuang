import numpy as np

from docguard_rag.server.vectorstores.base import VectorStore


class SimpleVectorStore(VectorStore):
    """Small in-memory vector store for local deployments and tests."""

    def __init__(self):
        self.vectors: dict[str, dict] = {}

    def add(self, chunk_id: str, vector: list[float], metadata: dict) -> None:
        self.vectors[chunk_id] = {"vector": vector, "metadata": metadata}

    def search(
        self,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict | None = None,
    ) -> list[dict]:
        query = np.asarray(query_vector, dtype=float)
        results = []
        for chunk_id, item in self.vectors.items():
            metadata = item["metadata"]
            if filters and not self._matches_filters(metadata, filters):
                continue
            vector = np.asarray(item["vector"], dtype=float)
            if query.shape != vector.shape:
                continue
            results.append(
                {
                    "chunk_id": chunk_id,
                    "score": self.cosine_similarity(query, vector),
                    "metadata": metadata,
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]

    def delete(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            self.vectors.pop(chunk_id, None)

    @staticmethod
    def _matches_filters(metadata: dict, filters: dict) -> bool:
        for key, expected in filters.items():
            if expected is None:
                continue
            actual = metadata.get(key)
            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator == 0:
            return 0.0
        return float(np.dot(a, b) / denominator)
