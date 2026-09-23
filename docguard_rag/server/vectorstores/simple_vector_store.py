import numpy as np

from docguard_rag.server.core.provider_errors import ProviderError
from docguard_rag.server.core.vector_matrix import MISSING, VectorMatrix, build_matrices
from docguard_rag.server.vectorstores.base import VectorStore


class SimpleVectorStore(VectorStore):
    """Small in-memory vector store for local deployments and tests.

    Search runs against a pre-normalised matrix per vector dimension rather than
    converting every stored Python list on every query, which at 2048 dimensions costs
    more than the arithmetic itself.
    """

    def __init__(self):
        self.vectors: dict[str, dict] = {}
        self._matrices: dict[int, VectorMatrix] | None = None

    def bind(self, snapshot) -> None:
        """Adopt a corpus snapshot wholesale.

        Both attributes move together so a concurrent reader can never see new vectors
        scored against stale matrices.
        """

        self._matrices = snapshot.matrices
        self.vectors = snapshot.vectors

    def add(self, chunk_id: str, vector: list[float], metadata: dict) -> None:
        self.vectors[chunk_id] = {"vector": vector, "metadata": metadata}
        self._matrices = None

    def delete(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            self.vectors.pop(chunk_id, None)
        self._matrices = None

    def search(
        self,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict | None = None,
    ) -> list[dict]:
        if top_k <= 0:
            return []
        matrices = self._ensure_matrices()
        query = np.asarray(query_vector, dtype=float)
        matrix = matrices.get(query.shape[0]) if query.ndim == 1 else None
        if matrix is None:
            if not matrices:
                return []
            raise ProviderError(
                f"Query vector of dimension {query.shape} does not match any indexed "
                f"vector dimension {sorted(matrices)}. After changing "
                "DOCGUARD_EMBEDDING_DIMENSION the corpus must be re-embedded."
            )

        scores = matrix.unit @ self._unit(query)
        return self._select(matrix, scores, self._mask(matrix, filters), top_k)

    def _ensure_matrices(self) -> dict[int, VectorMatrix]:
        # Deliberately unlocked: two threads may each build once and the later write
        # wins, but both results are identical and correct.
        matrices = self._matrices
        if matrices is None:
            matrices = build_matrices(self.vectors)
            self._matrices = matrices
        return matrices

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def _mask(self, matrix: VectorMatrix, filters: dict | None) -> np.ndarray | None:
        """Boolean row mask, or None when no filter narrows the corpus."""

        if not filters:
            return None
        keep = np.ones(matrix.unit.shape[0], dtype=bool)
        for key, expected in filters.items():
            if expected is None:
                continue
            values = expected if isinstance(expected, list | tuple | set) else [expected]
            table = matrix.codes.get(key)
            codes = self._codes(table, values) if table is not None else None
            if codes is None:
                keep &= np.fromiter(
                    (
                        self._matches_filters(metadata, {key: expected})
                        for metadata in matrix.metadata
                    ),
                    dtype=bool,
                    count=len(matrix.metadata),
                )
            elif codes:
                keep &= np.isin(matrix.column(key), codes)
            else:
                keep[:] = False
            if not keep.any():
                return keep
        return keep

    @staticmethod
    def _codes(table: dict, values: list) -> list[int] | None:
        """Translate expected values into row codes, or None if the column cannot answer.

        A value nobody carries contributes no code, so an unknown value selects nothing.
        Encoding it as MISSING instead would select every row that lacks the key.
        """

        codes: list[int] = []
        for value in values:
            if value is None:
                # metadata.get returns None for an absent key too, which is exactly what
                # the per-row predicate matched on.
                codes.append(MISSING)
                continue
            try:
                code = table.get(value)
            except TypeError:
                return None
            if code is not None:
                codes.append(code)
        return codes

    @staticmethod
    def _select(
        matrix: VectorMatrix,
        scores: np.ndarray,
        keep: np.ndarray | None,
        top_k: int,
    ) -> list[dict]:
        rows = None if keep is None else np.flatnonzero(keep)
        subset = scores if rows is None else scores[rows]
        if subset.size == 0:
            return []
        # stable, not argsort(subset)[::-1]: reversing flips the order of tied scores,
        # while the per-row loop this replaced kept them in insertion order.
        order = np.argsort(-subset, kind="stable")[:top_k]
        picked = order if rows is None else rows[order]
        return [
            {
                "chunk_id": matrix.chunk_ids[index],
                "score": float(scores[index]),
                "metadata": matrix.metadata[index],
            }
            for index in picked
        ]

    @staticmethod
    def _matches_filters(metadata: dict, filters: dict) -> bool:
        for key, expected in filters.items():
            if expected is None:
                continue
            actual = metadata.get(key)
            if isinstance(expected, list | tuple | set):
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
