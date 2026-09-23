"""Pre-normalised vector matrices, grouped by dimension, for the in-process store.

Separate from retrieval_index so the vector store does not pull in SQLAlchemy.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

# Rows that do not carry an encoded key get this code. It is never handed out to a real
# value, so "key missing" and "key equals X" can never be confused.
MISSING = -1

ENCODED_KEYS = ("tenant_id", "collection_id", "document_id")


@dataclass(frozen=True, eq=False)
class VectorMatrix:
    """One immutable (N, D) block of unit vectors plus the columns filters need.

    eq=False: the generated comparison would call into ndarray truthiness and raise.
    """

    dimension: int
    # Row order equals the source dict's insertion order. The stable argsort in the store
    # relies on it to break score ties exactly the way the per-row loop did.
    chunk_ids: tuple[str, ...]
    unit: np.ndarray
    metadata: tuple[dict, ...]
    tenant_codes: np.ndarray | None
    collection_codes: np.ndarray | None
    document_codes: np.ndarray | None
    codes: dict[str, dict]

    def column(self, key: str) -> np.ndarray | None:
        if key == "tenant_id":
            return self.tenant_codes
        if key == "collection_id":
            return self.collection_codes
        if key == "document_id":
            return self.document_codes
        return None


def build_matrices(vectors: Mapping[str, dict]) -> dict[int, VectorMatrix]:
    """Group vectors by width, one matrix per distinct dimension.

    Not a single "dominant dimension": a re-embed commits in batches, so 128- and
    2048-wide vectors coexist until it finishes. Picking one would silently exclude
    the minority for the whole window.
    """

    groups: dict[int, list[tuple[str, dict]]] = {}
    for chunk_id, item in vectors.items():
        groups.setdefault(len(item["vector"]), []).append((chunk_id, item))
    return {dimension: _matrix(rows) for dimension, rows in groups.items()}


def _matrix(rows: list[tuple[str, dict]]) -> VectorMatrix:
    dimension = len(rows[0][1]["vector"])
    raw = np.empty((len(rows), dimension), dtype=np.float64)
    for index, (_, item) in enumerate(rows):
        raw[index] = item["vector"]

    norms = np.linalg.norm(raw, axis=1)
    # Zero magnitude must yield a zero row, not NaN: cosine_similarity returns 0.0 when
    # either side is degenerate, and the scores have to agree.
    norms[norms == 0.0] = 1.0
    unit = raw / norms[:, None]

    metadata = tuple(item["metadata"] for _, item in rows)
    codes: dict[str, dict] = {}
    columns: dict[str, np.ndarray] = {}
    for key in ENCODED_KEYS:
        encoded = _encode(metadata, key)
        if encoded is not None:
            codes[key], columns[key] = encoded

    return VectorMatrix(
        dimension=dimension,
        chunk_ids=tuple(chunk_id for chunk_id, _ in rows),
        unit=unit,
        metadata=metadata,
        tenant_codes=columns.get("tenant_id"),
        collection_codes=columns.get("collection_id"),
        document_codes=columns.get("document_id"),
        codes=codes,
    )


def _encode(metadata: tuple[dict, ...], key: str) -> tuple[dict, np.ndarray] | None:
    """Integer-code one metadata key, or None when its values are not all hashable.

    Values come out of a JSON blob, so a list or dict is possible; those rows fall back
    to the per-row predicate instead of breaking every query at rebuild time.
    """

    table: dict = {}
    column = np.full(len(metadata), MISSING, dtype=np.int32)
    for index, entry in enumerate(metadata):
        value = entry.get(key)
        if value is None:
            continue
        try:
            code = table.get(value)
        except TypeError:
            return None
        if code is None:
            code = len(table)
            table[value] = code
        column[index] = code
    return table, column
