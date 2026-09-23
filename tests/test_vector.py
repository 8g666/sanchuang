import math
from pathlib import Path

import numpy as np
import pytest

from docguard_rag import DocGuard
from docguard_rag.server.core.provider_errors import ProviderError
from docguard_rag.server.core.retrieval_index import index_for_engine
from docguard_rag.server.services.embedding_service import EmbeddingService
from docguard_rag.server.vectorstores.simple_vector_store import SimpleVectorStore


def test_vector_search():
    store = SimpleVectorStore()

    store.add("chunk1", [1, 0, 0], {})

    store.add("chunk2", [0, 1, 0], {})

    result = store.search([0.9, 0.1, 0], top_k=1)

    assert result[0]["chunk_id"] == "chunk1"


def _reference_search(vectors, query_vector, top_k=3, filters=None):
    """Verbatim copy of the per-row loop that matrix search replaced.

    Kept as the oracle: the two implementations must agree on ordering exactly and on
    scores to within float noise, which is what makes the rewrite checkable at all.
    """

    query = np.asarray(query_vector, dtype=float)
    results = []
    for chunk_id, item in vectors.items():
        metadata = item["metadata"]
        if filters and not SimpleVectorStore._matches_filters(metadata, filters):
            continue
        vector = np.asarray(item["vector"], dtype=float)
        if query.shape != vector.shape:
            continue
        results.append(
            {
                "chunk_id": chunk_id,
                "score": SimpleVectorStore.cosine_similarity(query, vector),
                "metadata": metadata,
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _store(vectors: dict) -> SimpleVectorStore:
    store = SimpleVectorStore()
    store.vectors = vectors
    return store


def _rows(count: int, dimension: int, *, seed: int, start: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        f"chunk{start + index}": {
            "vector": rng.random(dimension).tolist(),
            "metadata": {
                "tenant_id": "default",
                "collection_id": f"c{index % 3}",
                "document_id": f"d{index % 7}",
                "text": f"文本 {index}",
            },
        }
        for index in range(count)
    }


def _sparse(count: int, dimension: int, *, seed: int) -> dict:
    """Some rows omit collection_id/document_id, so the "missing" code is exercised."""

    vectors = _rows(count, dimension, seed=seed)
    for index, (chunk_id, item) in enumerate(vectors.items()):
        if index % 4 == 0:
            item["metadata"].pop("document_id", None)
        if index % 5 == 0:
            item["metadata"]["collection_id"] = None
        if index % 6 == 0:
            item["metadata"] = {"text": item["metadata"]["text"]}
    return vectors


def _zero_rows(count: int, dimension: int, *, start: int) -> dict:
    return {
        f"zero{start + index}": {"vector": [0.0] * dimension, "metadata": {"tenant_id": "default"}}
        for index in range(count)
    }


CORPORA = {
    "single": lambda: _rows(1, 8, seed=11),
    "wide": lambda: _rows(500, 2048, seed=12),
    "zero-vector": lambda: {**_rows(40, 16, seed=13), **_zero_rows(3, 16, start=0)},
    "mixed-dimension": lambda: {
        **_rows(30, 128, seed=14),
        **_rows(30, 2048, seed=15, start=1000),
    },
    "sparse-metadata": lambda: _sparse(24, 12, seed=16),
}

# (corpus, query dimension, query seed)
CASES = [
    ("single", 8, 21),
    ("wide", 2048, 22),
    ("zero-vector", 16, 23),
    ("mixed-dimension", 128, 24),
    ("mixed-dimension", 2048, 25),
    ("sparse-metadata", 12, 26),
]

FILTERS = [
    None,
    {},
    {"tenant_id": "default"},
    {"tenant_id": "absent"},
    {"collection_id": None},
    {"collection_id": "c1"},
    {"collection_id": "absent"},
    {"document_id": ["d1", "d3"]},
    {"document_id": ["absent"]},
    {"document_id": [None, "d1"]},
    {"text": "文本 3"},
    {"unknown": "value"},
    {"tenant_id": "default", "collection_id": "c1", "document_id": ["d1", "d2"]},
    {"collection_id": "c1", "text": "文本 5"},
]


@pytest.mark.parametrize("top_k", [1, 5, 1000])
@pytest.mark.parametrize("filters", FILTERS)
@pytest.mark.parametrize("name,dimension,query_seed", CASES)
def test_matrix_search_matches_the_row_loop(name, dimension, query_seed, filters, top_k):
    vectors = CORPORA[name]()
    rng = np.random.default_rng(query_seed)
    query = rng.random(dimension).tolist()

    expected = _reference_search(vectors, query, top_k=top_k, filters=filters)
    actual = _store(vectors).search(query, top_k=top_k, filters=filters)

    assert [item["chunk_id"] for item in actual] == [item["chunk_id"] for item in expected]
    assert [item["metadata"] for item in actual] == [item["metadata"] for item in expected]
    # Scores cannot be bit-identical: the matrix path pre-normalises and uses a BLAS
    # dgemv instead of a per-row ddot. Ordering can be, and that is what matters.
    assert all(
        math.isclose(left["score"], right["score"], rel_tol=1e-9, abs_tol=1e-12)
        for left, right in zip(actual, expected, strict=True)
    )


def test_exact_ties_keep_insertion_order():
    vectors = {
        "b": {"vector": [1.0, 0.0], "metadata": {}},
        "a": {"vector": [1.0, 0.0], "metadata": {}},
        "c": {"vector": [0.0, 1.0], "metadata": {}},
    }
    query = [1.0, 0.0]

    actual = _store(vectors).search(query, top_k=3)

    assert [item["chunk_id"] for item in actual] == ["b", "a", "c"]
    assert [item["chunk_id"] for item in actual] == [
        item["chunk_id"] for item in _reference_search(vectors, query, top_k=3)
    ]


def test_empty_corpus_returns_nothing_instead_of_a_dimension_error():
    assert SimpleVectorStore().search([0.1] * 8, top_k=3) == []
    assert _reference_search({}, [0.1] * 8, top_k=3) == []


def test_non_positive_top_k_returns_nothing():
    store = _store(_rows(4, 8, seed=31))

    assert store.search([0.1] * 8, top_k=0) == []


def test_unmatched_query_dimension_reports_reembedding():
    store = _store(_rows(4, 128, seed=32))

    with pytest.raises(ProviderError, match="does not match any indexed vector dimension"):
        store.search([0.1] * 64, top_k=3)


def test_a_mixed_corpus_answers_both_dimensions():
    """A batched re-embed commits 128- and 2048-wide vectors side by side."""

    vectors = CORPORA["mixed-dimension"]()
    store = _store(vectors)

    narrow = store.search([0.5] * 128, top_k=100)
    wide = store.search([0.5] * 2048, top_k=100)

    assert len(narrow) == 30
    assert len(wide) == 30
    assert {item["chunk_id"] for item in narrow}.isdisjoint({item["chunk_id"] for item in wide})


def test_local_writes_invalidate_the_matrices():
    store = _store(_rows(4, 8, seed=33))
    assert store.search([0.1] * 8, top_k=10)

    store.add("extra", [0.1] * 8, {"tenant_id": "default"})
    after_add = store.search([0.1] * 8, top_k=10)
    expected_after_add = _reference_search(store.vectors, [0.1] * 8, top_k=10)

    store.delete(["extra"])
    after_delete = store.search([0.1] * 8, top_k=10)

    assert [item["chunk_id"] for item in after_add] == [
        item["chunk_id"] for item in expected_after_add
    ]
    assert "extra" in [item["chunk_id"] for item in after_add]
    assert "extra" not in [item["chunk_id"] for item in after_delete]


def test_searched_metadata_objects_are_the_snapshots_own(tmp_path):
    """The matrix reuses the snapshot dicts rather than copying them."""

    with _guard(tmp_path) as guard:
        guard.documents.upload(_write(tmp_path, "kepu.md", "光合作用发生在叶绿体中，需要光能。"))

        results = guard._vector_service.search(EmbeddingService().embed_query("光合作用"), top_k=5)
        with guard._session_factory() as db:
            snapshot = index_for_engine(guard._engine).ensure_fresh(db)

        assert results
        assert all(
            item["metadata"] is snapshot.vectors[item["chunk_id"]]["metadata"] for item in results
        )


def test_end_to_end_vector_path_matches_the_row_loop(tmp_path):
    with _guard(tmp_path) as guard:
        guard.documents.upload(_write(tmp_path, "kepu.md", "光合作用发生在叶绿体中，需要光能。"))
        guard.documents.upload(_write(tmp_path, "cell.md", "细胞呼吸发生在线粒体中，产生能量。"))
        query_vector = EmbeddingService().embed_query("光合作用")

        actual = guard._vector_service.search(query_vector, top_k=5)
        with guard._session_factory() as db:
            snapshot = index_for_engine(guard._engine).ensure_fresh(db)

        # The full query path reranks and may rewrite, so only the vector leg is
        # comparable against the oracle; the rest is asserted to still answer.
        answered = guard.query("光合作用", generate_answer=False)

    assert [item["chunk_id"] for item in actual] == [
        item["chunk_id"] for item in _reference_search(snapshot.vectors, query_vector, top_k=5)
    ]
    assert answered.sources


def _write(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _guard(tmp_path: Path) -> DocGuard:
    return DocGuard(
        database_url=f"sqlite:///{tmp_path / 'matrix.db'}",
        upload_dir=tmp_path / "uploads",
    )
