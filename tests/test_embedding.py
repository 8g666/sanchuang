import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from docguard_rag.server.core.embedding_cache import QueryEmbeddingCache, query_embedding_cache
from docguard_rag.server.core.provider_errors import ProviderError
from docguard_rag.server.embeddings.fake_embedding import FakeEmbedding
from docguard_rag.server.models.document_chunk import DocumentChunk
from docguard_rag.server.services.embedding_service import EmbeddingService


@pytest.fixture(autouse=True)
def _isolated_query_cache():
    """The cache is process-global by design, so every case starts and ends empty."""

    query_embedding_cache().clear()
    yield
    query_embedding_cache().clear()


def test_fake_embedding():
    model = FakeEmbedding(dimension=8)

    vectors = model.embed_texts(["RAG是一种技术"])

    assert len(vectors) == 1

    assert len(vectors[0]) == 8


class _FixedWidthProvider:
    """Stand-in for a remote model whose output width is set by the deployment."""

    def __init__(self, width: int):
        self.width = width

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.width for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [0.0] * self.width


def _chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        block_id="block-1",
        text="平台支持项目申报。",
        char_count=9,
        chunk_index=0,
        split_method="text",
    )


def test_embedding_service_accepts_the_configured_dimension():
    service = EmbeddingService(provider=_FixedWidthProvider(128))

    assert len(service.embed_query("问题")) == 128
    assert len(service.embed_chunks([_chunk()])[0]["vector"]) == 128


def test_embedding_service_rejects_a_dimension_mismatch():
    """A wrong DOCGUARD_EMBEDDING_DIMENSION must fail loudly at the first read or
    write instead of silently poisoning the corpus with unsearchable vectors."""

    service = EmbeddingService(provider=_FixedWidthProvider(2048))

    with pytest.raises(ProviderError, match="expected 128, provider returned 2048"):
        service.embed_query("问题")
    with pytest.raises(ProviderError, match="Embedding dimension mismatch"):
        service.embed_chunks([_chunk()])


class _CountingProvider:
    """Deterministic per-text vectors that record every query they were asked for."""

    def __init__(self, *, width: int = 128, name: str = "model-a", fail_times: int = 0):
        self.width = width
        self.name = name
        self.fail_times = fail_times
        self.queries: list[str] = []

    @property
    def fingerprint(self) -> tuple:
        return ("test", self.name, self.width)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ProviderError("embedding provider unavailable")
        return self._vector(query)

    def _vector(self, text: str) -> list[float]:
        seed = sum(f"{self.name}:{text}".encode()) % 2**32
        return np.random.default_rng(seed).random(self.width).tolist()


def test_a_repeated_query_hits_the_provider_once():
    provider = _CountingProvider()
    service = EmbeddingService(provider=provider)

    first = service.embed_query("怎么注册账号")
    second = service.embed_query("怎么注册账号")

    assert first == second
    assert provider.queries == ["怎么注册账号"]
    assert (query_embedding_cache().hits, query_embedding_cache().misses) == (1, 1)


def test_a_new_query_text_is_a_miss():
    provider = _CountingProvider()
    service = EmbeddingService(provider=provider)

    service.embed_query("怎么注册账号")
    service.embed_query("如何开通登录凭证")

    assert provider.queries == ["怎么注册账号", "如何开通登录凭证"]


def test_lru_eviction_refreshes_recency_on_a_hit():
    cache = QueryEmbeddingCache(maxsize=2)
    computed: list[str] = []

    def store(key: str) -> list[float]:
        def build() -> list[float]:
            computed.append(key)
            return [float(len(key))]

        return cache.get_or_compute(("test", key), build)

    store("a")
    store("b")
    store("a")  # hit, so "b" becomes the least recently used
    store("c")  # evicts "b"

    assert computed == ["a", "b", "c"]
    store("a")
    assert computed == ["a", "b", "c"]
    store("b")
    assert computed == ["a", "b", "c", "b"]


def test_two_models_do_not_share_entries_for_the_same_text():
    alpha = _CountingProvider(name="model-a")
    beta = _CountingProvider(name="model-b")

    first = EmbeddingService(provider=alpha).embed_query("平台怎么申报项目")
    second = EmbeddingService(provider=beta).embed_query("平台怎么申报项目")

    assert alpha.queries == ["平台怎么申报项目"]
    assert beta.queries == ["平台怎么申报项目"]
    assert first != second
    assert len(query_embedding_cache()) == 2


def test_the_cache_is_shared_by_every_embedding_service_in_the_process():
    """Two services over one provider are two retrieval stacks, not two caches."""

    provider = _CountingProvider()

    EmbeddingService(provider=provider).embed_query("材料要准备哪些")
    EmbeddingService(provider=provider).embed_query("材料要准备哪些")

    assert provider.queries == ["材料要准备哪些"]
    assert query_embedding_cache() is query_embedding_cache()


def test_concurrent_misses_never_compute_more_than_once_per_worker():
    provider = _CountingProvider()
    service = EmbeddingService(provider=provider)
    workers = 8
    gate = threading.Barrier(workers)

    def ask():
        gate.wait()
        return service.embed_query("审核要多久")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        concurrent = [future.result() for future in [pool.submit(ask) for _ in range(workers)]]

    assert all(result == concurrent[0] for result in concurrent)
    # Computing outside the lock means concurrent misses each reach the provider, but only
    # tasks that checked before the first insert can, and at most `workers` run at once.
    assert 1 <= len(provider.queries) <= workers

    assert service.embed_query("审核要多久") == concurrent[0]
    assert len(provider.queries) <= workers


def test_a_provider_failure_is_not_cached():
    provider = _CountingProvider(fail_times=1)
    service = EmbeddingService(provider=provider)

    with pytest.raises(ProviderError, match="embedding provider unavailable"):
        service.embed_query("申报失败怎么办")
    assert len(query_embedding_cache()) == 0

    assert len(service.embed_query("申报失败怎么办")) == provider.width
    assert provider.queries == ["申报失败怎么办", "申报失败怎么办"]


def test_a_dimension_mismatch_is_not_cached_either():
    service = EmbeddingService(provider=_CountingProvider(width=2048))

    with pytest.raises(ProviderError, match="expected 128, provider returned 2048"):
        service.embed_query("问题")
    assert len(query_embedding_cache()) == 0


def test_clear_drops_every_entry_and_the_counters():
    provider = _CountingProvider()
    service = EmbeddingService(provider=provider)
    service.embed_query("问题一")
    service.embed_query("问题一")

    query_embedding_cache().clear()
    service.embed_query("问题一")

    assert provider.queries == ["问题一", "问题一"]
    assert (query_embedding_cache().hits, query_embedding_cache().misses) == (0, 1)


def test_a_provider_without_a_fingerprint_bypasses_the_cache():
    """Only EmbeddingProvider is public API, so a third-party provider must still work."""

    service = EmbeddingService(provider=_FixedWidthProvider(128))

    assert len(service.embed_query("问题")) == 128
    assert len(service.embed_query("问题")) == 128
    assert len(query_embedding_cache()) == 0


def test_the_offline_hash_provider_is_cached_too():
    service = EmbeddingService(provider=FakeEmbedding(dimension=128))

    assert service.embed_query("平台怎么申报项目") == service.embed_query("平台怎么申报项目")
    assert (query_embedding_cache().hits, query_embedding_cache().misses) == (1, 1)
