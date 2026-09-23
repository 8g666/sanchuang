"""Process-global LRU cache for query embeddings.

A query embedding is a pure function of (model, text): unlike the corpus snapshot it does
not depend on the database, so there is nothing to invalidate against. The provider
fingerprint in the key *is* the invalidation mechanism — change the model or the dimension
and every key changes, leaving the old entries unreachable until the LRU drops them.
"""

import threading
from collections import OrderedDict
from collections.abc import Callable

from docguard_rag.server.core.config import get_settings


class QueryEmbeddingCache:
    def __init__(self, maxsize: int = 1024):
        self._maxsize = maxsize
        self._entries: OrderedDict[tuple, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key: tuple, compute: Callable[[], list[float]]) -> list[float]:
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self.hits += 1
                return cached

        # Computed outside the lock: holding it across a 30-second HTTP call would queue
        # every query in the process behind one slow request. Concurrent misses on the same
        # key therefore each hit the provider — duplicated work, never a wrong answer.
        vector = compute()

        with self._lock:
            self.misses += 1
            self._entries[key] = vector
            self._entries.move_to_end(key)
            while len(self._entries) > self._maxsize:
                self._entries.popitem(last=False)
        # The stored list is handed out rather than copied: consumers pass it straight to
        # np.asarray, which copies, and nothing downstream mutates it in place.
        return vector

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


_CACHE: QueryEmbeddingCache | None = None
_CACHE_LOCK = threading.Lock()


def query_embedding_cache() -> QueryEmbeddingCache:
    """Resolve the cache shared by every embedding service in the process.

    Shared across engines, not per engine: the two retrieval stacks each build their own
    EmbeddingService, and splitting the cache between them would only halve the hit rate.
    """

    global _CACHE
    cache = _CACHE
    if cache is None:
        with _CACHE_LOCK:
            cache = _CACHE
            if cache is None:
                cache = QueryEmbeddingCache(get_settings().embedding_query_cache_size)
                _CACHE = cache
    return cache
