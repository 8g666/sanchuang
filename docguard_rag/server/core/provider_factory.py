import logging
import threading

from docguard_rag.server.core.config import get_settings
from docguard_rag.server.embeddings.compatible_embedding import CompatibleEmbedding
from docguard_rag.server.embeddings.fake_embedding import FakeEmbedding
from docguard_rag.server.llms.compatible_llm import CompatibleLLM
from docguard_rag.server.rerankers.compatible_reranker import CompatibleReranker
from docguard_rag.server.rerankers.overlap_reranker import OverlapReranker

logger = logging.getLogger(__name__)

_CLOSEABLES: list = []
_CLOSE_LOCK = threading.Lock()


def _track(provider):
    """Register a provider so shutdown can find it.

    Providers are built in five places across two independent retrieval stacks plus the
    ingestion stack, and none of them are reachable from the FastAPI app object, so this
    registry is the only handle a lifespan has on them.
    """

    if provider is not None and hasattr(provider, "close"):
        with _CLOSE_LOCK:
            # Identity check, not equality: EmbeddingService and friends are constructed
            # repeatedly and would otherwise register the same provider twice.
            if not any(provider is item for item in _CLOSEABLES):
                _CLOSEABLES.append(provider)
    return provider


def close_providers() -> int:
    """Close every tracked provider once and return how many were closed."""

    with _CLOSE_LOCK:
        # Copy and clear inside the lock so a provider built during shutdown is tracked
        # for the next round instead of being dropped.
        pending = list(_CLOSEABLES)
        _CLOSEABLES.clear()
    closed = 0
    for provider in pending:
        try:
            provider.close()
        except Exception:
            logger.exception("Failed to close provider %r", provider)
        else:
            closed += 1
    return closed


def create_embedding_provider():
    settings = get_settings()
    if settings.embedding_provider == "compatible_api":
        if not settings.embedding_base_url or not settings.embedding_model_name:
            raise ValueError("Embedding base URL and model name are required")
        return _track(
            CompatibleEmbedding(
                base_url=settings.embedding_base_url,
                model_name=settings.embedding_model_name,
                api_key=settings.embedding_api_key,
                endpoint=settings.embedding_endpoint,
                timeout=settings.model_timeout,
                dimensions=settings.embedding_dimension,
                batch_size=settings.embedding_batch_size,
            )
        )
    return FakeEmbedding(dimension=settings.embedding_dimension)


def create_reranker():
    settings = get_settings()
    if not settings.reranker_enabled:
        return None
    if settings.reranker_provider == "compatible_api":
        if not settings.reranker_base_url or not settings.reranker_model_name:
            raise ValueError("Reranker base URL and model name are required")
        return _track(
            CompatibleReranker(
                base_url=settings.reranker_base_url,
                model_name=settings.reranker_model_name,
                api_key=settings.reranker_api_key,
                endpoint=settings.reranker_endpoint,
                timeout=settings.model_timeout,
            )
        )
    return OverlapReranker()


def create_llm_provider():
    settings = get_settings()
    if settings.llm_provider != "compatible_api":
        return None
    if not settings.llm_base_url or not settings.llm_model_name:
        raise ValueError("LLM base URL and model name are required")
    return _track(
        CompatibleLLM(
            base_url=settings.llm_base_url,
            model_name=settings.llm_model_name,
            api_key=settings.llm_api_key,
            endpoint=settings.llm_endpoint,
            timeout=settings.llm_timeout,
            temperature=settings.llm_temperature,
        )
    )
