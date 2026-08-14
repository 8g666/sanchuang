from docguard_rag.server.core.config import get_settings
from docguard_rag.server.embeddings.compatible_embedding import CompatibleEmbedding
from docguard_rag.server.embeddings.fake_embedding import FakeEmbedding
from docguard_rag.server.llms.compatible_llm import CompatibleLLM
from docguard_rag.server.rerankers.compatible_reranker import CompatibleReranker
from docguard_rag.server.rerankers.overlap_reranker import OverlapReranker


def create_embedding_provider():
    settings = get_settings()
    if settings.embedding_provider == "compatible_api":
        if not settings.embedding_base_url or not settings.embedding_model_name:
            raise ValueError("Embedding base URL and model name are required")
        return CompatibleEmbedding(
            base_url=settings.embedding_base_url,
            model_name=settings.embedding_model_name,
            api_key=settings.embedding_api_key,
            endpoint=settings.embedding_endpoint,
            timeout=settings.model_timeout,
        )
    return FakeEmbedding(dimension=settings.embedding_dimension)


def create_reranker():
    settings = get_settings()
    if not settings.reranker_enabled:
        return None
    if settings.reranker_provider == "compatible_api":
        if not settings.reranker_base_url or not settings.reranker_model_name:
            raise ValueError("Reranker base URL and model name are required")
        return CompatibleReranker(
            base_url=settings.reranker_base_url,
            model_name=settings.reranker_model_name,
            api_key=settings.reranker_api_key,
            endpoint=settings.reranker_endpoint,
            timeout=settings.model_timeout,
        )
    return OverlapReranker()


def create_llm_provider():
    settings = get_settings()
    if settings.llm_provider != "compatible_api":
        return None
    if not settings.llm_base_url or not settings.llm_model_name:
        raise ValueError("LLM base URL and model name are required")
    return CompatibleLLM(
        base_url=settings.llm_base_url,
        model_name=settings.llm_model_name,
        api_key=settings.llm_api_key,
        endpoint=settings.llm_endpoint,
        timeout=settings.llm_timeout,
        temperature=settings.llm_temperature,
    )
