from docguard_rag.server.core.config import get_settings
from docguard_rag.server.vectorstores.qdrant_vector_store import QdrantVectorStore
from docguard_rag.server.vectorstores.simple_vector_store import SimpleVectorStore


def create_vector_store():
    settings = get_settings()
    if settings.vector_store_provider == "qdrant":
        return QdrantVectorStore(
            base_url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            dimension=settings.embedding_dimension,
            api_key=settings.qdrant_api_key,
            timeout=settings.qdrant_timeout,
        )
    if settings.vector_store_provider != "local":
        raise ValueError(f"Unsupported vector store provider: {settings.vector_store_provider}")
    return SimpleVectorStore()
