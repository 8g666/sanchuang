import httpx

from docguard_rag.server.core.provider_errors import ProviderError
from docguard_rag.server.embeddings.base import EmbeddingProvider


class CompatibleEmbedding(EmbeddingProvider):
    """Embedding provider for OpenAI-compatible HTTP APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        endpoint: str = "/embeddings",
        timeout: float = 30.0,
    ):
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.model_name = model_name
        self.endpoint = endpoint
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self.client.post(
                self.endpoint,
                json={"model": self.model_name, "input": texts},
            )
            response.raise_for_status()
            body = response.json()
            data = sorted(body["data"], key=lambda item: item.get("index", 0))
            vectors = [item["embedding"] for item in data]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise ProviderError(f"Embedding provider request failed: {exc}") from exc
        if len(vectors) != len(texts):
            raise ProviderError("Embedding provider returned an unexpected vector count")
        return vectors

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

    def close(self) -> None:
        self.client.close()
