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
        dimensions: int | None = None,
        batch_size: int = 64,
    ):
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.model_name = model_name
        self.endpoint = endpoint
        self.dimensions = dimensions
        self.batch_size = max(1, batch_size)
        self._base_url = base_url.rstrip("/")
        self._headers = headers
        self._timeout = timeout
        self._client: httpx.Client | None = None

    @property
    def fingerprint(self) -> tuple:
        """Identifies the model whose vectors this provider produces."""

        return ("compatible_api", self.model_name, self.dimensions)

    @property
    def client(self) -> httpx.Client:
        # Built on first use rather than in __init__: providers are constructed at import
        # time and closed at shutdown, and httpx refuses to reuse a closed client.
        if self._client is None:
            self._client = httpx.Client(
                base_url=self._base_url,
                headers=self._headers,
                timeout=self._timeout,
            )
        return self._client

    @client.setter
    def client(self, client: httpx.Client) -> None:
        self._client = client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload: dict = {"model": self.model_name, "input": texts}
        # Sent only when configured: providers that predate the OpenAI `dimensions` field
        # reject unknown keys, and omitting it keeps those requests byte-identical.
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        try:
            response = self.client.post(self.endpoint, json=payload)
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
        if self._client is not None:
            self._client.close()
            self._client = None
