import httpx

from docguard_rag.server.core.provider_errors import ProviderError
from docguard_rag.server.rerankers.base import Reranker


class CompatibleReranker(Reranker):
    """Reranker for common HTTP APIs accepting query and documents."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        endpoint: str = "/rerank",
        timeout: float = 30.0,
        batch_size: int = 128,
    ):
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.model_name = model_name
        self.endpoint = endpoint
        self.batch_size = max(1, batch_size)
        self._base_url = base_url.rstrip("/")
        self._headers = headers
        self._timeout = timeout
        self._client: httpx.Client | None = None

    @property
    def fingerprint(self) -> tuple:
        """Identifies the model whose scores this provider produces."""

        return ("compatible_api", self.model_name)

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

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        if not candidates:
            return []
        if len(candidates) <= self.batch_size:
            return self._rerank_batch(query, candidates, top_k)
        # Scores are per (query, document) pair, so splitting and merging on score is
        # equivalent to one oversized request the API would reject.
        scored: list[dict] = []
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            scored.extend(self._rerank_batch(query, batch, len(batch)))
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def _rerank_batch(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        documents = [item.get("metadata", {}).get("text", "") for item in candidates]
        try:
            response = self.client.post(
                self.endpoint,
                json={
                    "model": self.model_name,
                    "query": query,
                    "documents": documents,
                    "top_n": min(top_k, len(documents)),
                },
            )
            response.raise_for_status()
            body = response.json()
            results = body.get("results", body.get("data", []))
            reranked = []
            for result in results:
                index = int(result["index"])
                item = dict(candidates[index])
                item["score"] = float(result.get("relevance_score", result.get("score", 0.0)))
                reranked.append(item)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"Reranker provider request failed: {exc}") from exc
        return reranked[:top_k]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
