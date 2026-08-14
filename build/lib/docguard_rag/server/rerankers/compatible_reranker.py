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

    def rerank(
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
                    "top_n": top_k,
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
        self.client.close()
