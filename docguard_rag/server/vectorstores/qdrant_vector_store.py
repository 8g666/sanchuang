from uuid import NAMESPACE_URL, uuid5

import httpx

from docguard_rag.server.core.provider_errors import ProviderError
from docguard_rag.server.models.embedding import EmbeddingRecord
from docguard_rag.server.vectorstores.base import VectorStore


class QdrantVectorStore(VectorStore):
    """Qdrant HTTP adapter using deterministic UUID point identifiers."""

    def __init__(
        self,
        *,
        base_url: str,
        collection: str,
        dimension: int,
        api_key: str | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ):
        headers = {"api-key": api_key} if api_key else None
        self.collection = collection
        self.dimension = dimension
        self._owns_client = client is None
        self.client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
        )
        self._collection_ready = False

    def add(self, chunk_id: str, vector: list[float], metadata: dict) -> None:
        self.add_many([EmbeddingRecord(chunk_id=chunk_id, vector=vector, metadata=metadata)])

    def add_many(self, embeddings: list[EmbeddingRecord]) -> None:
        if not embeddings:
            return
        self._ensure_collection()
        points = [
            {
                "id": self._point_id(item["chunk_id"]),
                "vector": item["vector"],
                "payload": {**item["metadata"], "chunk_id": item["chunk_id"]},
            }
            for item in embeddings
        ]
        self._request(
            "PUT",
            f"/collections/{self.collection}/points",
            params={"wait": "true"},
            json={"points": points},
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[dict]:
        self._ensure_collection()
        body = {
            "vector": query_vector,
            "limit": top_k,
            "with_payload": True,
        }
        qdrant_filter = self._filter(filters or {})
        if qdrant_filter:
            body["filter"] = qdrant_filter
        response = self._request(
            "POST",
            f"/collections/{self.collection}/points/search",
            json=body,
        )
        return [
            {
                "chunk_id": item.get("payload", {}).get("chunk_id", str(item["id"])),
                "score": float(item.get("score", 0.0)),
                "metadata": item.get("payload", {}),
            }
            for item in response.get("result", [])
        ]

    def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        self._ensure_collection()
        self._request(
            "POST",
            f"/collections/{self.collection}/points/delete",
            params={"wait": "true"},
            json={"points": [self._point_id(chunk_id) for chunk_id in chunk_ids]},
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        try:
            response = self.client.get(f"/collections/{self.collection}")
        except httpx.HTTPError as exc:
            raise ProviderError(f"Qdrant collection check failed: {exc}") from exc
        if response.status_code == 404:
            self._request(
                "PUT",
                f"/collections/{self.collection}",
                json={"vectors": {"size": self.dimension, "distance": "Cosine"}},
            )
        else:
            try:
                response.raise_for_status()
                payload = response.json()
                vector_config = payload["result"]["config"]["params"]["vectors"]
                existing_dimension = vector_config.get("size")
                if existing_dimension is not None and existing_dimension != self.dimension:
                    raise ProviderError(
                        "Qdrant collection dimension mismatch: "
                        f"expected {self.dimension}, found {existing_dimension}"
                    )
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                raise ProviderError(f"Qdrant collection check failed: {exc}") from exc
        self._collection_ready = True

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self.client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(f"Qdrant request failed: {exc}") from exc

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"docguard:{chunk_id}"))

    @staticmethod
    def _filter(filters: dict) -> dict | None:
        conditions = []
        for key, value in filters.items():
            if value is None:
                continue
            match = (
                {"any": list(value)} if isinstance(value, list | tuple | set) else {"value": value}
            )
            conditions.append({"key": key, "match": match})
        return {"must": conditions} if conditions else None
