import json

import httpx

from docguard_rag.server.models.embedding import EmbeddingRecord
from docguard_rag.server.vectorstores.qdrant_vector_store import QdrantVectorStore


def test_qdrant_vector_store_lifecycle():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        requests.append((request.method, request.url.path, body))
        if request.method == "GET":
            return httpx.Response(404, json={"status": "not found"})
        if request.url.path.endswith("/points/search"):
            return httpx.Response(
                200,
                json={
                    "result": [
                        {
                            "id": "point-id",
                            "score": 0.91,
                            "payload": {"chunk_id": "chunk-1", "tenant_id": "tenant-a"},
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"result": {"status": "ok"}})

    client = httpx.Client(base_url="http://qdrant", transport=httpx.MockTransport(handler))
    store = QdrantVectorStore(
        base_url="http://qdrant",
        collection="documents",
        dimension=3,
        client=client,
    )
    store.add_many(
        [
            EmbeddingRecord(
                chunk_id="chunk-1",
                vector=[1.0, 0.0, 0.0],
                metadata={"tenant_id": "tenant-a"},
            )
        ]
    )
    result = store.search([1.0, 0.0, 0.0], 5, {"tenant_id": "tenant-a"})
    store.delete(["chunk-1"])

    assert result[0]["chunk_id"] == "chunk-1"
    create = next(
        item for item in requests if item[0] == "PUT" and item[1] == "/collections/documents"
    )
    assert create[2]["vectors"] == {"size": 3, "distance": "Cosine"}
    upsert = next(item for item in requests if item[1].endswith("/points"))
    assert upsert[2]["points"][0]["payload"]["chunk_id"] == "chunk-1"
    search = next(item for item in requests if item[1].endswith("/points/search"))
    assert search[2]["filter"]["must"][0]["match"] == {"value": "tenant-a"}
    delete = next(item for item in requests if item[1].endswith("/points/delete"))
    assert delete[2]["points"] == [upsert[2]["points"][0]["id"]]
