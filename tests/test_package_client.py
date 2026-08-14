import json

import httpx

from docguard_rag import DocGuardClient


def test_http_client_preserves_versioned_query_interface():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/query"
        payload = json.loads(request.content)
        assert "corrective" in payload
        assert "max_retries" in payload
        return httpx.Response(
            200,
            json={
                "code": 0,
                "message": "success",
                "request_id": "req_test",
                "data": {
                    "answer": "使用 /v1。",
                    "confidence": 0.95,
                    "sources": [
                        {
                            "chunk_id": "chunk_1",
                            "document_id": "doc_1",
                            "filename": "design.md",
                            "page": None,
                            "section_path": ["接口设计"],
                            "text": "使用 /v1。",
                            "score": 1.0,
                        }
                    ],
                },
            },
        )

    http_client = httpx.Client(
        base_url="http://docguard.test",
        transport=httpx.MockTransport(handler),
    )
    client = DocGuardClient(
        "http://docguard.test",
        http_client=http_client,
    )

    result = client.query("接口版本", document_ids=["doc_1"])

    assert result.answer == "使用 /v1。"
    assert result.sources[0].section_path == ["接口设计"]
    client.close()
    http_client.close()


def test_http_client_exposes_document_lifecycle(tmp_path):
    source = tmp_path / "replacement.txt"
    source.write_text("replacement", encoding="utf-8")
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.method == "DELETE":
            data = {"document_id": "doc_1", "status": "DELETED"}
        else:
            data = {
                "document_id": "doc_1",
                "status": "READY",
                "blocks_count": 1,
                "chunks_count": 1,
                "vectors_count": 1,
            }
        return httpx.Response(
            200,
            json={"code": 0, "message": "success", "request_id": "req_1", "data": data},
        )

    http_client = httpx.Client(
        base_url="http://docguard.test",
        transport=httpx.MockTransport(handler),
    )
    with DocGuardClient("http://docguard.test", http_client=http_client) as client:
        assert client.documents.update("doc_1", source).status == "READY"
        assert client.documents.reindex("doc_1").chunks_count == 1
        assert client.documents.delete("doc_1").status == "DELETED"

    assert requests == [
        ("PUT", "/v1/documents/doc_1"),
        ("POST", "/v1/documents/doc_1/reindex"),
        ("DELETE", "/v1/documents/doc_1"),
    ]
    http_client.close()
