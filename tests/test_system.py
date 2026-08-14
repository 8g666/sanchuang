from fastapi.testclient import TestClient

from docguard_rag.server.main import app

client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["code"] == 0
    assert body["message"] == "success"
    assert body["data"]["status"] == "healthy"
    assert body["request_id"].startswith("req_")
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_readiness_check() -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["data"]["status"] == "ready"


def test_system_info() -> None:
    response = client.get("/v1/system/info")

    assert response.status_code == 200

    body = response.json()

    assert body["code"] == 0
    assert body["data"]["app_name"] == "DocGuard-RAG"
    assert body["data"]["app_version"] == "0.6.0"
    assert body["data"]["environment"] == "development"


def test_removed_legacy_routes_are_not_available() -> None:
    assert client.get("/api/v1/system/info").status_code == 404
    assert client.post("/v1/query/").status_code == 404
    assert client.post("/v1/documents/upload").status_code in {404, 405}


def test_request_ids_are_unique() -> None:
    first_response = client.get("/health")
    second_response = client.get("/health")

    first_request_id = first_response.json()["request_id"]
    second_request_id = second_response.json()["request_id"]

    assert first_request_id != second_request_id
