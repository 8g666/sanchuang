import time

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from docguard_rag.server.api.v1.endpoints import documents
from docguard_rag.server.core.security import Principal, get_principal
from docguard_rag.server.database.connection import Base, get_db
from docguard_rag.server.database.models import QueryLogTable
from docguard_rag.server.main import app
from docguard_rag.server.storage.file_storage import FileStorage


def test_upload_query_and_review_workflow(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'workflow.db'}",
        connect_args={"check_same_thread": False},
    )
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    def override_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_db
    monkeypatch.setattr(
        "docguard_rag.server.retrievers.lexical_retriever.SessionLocal", testing_session
    )
    monkeypatch.setattr("docguard_rag.server.services.vector_service.SessionLocal", testing_session)
    monkeypatch.setattr(documents, "SessionLocal", testing_session)
    monkeypatch.setattr(
        documents.ingestion_service,
        "storage",
        FileStorage(tmp_path / "uploads"),
    )

    from fastapi.testclient import TestClient

    try:
        with TestClient(app) as client:
            correction = client.post(
                "/v1/correct",
                json={"text": "系统能够被直接掉用。。"},
            )
            assert correction.status_code == 200
            assert correction.json()["data"]["corrected_text"] == "系统能够被直接调用。"

            upload = client.post(
                "/v1/documents",
                files={
                    "file": (
                        "design.txt",
                        "系统通过 /v1 接口提供服务。系统能够被直接掉用。\n"
                        "该方法在提高性能后，因此能够降低检索时间。",
                        "text/plain",
                    )
                },
            )
            assert upload.status_code == 200, upload.text
            document_id = upload.json()["data"]["document_id"]
            assert upload.json()["data"]["status"] == "READY"

            status = client.get(f"/v1/documents/{document_id}")
            assert status.status_code == 200
            assert status.json()["data"]["chunk_count"] > 0
            assert status.json()["data"]["sentence_count"] == 3

            app.dependency_overrides[get_principal] = lambda: Principal("tenant-b")
            assert client.get(f"/v1/documents/{document_id}").status_code == 404
            del app.dependency_overrides[get_principal]

            query = client.post(
                "/v1/query",
                json={
                    "query": "接口版本是什么",
                    "document_ids": [document_id],
                    "top_k": 3,
                },
            )
            assert query.status_code == 200, query.text
            query_data = query.json()["data"]
            assert query_data["status"] == "ANSWERED"
            assert query_data["verified"] is True
            assert query_data["attempt_count"] >= 1
            assert query_data["sources"]
            assert query_data["sources"][0]["document_id"] == document_id
            assert "/v1" in query_data["sources"][0]["text"]
            with testing_session() as db:
                audit = (
                    db.query(QueryLogTable)
                    .filter(QueryLogTable.request_id == query.json()["request_id"])
                    .one()
                )
                assert audit.tenant_id == "default"
                assert audit.retrieved_chunk_ids_json != "[]"

            queued = client.post(
                "/v1/documents",
                files={"file": ("async.txt", "background ingestion", "text/plain")},
                data={"async": "true"},
            )
            assert queued.status_code == 200, queued.text
            queued_data = queued.json()["data"]
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                job = client.get(f"/v1/jobs/{queued_data['job_id']}")
                assert job.status_code == 200
                if job.json()["data"]["status"] in {"COMPLETED", "FAILED"}:
                    break
                time.sleep(0.01)
            assert job.json()["data"]["status"] == "COMPLETED"
            assert client.get(f"/v1/documents/{queued_data['document_id']}").status_code == 200

            review = client.post(
                "/v1/reviews",
                json={
                    "document_ids": [document_id],
                    "check_types": ["TYPO", "SEMANTIC_UNCLEAR"],
                    "mode": "BALANCED",
                },
            )
            assert review.status_code == 200, review.text
            review_id = review.json()["data"]["review_id"]
            job = client.get(f"/v1/jobs/{review_id}")
            assert job.status_code == 200
            assert job.json()["data"]["progress"] == 100
            issues = client.get(f"/v1/reviews/{review_id}/issues")
            assert issues.status_code == 200, issues.text
            issue_items = issues.json()["data"]["items"]
            issue_types = {item["issue_type"] for item in issue_items}
            assert {"TYPO", "SEMANTIC_UNCLEAR"} <= issue_types

            update = client.patch(
                f"/v1/issues/{issue_items[0]['issue_id']}",
                json={"status": "ACCEPTED", "comment": "已确认"},
            )
            assert update.status_code == 200
            assert update.json()["data"]["status"] == "ACCEPTED"

            replace = client.put(
                f"/v1/documents/{document_id}",
                files={"file": ("replacement.txt", "replacement lifecycle text", "text/plain")},
                data={"metadata": '{"revision":2}'},
            )
            assert replace.status_code == 200, replace.text
            assert replace.json()["data"]["document_id"] == document_id

            reindex = client.post(f"/v1/documents/{document_id}/reindex")
            assert reindex.status_code == 200, reindex.text
            assert reindex.json()["data"]["chunks_count"] > 0

            app.dependency_overrides[get_principal] = lambda: Principal("tenant-b")
            assert client.delete(f"/v1/documents/{document_id}").status_code == 404
            del app.dependency_overrides[get_principal]

            deleted = client.delete(f"/v1/documents/{document_id}")
            assert deleted.status_code == 200, deleted.text
            assert deleted.json()["data"] == {
                "document_id": document_id,
                "status": "DELETED",
            }
            assert client.get(f"/v1/documents/{document_id}").status_code == 404
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
