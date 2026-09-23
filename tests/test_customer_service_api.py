from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from docguard_rag.server.api.v1.endpoints import customer_service as endpoint
from docguard_rag.server.api.v1.endpoints import documents
from docguard_rag.server.core.security import Principal, get_principal
from docguard_rag.server.database.connection import Base, get_db
from docguard_rag.server.database.models import (
    QueryLogTable,
    SupportMessageTable,
    SupportSessionTable,
)
from docguard_rag.server.main import app
from docguard_rag.server.services.customer_service import CustomerServiceService
from docguard_rag.server.storage.file_storage import FileStorage


def _isolated_app(tmp_path, monkeypatch):
    """Bind every import-time SessionLocal in the request path to a throwaway engine.

    The endpoint module builds its CustomerServiceService at import time, so overriding
    get_db alone still leaves the retrievers reading whatever connection.py bound.
    """

    engine = create_engine(
        f"sqlite:///{tmp_path / 'customer-service.db'}",
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
    monkeypatch.setattr(documents.ingestion_service, "storage", FileStorage(tmp_path / "uploads"))
    return engine, testing_session


def test_customer_service_endpoint_handoffs_without_knowledge_base(tmp_path, monkeypatch):
    engine, _ = _isolated_app(tmp_path, monkeypatch)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/support/chat",
                json={"message": "请转人工客服", "handoff_if_unresolved": True},
            )
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["status"] == "HANDOFF_REQUIRED"
        assert data["handoff_required"] is True
        assert data["intent"] == "human"
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_customer_service_endpoint_answers_from_the_knowledge_base(tmp_path, monkeypatch):
    engine, testing_session = _isolated_app(tmp_path, monkeypatch)
    try:
        with TestClient(app) as client:
            upload = client.post(
                "/v1/documents",
                files={
                    "file": (
                        "service.md",
                        "# 通境三创服务平台\n平台支持项目申报、材料提交和进度查询。\n",
                        "text/markdown",
                    )
                },
            )
            assert upload.status_code == 200, upload.text
            document_id = upload.json()["data"]["document_id"]

            response = client.post(
                "/v1/support/chat",
                json={"message": "平台支持哪些服务？", "document_ids": [document_id]},
            )
            assert response.status_code == 200, response.text
            data = response.json()["data"]
            assert data["intent"] == "platform"
            assert data["query_status"] == "ANSWERED"
            assert data["sources"]
            assert "项目申报" in data["message"]
            assert data["session_id"]

            request_id = response.json()["request_id"]
            with testing_session() as db:
                audit = db.query(QueryLogTable).filter(QueryLogTable.request_id == request_id).one()
                assert audit.tenant_id == "default"
                assert audit.retrieved_chunk_ids_json != "[]"
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def _stored(testing_session, session_id, tenant_id="default"):
    with testing_session() as db:
        rows = db.execute(
            select(SupportMessageTable.role, SupportMessageTable.content)
            .where(
                SupportMessageTable.tenant_id == tenant_id,
                SupportMessageTable.session_id == session_id,
            )
            .order_by(SupportMessageTable.id)
        ).all()
        session = db.execute(
            select(SupportSessionTable).where(
                SupportSessionTable.tenant_id == tenant_id,
                SupportSessionTable.session_id == session_id,
            )
        ).scalar_one_or_none()
    return [(role, content) for role, content in rows], session


def test_two_turns_on_one_session_accumulate_in_order(tmp_path, monkeypatch):
    engine, testing_session = _isolated_app(tmp_path, monkeypatch)
    try:
        with TestClient(app) as client:
            first = client.post(
                "/v1/support/chat",
                json={"message": "第一个问题", "session_id": "cs_api"},
            )
            second = client.post(
                "/v1/support/chat",
                json={"message": "第二个问题", "session_id": "cs_api"},
            )
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert first.json()["data"]["session_id"] == "cs_api"

        rows, session = _stored(testing_session, "cs_api")
        assert [role for role, _ in rows] == ["user", "assistant", "user", "assistant"]
        assert [content for role, content in rows if role == "user"] == [
            "第一个问题",
            "第二个问题",
        ]
        assert session.message_count == 4
        assert session.status == "ACTIVE"
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_history_survives_a_new_service_instance(tmp_path, monkeypatch):
    """Stands in for a process restart: a fresh instance holding no in-memory state."""

    engine, testing_session = _isolated_app(tmp_path, monkeypatch)
    original = endpoint.customer_service
    try:
        with TestClient(app) as client:
            client.post(
                "/v1/support/chat", json={"message": "重启前的问题", "session_id": "cs_boot"}
            )

        monkeypatch.setattr(endpoint, "customer_service", CustomerServiceService())
        with TestClient(app) as restarted:
            restarted.post(
                "/v1/support/chat", json={"message": "重启后的问题", "session_id": "cs_boot"}
            )

        rows, _ = _stored(testing_session, "cs_boot")
        assert [content for role, content in rows if role == "user"] == [
            "重启前的问题",
            "重启后的问题",
        ]
    finally:
        monkeypatch.setattr(endpoint, "customer_service", original)
        app.dependency_overrides.clear()
        engine.dispose()


def test_delete_soft_resets_and_reports_unknown_sessions(tmp_path, monkeypatch):
    engine, testing_session = _isolated_app(tmp_path, monkeypatch)
    try:
        with TestClient(app) as client:
            client.post(
                "/v1/support/chat", json={"message": "待清除的问题", "session_id": "cs_del"}
            )

            deleted = client.delete("/v1/support/sessions/cs_del")
            unknown = client.delete("/v1/support/sessions/cs_never_existed")

        # Stays 200 for an unknown id: callers treat this as fire-and-forget.
        assert deleted.status_code == 200, deleted.text
        assert deleted.json()["data"] == {
            "session_id": "cs_del",
            "status": "RESET",
            "found": True,
        }
        assert unknown.status_code == 200, unknown.text
        assert unknown.json()["data"]["found"] is False

        rows, session = _stored(testing_session, "cs_del")
        assert rows == []
        # The session row is kept as an audit trail that the conversation happened.
        assert session is not None
        assert session.status == "RESET"
        assert session.message_count == 0
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_two_tenants_sharing_a_session_id_stay_isolated(tmp_path, monkeypatch):
    engine, testing_session = _isolated_app(tmp_path, monkeypatch)
    try:
        with TestClient(app) as client:
            for tenant in ("alpha", "beta"):
                app.dependency_overrides[get_principal] = lambda tenant=tenant: Principal(
                    tenant_id=tenant
                )
                response = client.post(
                    "/v1/support/chat",
                    json={"message": f"{tenant} 的问题", "session_id": "cs_shared"},
                )
                assert response.status_code == 200, response.text

            app.dependency_overrides[get_principal] = lambda: Principal(tenant_id="alpha")
            deleted = client.delete("/v1/support/sessions/cs_shared")

        assert deleted.json()["data"]["found"] is True

        alpha_rows, alpha_session = _stored(testing_session, "cs_shared", tenant_id="alpha")
        beta_rows, beta_session = _stored(testing_session, "cs_shared", tenant_id="beta")

        assert alpha_rows == []
        assert alpha_session.status == "RESET"
        # alpha resetting its own session must not touch beta's conversation.
        assert [content for role, content in beta_rows if role == "user"] == ["beta 的问题"]
        assert beta_session.status == "ACTIVE"
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
