import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from docguard_rag import DocGuard
from docguard_rag.server.api.v1.endpoints import documents
from docguard_rag.server.core.provider_errors import ProviderError
from docguard_rag.server.core.retrieval_index import index_for_engine
from docguard_rag.server.core.security import Principal, get_principal
from docguard_rag.server.database.connection import Base, get_db
from docguard_rag.server.database.models import DocumentChunkTable, DocumentVectorTable
from docguard_rag.server.main import app
from docguard_rag.server.services.embedding_service import EmbeddingService
from docguard_rag.server.services.reembed_service import ReembedService
from docguard_rag.server.storage.file_storage import FileStorage


class CountingEmbedding:
    """Deterministic fixed-width vectors that record every batch they were handed."""

    def __init__(self, dimension: int = 128, fail_after: int | None = None):
        self.dimension = dimension
        self.fail_after = fail_after
        self.batches: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        if self.fail_after is not None and len(self.batches) > self.fail_after:
            raise ProviderError("embedding provider unavailable")
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)

    def close(self) -> None:
        return None

    def _vector(self, text: str) -> list[float]:
        rng = np.random.default_rng(sum(text.encode("utf-8")) % 2**32)
        return rng.random(self.dimension).tolist()


def _write(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _guard(tmp_path: Path, tenant_id: str = "default") -> DocGuard:
    return DocGuard(
        database_url=f"sqlite:///{tmp_path / 'reembed.db'}",
        upload_dir=tmp_path / "uploads",
        tenant_id=tenant_id,
    )


def _service(guard: DocGuard, provider) -> ReembedService:
    return ReembedService(
        embedding_service=EmbeddingService(provider=provider),
        vector_service=guard._vector_service,
    )


def _reembed(guard: DocGuard, provider, **kwargs):
    with guard._session() as db:
        return _service(guard, provider).reembed(db, tenant_id=guard.tenant_id, **kwargs)


def _chunks(guard: DocGuard) -> list:
    with guard._session() as db:
        return db.execute(
            select(DocumentChunkTable.chunk_id, DocumentChunkTable.text).order_by(
                DocumentChunkTable.id
            )
        ).all()


def _vectors(guard: DocGuard) -> dict[str, tuple[str, str | None]]:
    with guard._session() as db:
        rows = db.execute(
            select(
                DocumentVectorTable.chunk_id,
                DocumentVectorTable.vector,
                DocumentVectorTable.metadata_json,
            )
        ).all()
    return {row.chunk_id: (row.vector, row.metadata_json) for row in rows}


def test_reembedding_reuses_every_chunk_id(tmp_path):
    with _guard(tmp_path) as guard:
        guard.documents.upload(_write(tmp_path, "kepu.md", "光合作用发生在叶绿体中，需要光能。"))
        guard.documents.upload(_write(tmp_path, "cell.md", "细胞呼吸发生在线粒体中，产生能量。"))
        chunks = _chunks(guard)
        before = _vectors(guard)

        provider = CountingEmbedding()
        result = _reembed(guard, provider)
        after = _vectors(guard)

    assert result.reembedded == len(chunks) > 0
    assert set(after) == {chunk_id for chunk_id, _ in chunks}
    assert set(before) == set(after)
    assert all(after[chunk_id][0] != before[chunk_id][0] for chunk_id in before)


def test_existing_metadata_survives_byte_for_byte(tmp_path):
    with _guard(tmp_path) as guard:
        guard.documents.upload(
            _write(tmp_path, "kepu.md", "光合作用发生在叶绿体中，需要光能。"),
            collection_id="kepu",
            metadata={"topic": "biology"},
        )
        before = _vectors(guard)

        _reembed(guard, CountingEmbedding())
        after = _vectors(guard)

    assert [after[chunk_id][1] for chunk_id in before] == [
        before[chunk_id][1] for chunk_id in before
    ]


def test_batching_preserves_input_order(tmp_path):
    with _guard(tmp_path) as guard:
        for index in range(5):
            guard.documents.upload(
                _write(tmp_path, f"d{index}.md", f"第 {index} 份文档的内容彼此都不相同。")
            )
        chunks = _chunks(guard)

        provider = CountingEmbedding()
        result = _reembed(guard, provider, batch_size=2)

    assert result.batches == 3
    assert [len(batch) for batch in provider.batches] == [2, 2, 1]
    assert [text for batch in provider.batches for text in batch] == [text for _, text in chunks]
    assert result.reembedded == 5


def test_reembedded_vectors_reach_the_retrieval_cache(tmp_path):
    with _guard(tmp_path) as guard:
        guard.documents.upload(_write(tmp_path, "kepu.md", "光合作用发生在叶绿体中，需要光能。"))
        guard.query("光合作用", generate_answer=False)
        index = index_for_engine(guard._engine)
        stale = {chunk_id: row[0] for chunk_id, row in _vectors(guard).items()}
        rebuilds_before = index.rebuilds

        provider = CountingEmbedding()
        _reembed(guard, provider)
        searched = guard._vector_service.search(provider.embed_query("光合作用"), top_k=5)
        fresh = _vectors(guard)

    assert index.rebuilds > rebuilds_before
    assert searched
    for item in searched:
        snapshot_vector = index.snapshot.vectors[item["chunk_id"]]["vector"]
        assert snapshot_vector == json.loads(fresh[item["chunk_id"]][0])
        assert fresh[item["chunk_id"]][0] != stale[item["chunk_id"]]


def test_only_the_calling_tenant_is_reembedded(tmp_path):
    with DocGuard(
        database_url=f"sqlite:///{tmp_path / 'reembed.db'}",
        upload_dir=tmp_path / "uploads",
        tenant_id="alpha",
    ) as alpha:
        alpha.documents.upload(_write(tmp_path, "alpha.md", "光合作用发生在叶绿体中。"))
        # Chunks and vectors carry no tenant column; only the JOIN to documents scopes
        # them, so capture alpha's ids before beta's rows exist.
        alpha_chunks = {chunk_id for chunk_id, _ in _chunks(alpha)}
        with DocGuard(
            database_url=f"sqlite:///{tmp_path / 'reembed.db'}",
            upload_dir=tmp_path / "uploads",
            tenant_id="beta",
        ) as beta:
            beta.documents.upload(_write(tmp_path, "beta.md", "细胞呼吸发生在线粒体中。"))
            beta_chunks = {chunk_id for chunk_id, _ in _chunks(beta)} - alpha_chunks
            untouched = {k: v for k, v in _vectors(alpha).items() if k in alpha_chunks}

            provider = CountingEmbedding()
            result = _reembed(beta, provider)
            still_untouched = {k: v for k, v in _vectors(alpha).items() if k in alpha_chunks}

    assert result.reembedded == len(beta_chunks) == 1
    assert [text for batch in provider.batches for text in batch] == ["细胞呼吸发生在线粒体中。"]
    assert still_untouched == untouched
    assert alpha_chunks.isdisjoint(beta_chunks)


def test_a_provider_failure_leaves_committed_batches_behind(tmp_path):
    with _guard(tmp_path) as guard:
        for index in range(4):
            guard.documents.upload(
                _write(tmp_path, f"d{index}.md", f"第 {index} 份文档的内容彼此都不相同。")
            )
        chunks = _chunks(guard)
        before = _vectors(guard)

        provider = CountingEmbedding(fail_after=1)
        with guard._session() as db:
            with pytest.raises(ProviderError, match="embedding provider unavailable"):
                _service(guard, provider).reembed(db, tenant_id="default", batch_size=2)
        after = _vectors(guard)

    changed = [chunk_id for chunk_id, _ in chunks if after[chunk_id][0] != before[chunk_id][0]]
    assert changed == [chunk_id for chunk_id, _ in chunks[:2]]


def test_chunks_with_no_stored_vector_are_reported_not_hidden(tmp_path):
    with _guard(tmp_path) as guard:
        document = guard.documents.upload(
            _write(tmp_path, "kepu.md", "光合作用发生在叶绿体中，需要光能。"),
            collection_id="kepu",
            metadata={"topic": "biology"},
        )
        with guard._session() as db:
            db.query(DocumentVectorTable).delete()
            db.commit()

        result = _reembed(guard, CountingEmbedding())
        after = _vectors(guard)
        texts = {chunk_id: text for chunk_id, text in _chunks(guard)}

    assert result.reembedded == len(after) > 0
    assert result.embedded_without_offsets == result.reembedded
    assert result.document_ids_needing_reindex == [document.document_id]
    for chunk_id, (_, metadata_json) in after.items():
        metadata = json.loads(metadata_json)
        # Offsets are derived from a chunk's position in its block and never persisted,
        # so there is nothing to copy them from. Everything else is rebuilt from the JOIN.
        assert metadata["start_offset"] is None
        assert metadata["end_offset"] is None
        assert metadata["text"] == texts[chunk_id]
        assert metadata["tenant_id"] == "default"
        assert metadata["collection_id"] == "kepu"
        assert metadata["custom_metadata"] == {"topic": "biology"}


def _isolated_app(tmp_path, monkeypatch):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'reembed-api.db'}",
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
    monkeypatch.setattr("docguard_rag.server.services.vector_service.SessionLocal", testing_session)
    monkeypatch.setattr(
        "docguard_rag.server.retrievers.lexical_retriever.SessionLocal", testing_session
    )
    monkeypatch.setattr(documents, "SessionLocal", testing_session)
    monkeypatch.setattr(documents.ingestion_service, "storage", FileStorage(tmp_path / "uploads"))
    return engine, testing_session


def _stored_vectors(session_factory) -> dict[str, str]:
    with session_factory() as db:
        rows = db.execute(select(DocumentVectorTable.chunk_id, DocumentVectorTable.vector)).all()
    return {row.chunk_id: row.vector for row in rows}


def test_the_reembed_endpoint_rewrites_the_corpus(tmp_path, monkeypatch):
    engine, session = _isolated_app(tmp_path, monkeypatch)
    provider = CountingEmbedding()
    monkeypatch.setattr(
        documents.reembed_service, "embedding_service", EmbeddingService(provider=provider)
    )
    try:
        with TestClient(app) as client:
            uploaded = client.post(
                "/v1/documents",
                files={
                    "file": (
                        "kepu.md",
                        "光合作用发生在叶绿体中，需要光能。".encode(),
                        "text/markdown",
                    )
                },
            ).json()["data"]
            before = _stored_vectors(session)

            response = client.post("/v1/documents/reembed", json={})

            after = _stored_vectors(session)
    finally:
        app.dependency_overrides.clear()
        engine.dispose()

    assert response.status_code == 200
    body = response.json()["data"]
    assert body["status"] == "COMPLETED"
    assert body["reembedded"] == uploaded["chunks_count"] > 0
    assert body["dimension"] == 128
    assert set(after) == set(before)
    assert all(after[chunk_id] != before[chunk_id] for chunk_id in before)
    assert provider.batches


def test_the_endpoint_refuses_another_tenants_document_id(tmp_path, monkeypatch):
    engine, _ = _isolated_app(tmp_path, monkeypatch)
    app.dependency_overrides[get_principal] = lambda: Principal(tenant_id="alpha")
    try:
        with TestClient(app) as client:
            uploaded = client.post(
                "/v1/documents",
                files={"file": ("kepu.md", "光合作用发生在叶绿体中。".encode(), "text/markdown")},
            ).json()["data"]

            app.dependency_overrides[get_principal] = lambda: Principal(tenant_id="beta")
            response = client.post(
                "/v1/documents/reembed", json={"document_ids": [uploaded["document_id"]]}
            )
    finally:
        app.dependency_overrides.clear()
        engine.dispose()

    assert response.status_code == 404
