from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from docguard_rag import DocGuard, DocGuardError
from docguard_rag.server.database.models import (
    DocumentBlockTable,
    DocumentChunkTable,
    DocumentTable,
    DocumentVectorTable,
)


def test_local_document_replace_reindex_and_delete(tmp_path):
    database_path = tmp_path / "lifecycle.db"
    database_url = f"sqlite:///{database_path}"
    upload_dir = tmp_path / "uploads"
    original = tmp_path / "original.txt"
    replacement = tmp_path / "replacement.txt"
    original.write_text("Original lifecycle content.", encoding="utf-8")
    replacement.write_text("Replacement content supports reindexing.", encoding="utf-8")

    with DocGuard(database_url=database_url, upload_dir=upload_dir) as guard:
        uploaded = guard.documents.upload(original)
        document_id = uploaded.document_id

        updated = guard.documents.update(document_id, replacement, metadata={"revision": 2})
        assert updated.document_id == document_id
        assert "Replacement" in guard.documents.blocks(document_id)[0].text

        rebuilt = guard.documents.reindex(document_id)
        assert rebuilt.document_id == document_id
        assert rebuilt.chunks_count > 0

        stored_files = list(upload_dir.iterdir())
        assert len(stored_files) == 1
        assert Path(stored_files[0]).is_file()

        deleted = guard.documents.delete(document_id)
        assert deleted.status == "DELETED"
        with pytest.raises(DocGuardError, match="Document not found"):
            guard.documents.get(document_id)

    engine = create_engine(database_url)
    testing_session = sessionmaker(bind=engine)
    with testing_session() as db:
        assert db.query(DocumentTable).count() == 0
        assert db.query(DocumentBlockTable).count() == 0
        assert db.query(DocumentChunkTable).count() == 0
        assert db.query(DocumentVectorTable).count() == 0
    assert list(upload_dir.iterdir()) == []
    engine.dispose()


def test_document_lifecycle_enforces_tenant_ownership(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'tenants.db'}"
    source = tmp_path / "document.txt"
    source.write_text("Tenant private content.", encoding="utf-8")
    tenant_a = DocGuard(database_url=database_url, upload_dir=tmp_path / "uploads", tenant_id="a")
    tenant_b = DocGuard(database_url=database_url, upload_dir=tmp_path / "uploads", tenant_id="b")
    try:
        document = tenant_a.documents.upload(source)
        with pytest.raises(DocGuardError, match="Document not found"):
            tenant_b.documents.delete(document.document_id)
        assert tenant_a.documents.get(document.document_id).status == "READY"
    finally:
        tenant_a.close()
        tenant_b.close()
