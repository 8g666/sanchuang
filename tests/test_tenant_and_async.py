import time

import pytest

from docguard_rag import DocGuard, DocGuardError


def wait_for_job(guard: DocGuard, job_id: str):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = guard.jobs.get(job_id)
        if job.status in {"COMPLETED", "FAILED"}:
            return job
        time.sleep(0.01)
    raise AssertionError("Background job did not finish")


def test_local_tenant_isolation_and_async_ingestion(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'tenant.db'}"
    source = tmp_path / "document.txt"
    source.write_text("系统使用 /v1 接口。", encoding="utf-8")

    tenant_a = DocGuard(
        database_url=database_url,
        upload_dir=tmp_path / "uploads-a",
        tenant_id="tenant-a",
    )
    tenant_b = DocGuard(
        database_url=database_url,
        upload_dir=tmp_path / "uploads-b",
        tenant_id="tenant-b",
    )
    try:
        document = tenant_a.documents.upload(source, async_mode=True)
        job = wait_for_job(tenant_a, document.job_id)
        assert job.status == "COMPLETED"
        assert tenant_a.documents.get(document.document_id).status == "READY"

        with pytest.raises(DocGuardError, match="Document not found"):
            tenant_b.documents.get(document.document_id)
        result = tenant_b.query("接口", document_ids=[document.document_id])
        assert result.sources == []
    finally:
        tenant_a.close()
        tenant_b.close()
