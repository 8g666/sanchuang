import json
from datetime import datetime

from sqlalchemy.orm import Session

from docguard_rag.server.database.models import JobTable


class JobRepository:
    def create(
        self,
        db: Session,
        *,
        job_id: str,
        tenant_id: str,
        job_type: str,
        payload: dict | None = None,
    ) -> JobTable:
        record = JobTable(
            job_id=job_id,
            tenant_id=tenant_id,
            job_type=job_type,
            status="PENDING",
            progress=0,
            current_stage="QUEUED",
            payload_json=json.dumps(payload or {}, ensure_ascii=False),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    def get(
        self,
        db: Session,
        job_id: str,
        tenant_id: str | None = None,
    ) -> JobTable | None:
        query = db.query(JobTable).filter(JobTable.job_id == job_id)
        if tenant_id is not None:
            query = query.filter(JobTable.tenant_id == tenant_id)
        return query.one_or_none()

    def update(
        self,
        db: Session,
        job_id: str,
        *,
        status: str,
        progress: int,
        current_stage: str,
        result: dict | None = None,
        error: str | None = None,
    ) -> JobTable | None:
        record = self.get(db, job_id)
        if record is None:
            return None
        record.status = status
        record.progress = progress
        record.current_stage = current_stage
        record.result_json = json.dumps(result, ensure_ascii=False) if result is not None else None
        record.error_message = error
        record.updated_at = datetime.now()
        if status in {"COMPLETED", "FAILED", "CANCELLED"}:
            record.completed_at = datetime.now()
        db.commit()
        db.refresh(record)
        return record
