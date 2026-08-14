import json

from sqlalchemy.orm import Session

from docguard_rag.server.database.models import QueryLogTable


class QueryLogRepository:
    """Persist tenant-scoped query traces for auditing and diagnostics."""

    def create(
        self,
        db: Session,
        *,
        request_id: str,
        tenant_id: str,
        query_text: str,
        document_ids: list[str],
        retrieved_chunk_ids: list[str],
        answer_text: str | None,
        confidence: float,
        latency_ms: int,
        token_usage: dict | None = None,
    ) -> QueryLogTable:
        record = QueryLogTable(
            request_id=request_id,
            tenant_id=tenant_id,
            query_text=query_text,
            document_ids_json=json.dumps(document_ids, ensure_ascii=False),
            retrieved_chunk_ids_json=json.dumps(retrieved_chunk_ids, ensure_ascii=False),
            answer_text=answer_text,
            confidence=confidence,
            latency_ms=latency_ms,
            token_usage_json=(
                json.dumps(token_usage, ensure_ascii=False) if token_usage is not None else None
            ),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    def get(
        self,
        db: Session,
        request_id: str,
        tenant_id: str | None = None,
    ) -> QueryLogTable | None:
        query = db.query(QueryLogTable).filter(QueryLogTable.request_id == request_id)
        if tenant_id is not None:
            query = query.filter(QueryLogTable.tenant_id == tenant_id)
        return query.one_or_none()
