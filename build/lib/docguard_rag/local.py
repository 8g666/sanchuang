import json
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from docguard_rag.errors import DocGuardError
from docguard_rag.server.database.connection import SessionLocal
from docguard_rag.server.database.init_db import init_database
from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.models.document_chunk import DocumentChunk
from docguard_rag.server.repositories.chunk_repository import ChunkRepository
from docguard_rag.server.repositories.document_repository import DocumentRepository
from docguard_rag.server.repositories.job_repository import JobRepository
from docguard_rag.server.repositories.review_repository import ReviewRepository
from docguard_rag.server.retrievers.hybrid_retriever import HybridRetriever
from docguard_rag.server.retrievers.lexical_retriever import LexicalRetriever
from docguard_rag.server.retrievers.vector_retriever import VectorRetriever
from docguard_rag.server.schemas.correction import CorrectionRequest
from docguard_rag.server.schemas.query import QueryRequest
from docguard_rag.server.schemas.review import ReviewRequest
from docguard_rag.server.services.correction_service import CorrectionService
from docguard_rag.server.services.document_lifecycle_service import DocumentLifecycleService
from docguard_rag.server.services.index_service import IndexService
from docguard_rag.server.services.ingestion_service import IngestionError, IngestionService
from docguard_rag.server.services.query_service import QueryService
from docguard_rag.server.services.retrieval_service import RetrievalService
from docguard_rag.server.services.review_service import ReviewService
from docguard_rag.server.services.task_dispatcher import InProcessTaskDispatcher
from docguard_rag.server.services.vector_service import VectorService
from docguard_rag.server.storage.file_storage import FileStorage
from docguard_rag.types import (
    CorrectionResult,
    DocumentDeleteResult,
    DocumentResult,
    DocumentStatus,
    Issue,
    IssuePage,
    JobStatus,
    QueryResult,
    ReviewResult,
)


class LocalDocuments:
    def __init__(self, guard: "DocGuard"):
        self._guard = guard

    def upload(
        self,
        file_path: str | Path,
        *,
        collection_id: str | None = None,
        enable_review: bool = False,
        metadata: dict | None = None,
        async_mode: bool = False,
    ) -> DocumentResult:
        path = Path(file_path)
        if async_mode:
            if not path.is_file():
                raise FileNotFoundError(path)
            document_id = f"doc_{uuid4().hex[:12]}"
            job_id = f"job_{uuid4().hex[:12]}"
            arguments = {
                "filename": path.name,
                "content": path.read_bytes(),
                "collection_id": collection_id,
                "enable_review": enable_review,
                "metadata": metadata,
                "tenant_id": self._guard.tenant_id,
                "owner_id": self._guard.owner_id,
                "document_id": document_id,
            }
            with self._guard._session() as db:
                self._guard._job_repository.create(
                    db,
                    job_id=job_id,
                    tenant_id=self._guard.tenant_id,
                    job_type="DOCUMENT_INGESTION",
                    payload={"document_id": document_id, "filename": path.name},
                )
            self._guard._dispatcher.submit(self._guard._run_ingestion_job, job_id, arguments)
            return DocumentResult(
                document_id=document_id,
                job_id=job_id,
                status="PROCESSING",
            )
        with self._guard._session() as db:
            result = self._guard._ingestion.ingest_file(
                db,
                file_path,
                collection_id=collection_id,
                enable_review=enable_review,
                metadata=metadata,
                tenant_id=self._guard.tenant_id,
                owner_id=self._guard.owner_id,
            )
        return DocumentResult.model_validate(result.model_dump())

    def get(self, document_id: str) -> DocumentStatus:
        with self._guard._session() as db:
            document = self._guard._documents.get_document(db, document_id, self._guard.tenant_id)
            if document is None:
                raise DocGuardError("Document not found")
            blocks = self._guard._documents.get_blocks(db, document_id)
            pages = [block.page_number for block in blocks if block.page_number is not None]
            return DocumentStatus(
                document_id=document.document_id,
                filename=document.filename,
                file_type=document.file_type,
                status=document.status,
                page_count=max(pages) if pages else None,
                chunk_count=self._guard._chunks.count_chunks(db, document_id),
                sentence_count=sum(block.block_type == "sentence" for block in blocks),
            )

    def update(
        self,
        document_id: str,
        file_path: str | Path,
        *,
        collection_id: str | None = None,
        enable_review: bool = False,
        metadata: dict | None = None,
    ) -> DocumentResult:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with self._guard._session() as db:
            try:
                result = self._guard._lifecycle.replace_bytes(
                    db,
                    document_id,
                    filename=path.name,
                    content=path.read_bytes(),
                    collection_id=collection_id,
                    enable_review=enable_review,
                    metadata=metadata,
                    tenant_id=self._guard.tenant_id,
                    owner_id=self._guard.owner_id,
                )
            except IngestionError as exc:
                raise DocGuardError(str(exc)) from exc
        return DocumentResult.model_validate(result.model_dump())

    def reindex(self, document_id: str) -> DocumentResult:
        with self._guard._session() as db:
            try:
                result = self._guard._lifecycle.reindex(
                    db,
                    document_id,
                    tenant_id=self._guard.tenant_id,
                    owner_id=self._guard.owner_id,
                )
            except IngestionError as exc:
                raise DocGuardError(str(exc)) from exc
        return DocumentResult.model_validate(result.model_dump())

    def delete(self, document_id: str) -> DocumentDeleteResult:
        with self._guard._session() as db:
            try:
                result = self._guard._lifecycle.delete(
                    db,
                    document_id,
                    tenant_id=self._guard.tenant_id,
                )
            except IngestionError as exc:
                raise DocGuardError(str(exc)) from exc
        return DocumentDeleteResult.model_validate(result.model_dump())

    def blocks(self, document_id: str) -> list[DocumentBlock]:
        with self._guard._session() as db:
            if self._guard._documents.get_document(db, document_id, self._guard.tenant_id) is None:
                raise DocGuardError("Document not found")
            records = self._guard._documents.get_blocks(db, document_id)
            return [
                DocumentBlock(
                    block_id=record.block_id,
                    document_id=record.document_id,
                    block_type=record.block_type,
                    text=record.text,
                    page_number=record.page_number,
                    section_path=json.loads(record.section_path or "[]"),
                    paragraph_index=record.paragraph_index,
                    sentence_index=record.sentence_index,
                    start_offset=record.start_offset,
                    end_offset=record.end_offset,
                    metadata=json.loads(record.metadata_json or "{}"),
                )
                for record in records
            ]

    def chunks(self, document_id: str) -> list[DocumentChunk]:
        with self._guard._session() as db:
            if self._guard._documents.get_document(db, document_id, self._guard.tenant_id) is None:
                raise DocGuardError("Document not found")
            records = self._guard._chunks.get_chunks(db, document_id)
            return [
                DocumentChunk(
                    chunk_id=record.chunk_id,
                    document_id=record.document_id,
                    block_id=record.block_id,
                    block_ids=[record.block_id],
                    text=record.text,
                    char_count=record.char_count,
                    token_count=record.token_count,
                    chunk_index=record.chunk_index,
                    split_method=record.split_method,
                )
                for record in records
            ]


class LocalReviews:
    def __init__(self, guard: "DocGuard"):
        self._guard = guard

    def create(
        self,
        *,
        document_ids: list[str],
        check_types: list[str] | None = None,
        scope: str = "FULL_DOCUMENT",
        mode: str = "BALANCED",
        async_mode: bool = False,
    ) -> ReviewResult:
        payload = ReviewRequest.model_validate(
            {
                "document_ids": document_ids,
                "check_types": check_types or ["TYPO", "SEMANTIC_UNCLEAR"],
                "scope": scope,
                "mode": mode,
                "async": async_mode,
            }
        )
        with self._guard._session() as db:
            missing = [
                document_id
                for document_id in payload.document_ids
                if self._guard._documents.get_document(db, document_id, self._guard.tenant_id)
                is None
            ]
            if missing:
                raise DocGuardError(f"Documents not found: {', '.join(missing)}")
            unsupported = [
                value
                for value in payload.check_types
                if value.upper() not in self._guard._review.supported_types
            ]
            if unsupported:
                raise DocGuardError(f"Unsupported review types: {', '.join(unsupported)}")
            if payload.async_mode:
                review_id, normalized_types = self._guard._review.prepare_review(
                    db,
                    payload.document_ids,
                    payload.check_types,
                    payload.mode,
                    tenant_id=self._guard.tenant_id,
                )
                self._guard._job_repository.create(
                    db,
                    job_id=review_id,
                    tenant_id=self._guard.tenant_id,
                    job_type="DOCUMENT_REVIEW",
                    payload={"document_ids": payload.document_ids},
                )
                self._guard._dispatcher.submit(
                    self._guard._run_review_job,
                    review_id,
                    payload.document_ids,
                    normalized_types,
                    payload.mode,
                )
                return ReviewResult(
                    review_id=review_id,
                    job_id=review_id,
                    status="PROCESSING",
                    issue_count=0,
                )
            review_id, issues = self._guard._review.create_review(
                db,
                payload.document_ids,
                payload.check_types,
                payload.mode,
                tenant_id=self._guard.tenant_id,
            )
        return ReviewResult(
            review_id=review_id,
            job_id=review_id,
            status="COMPLETED",
            issue_count=len(issues),
        )

    def list_issues(
        self,
        review_id: str,
        *,
        issue_type: str | None = None,
        min_confidence: float = 0.0,
        page: int | None = None,
        status: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> IssuePage:
        with self._guard._session() as db:
            if (
                self._guard._review_repository.get_review(db, review_id, self._guard.tenant_id)
                is None
            ):
                raise DocGuardError("Review not found")
            total, records = self._guard._review_repository.list_issues(
                db,
                review_id,
                issue_type,
                min_confidence,
                page,
                status,
                offset,
                limit,
            )
            return IssuePage(
                total=total,
                items=[_issue_from_record(record) for record in records],
            )


class LocalIssues:
    def __init__(self, guard: "DocGuard"):
        self._guard = guard

    def update(
        self,
        issue_id: str,
        *,
        status: str,
        comment: str | None = None,
    ) -> Issue:
        allowed = {"PENDING", "ACCEPTED", "REJECTED", "IGNORED", "FIXED"}
        if status not in allowed:
            raise DocGuardError(f"Unsupported issue status: {status}")
        with self._guard._session() as db:
            existing = self._guard._review_repository.get_issue(db, issue_id)
            if (
                existing is None
                or self._guard._documents.get_document(
                    db, existing.document_id, self._guard.tenant_id
                )
                is None
            ):
                raise DocGuardError("Issue not found")
            record = self._guard._review_repository.update_issue(db, issue_id, status, comment)
            if record is None:
                raise DocGuardError("Issue not found")
            return _issue_from_record(record)


class LocalJobs:
    def __init__(self, guard: "DocGuard"):
        self._guard = guard

    def get(self, job_id: str) -> JobStatus:
        with self._guard._session() as db:
            task = self._guard._job_repository.get(db, job_id, self._guard.tenant_id)
            if task is not None:
                return JobStatus(
                    job_id=task.job_id,
                    status=task.status,
                    progress=task.progress,
                    current_stage=task.current_stage,
                    error=task.error_message,
                )
            job = self._guard._review_repository.get_review(db, job_id)
            if job is not None and job.tenant_id != self._guard.tenant_id:
                job = None
            if job is None:
                raise DocGuardError("Job not found")
            return JobStatus(
                job_id=job.review_id,
                status=job.status,
                progress=job.progress,
                current_stage=("COMPLETED" if job.status == "COMPLETED" else "DOCUMENT_REVIEW"),
                error=job.error_message,
            )


class DocGuard:
    """In-process facade exposing the same resources as the HTTP client."""

    def __init__(
        self,
        *,
        database_url: str | None = None,
        upload_dir: str | Path | None = None,
        tenant_id: str = "default",
        owner_id: str | None = None,
    ):
        self.tenant_id = tenant_id
        self.owner_id = owner_id
        if database_url is None:
            self._session_factory = SessionLocal
            init_database()
        else:
            connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
            self._engine = create_engine(database_url, connect_args=connect_args)
            self._session_factory = sessionmaker(
                bind=self._engine,
                autoflush=False,
                autocommit=False,
            )
            init_database(self._engine)
        storage = FileStorage(upload_dir) if upload_dir is not None else FileStorage()
        self._vector_service = VectorService(session_factory=self._session_factory)
        self._ingestion = IngestionService(
            storage=storage,
            index_service=IndexService(vector_service=self._vector_service),
        )
        self._lifecycle = DocumentLifecycleService(self._ingestion)
        lexical = LexicalRetriever(session_factory=self._session_factory)
        dense = VectorRetriever(vector_service=self._vector_service)
        retriever = HybridRetriever(lexical=lexical, dense=dense)
        self._query = QueryService(RetrievalService(retriever=retriever))
        self._review = ReviewService()
        self._documents = DocumentRepository()
        self._chunks = ChunkRepository()
        self._review_repository = ReviewRepository()
        self._job_repository = JobRepository()
        self._dispatcher = InProcessTaskDispatcher()
        self._correction = CorrectionService()
        self.documents = LocalDocuments(self)
        self.reviews = LocalReviews(self)
        self.issues = LocalIssues(self)
        self.jobs = LocalJobs(self)

    @contextmanager
    def _session(self):
        db = self._session_factory()
        try:
            yield db
        finally:
            db.close()

    def query(
        self,
        query: str,
        *,
        document_ids: list[str] | None = None,
        collection_id: str | None = None,
        top_k: int = 5,
        generate_answer: bool = True,
        include_sources: bool = True,
        filters: dict | None = None,
        corrective: bool | None = None,
        max_retries: int | None = None,
    ) -> QueryResult:
        payload = QueryRequest(
            query=query,
            document_ids=document_ids or [],
            collection_id=collection_id,
            top_k=top_k,
            generate_answer=generate_answer,
            include_sources=include_sources,
            filters=filters or {},
            corrective=corrective,
            max_retries=max_retries,
        )
        with self._session() as db:
            result = self._query.query(
                db,
                payload,
                tenant_id=self.tenant_id,
                request_id=f"local_{uuid4().hex}",
            )
        return QueryResult.model_validate(result.model_dump())

    def correct(
        self,
        text: str,
        *,
        check_types: list[str] | None = None,
        mode: str = "BALANCED",
        apply_changes: bool = True,
    ) -> CorrectionResult:
        result = self._correction.correct(
            CorrectionRequest(
                text=text,
                check_types=check_types or ["TYPO", "GRAMMAR", "SEMANTIC_UNCLEAR"],
                mode=mode,
                apply_changes=apply_changes,
            )
        )
        return CorrectionResult.model_validate(result.model_dump())

    def close(self) -> None:
        """Provided for symmetry with DocGuardClient; local sessions are per-call."""

        self._dispatcher.shutdown(wait=True)
        self._vector_service.close()
        engine = getattr(self, "_engine", None)
        if engine is not None:
            engine.dispose()

    def _run_review_job(
        self,
        review_id: str,
        document_ids: list[str],
        check_types: list[str],
        mode: str,
    ) -> None:
        with self._session() as db:
            self._job_repository.update(
                db,
                review_id,
                status="PROCESSING",
                progress=10,
                current_stage="DOCUMENT_REVIEW",
            )
            try:
                issues = self._review.run_review(db, review_id, document_ids, check_types, mode)
                self._job_repository.update(
                    db,
                    review_id,
                    status="COMPLETED",
                    progress=100,
                    current_stage="COMPLETED",
                    result={"issue_count": len(issues)},
                )
            except Exception as exc:
                db.rollback()
                self._review_repository.fail_review(db, review_id, str(exc))
                self._job_repository.update(
                    db,
                    review_id,
                    status="FAILED",
                    progress=100,
                    current_stage="FAILED",
                    error=str(exc),
                )

    def _run_ingestion_job(self, job_id: str, arguments: dict) -> None:
        with self._session() as db:
            self._job_repository.update(
                db,
                job_id,
                status="PROCESSING",
                progress=10,
                current_stage="DOCUMENT_INGESTION",
            )
            try:
                result = self._ingestion.ingest_bytes(db, **arguments)
                self._job_repository.update(
                    db,
                    job_id,
                    status="COMPLETED",
                    progress=100,
                    current_stage="COMPLETED",
                    result=result.model_dump(mode="json"),
                )
            except Exception as exc:
                db.rollback()
                self._job_repository.update(
                    db,
                    job_id,
                    status="FAILED",
                    progress=100,
                    current_stage="FAILED",
                    error=str(exc),
                )

    def __enter__(self) -> "DocGuard":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _issue_from_record(record) -> Issue:
    return Issue.model_validate(
        {
            "issue_id": record.issue_id,
            "document_id": record.document_id,
            "issue_type": record.issue_type,
            "severity": record.severity,
            "original_text": record.original_text,
            "problem_text": record.problem_text,
            "suggestion": record.suggestion,
            "reason": record.reason,
            "confidence": record.confidence,
            "location": {
                "page": record.page_number,
                "section_path": json.loads(record.section_path or "[]"),
                "paragraph_index": record.paragraph_index,
                "sentence_index": record.sentence_index,
                "start_offset": record.start_offset,
                "end_offset": record.end_offset,
            },
            "status": record.status,
            "comment": record.comment,
        }
    )
