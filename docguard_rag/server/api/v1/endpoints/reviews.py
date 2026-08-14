import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from docguard_rag.server.core.security import Principal, get_principal
from docguard_rag.server.database.connection import SessionLocal, get_db
from docguard_rag.server.repositories.document_repository import DocumentRepository
from docguard_rag.server.repositories.job_repository import JobRepository
from docguard_rag.server.repositories.review_repository import ReviewRepository
from docguard_rag.server.schemas.common import ApiResponse
from docguard_rag.server.schemas.review import (
    IssueLocation,
    IssuePageResult,
    IssueResult,
    IssueStatusUpdate,
    ReviewRequest,
    ReviewResult,
)
from docguard_rag.server.services.review_service import ReviewService
from docguard_rag.server.services.task_dispatcher import dispatcher

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Reviews"])

review_service = ReviewService()
review_repository = ReviewRepository()
document_repository = DocumentRepository()
job_repository = JobRepository()


def _run_review_job(
    review_id: str,
    document_ids: list[str],
    check_types: list[str],
    mode: str,
) -> dict:
    with SessionLocal() as db:
        task = job_repository.get(db, review_id)
        if task is not None and task.status == "COMPLETED":
            return json.loads(task.result_json or "{}")
        review = review_repository.get_review(db, review_id)
        if review is not None and review.status == "COMPLETED":
            issue_count, _ = review_repository.list_issues(db, review_id, limit=1)
            result = {"review_id": review_id, "issue_count": issue_count}
            job_repository.update(
                db,
                review_id,
                status="COMPLETED",
                progress=100,
                current_stage="COMPLETED",
                result=result,
            )
            return result

        job_repository.update(
            db,
            review_id,
            status="PROCESSING",
            progress=10,
            current_stage="DOCUMENT_REVIEW",
        )
        try:
            issues = review_service.run_review(
                db,
                review_id,
                document_ids,
                check_types,
                mode,
            )
            result = {"review_id": review_id, "issue_count": len(issues)}
            job_repository.update(
                db,
                review_id,
                status="COMPLETED",
                progress=100,
                current_stage="COMPLETED",
                result=result,
            )
            return result
        except Exception as exc:
            db.rollback()
            logger.exception("Document review job %s failed", review_id)
            review_repository.fail_review(db, review_id, str(exc))
            job_repository.update(
                db,
                review_id,
                status="FAILED",
                progress=100,
                current_stage="FAILED",
                error=str(exc),
            )
            raise


@router.post("/reviews", response_model=ApiResponse[ReviewResult])
def create_review(
    payload: ReviewRequest,
    request: Request,
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> ApiResponse[ReviewResult]:
    missing = [
        document_id
        for document_id in payload.document_ids
        if document_repository.get_document(db, document_id, principal.tenant_id) is None
    ]
    if missing:
        raise HTTPException(status_code=404, detail=f"Documents not found: {', '.join(missing)}")

    unsupported = [
        issue_type
        for issue_type in payload.check_types
        if issue_type.upper() not in review_service.supported_types
    ]
    if unsupported:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported review types: {', '.join(unsupported)}",
        )

    if payload.async_mode:
        review_id, normalized_types = review_service.prepare_review(
            db,
            payload.document_ids,
            payload.check_types,
            payload.mode,
            tenant_id=principal.tenant_id,
        )
        job_repository.create(
            db,
            job_id=review_id,
            tenant_id=principal.tenant_id,
            job_type="DOCUMENT_REVIEW",
            payload={
                "review_id": review_id,
                "document_ids": payload.document_ids,
                "check_types": normalized_types,
                "mode": payload.mode,
            },
        )
        try:
            dispatcher.submit(
                _run_review_job,
                review_id,
                payload.document_ids,
                normalized_types,
                payload.mode,
            )
        except Exception as exc:
            review_repository.fail_review(db, review_id, str(exc))
            job_repository.update(
                db,
                review_id,
                status="FAILED",
                progress=100,
                current_stage="DISPATCH_FAILED",
                error=str(exc),
            )
            raise HTTPException(status_code=503, detail="Task queue unavailable") from exc
        result = ReviewResult(
            review_id=review_id,
            job_id=review_id,
            status="PROCESSING",
            issue_count=0,
        )
    else:
        review_id, issues = review_service.create_review(
            db,
            payload.document_ids,
            payload.check_types,
            payload.mode,
            tenant_id=principal.tenant_id,
        )
        result = ReviewResult(
            review_id=review_id,
            job_id=review_id,
            status="COMPLETED",
            issue_count=len(issues),
        )

    return ApiResponse(request_id=request.state.request_id, data=result)


@router.get(
    "/reviews/{review_id}/issues",
    response_model=ApiResponse[IssuePageResult],
)
def list_issues(
    review_id: str,
    request: Request,
    issue_type: str | None = None,
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    page: int | None = Query(default=None, ge=1),
    status: str | None = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> ApiResponse[IssuePageResult]:
    if review_repository.get_review(db, review_id, principal.tenant_id) is None:
        raise HTTPException(status_code=404, detail="Review not found")
    total, records = review_repository.list_issues(
        db,
        review_id,
        issue_type,
        min_confidence,
        page,
        status,
        offset,
        limit,
    )
    return ApiResponse(
        request_id=request.state.request_id,
        data=IssuePageResult(
            total=total,
            items=[_serialize_issue(record) for record in records],
        ),
    )


@router.patch("/issues/{issue_id}", response_model=ApiResponse[IssueResult])
def update_issue(
    issue_id: str,
    payload: IssueStatusUpdate,
    request: Request,
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> ApiResponse[IssueResult]:
    existing = review_repository.get_issue(db, issue_id)
    if (
        existing is None
        or document_repository.get_document(
            db,
            existing.document_id,
            principal.tenant_id,
        )
        is None
    ):
        raise HTTPException(status_code=404, detail="Issue not found")
    issue = review_repository.update_issue(db, issue_id, payload.status, payload.comment)
    if issue is None:
        raise HTTPException(status_code=404, detail="Issue not found")
    return ApiResponse(
        request_id=request.state.request_id,
        data=_serialize_issue(issue),
    )


def _serialize_issue(record) -> IssueResult:
    return IssueResult(
        issue_id=record.issue_id,
        document_id=record.document_id,
        issue_type=record.issue_type,
        severity=record.severity,
        original_text=record.original_text,
        problem_text=record.problem_text,
        suggestion=record.suggestion,
        reason=record.reason,
        confidence=record.confidence,
        location=IssueLocation(
            page=record.page_number,
            section_path=json.loads(record.section_path or "[]"),
            paragraph_index=record.paragraph_index,
            sentence_index=record.sentence_index,
            start_offset=record.start_offset,
            end_offset=record.end_offset,
        ),
        status=record.status,
        comment=record.comment,
    )
