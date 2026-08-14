from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from docguard_rag.server.core.security import Principal, get_principal
from docguard_rag.server.database.connection import get_db
from docguard_rag.server.repositories.job_repository import JobRepository
from docguard_rag.server.repositories.review_repository import ReviewRepository
from docguard_rag.server.schemas.common import ApiResponse
from docguard_rag.server.schemas.job import JobStatus

router = APIRouter(prefix="/jobs", tags=["Jobs"])
repository = ReviewRepository()
job_repository = JobRepository()


@router.get("/{job_id}", response_model=ApiResponse[JobStatus])
def get_job_status(
    job_id: str,
    request: Request,
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
):
    task = job_repository.get(db, job_id, principal.tenant_id)
    if task is not None:
        return ApiResponse(
            request_id=request.state.request_id,
            data=JobStatus(
                job_id=task.job_id,
                status=task.status,
                progress=task.progress,
                current_stage=task.current_stage,
                error=task.error_message,
            ),
        )
    job = repository.get_review(db, job_id, principal.tenant_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    stage = "COMPLETED" if job.status == "COMPLETED" else "DOCUMENT_REVIEW"
    return ApiResponse(
        request_id=request.state.request_id,
        data=JobStatus(
            job_id=job.review_id,
            status=job.status,
            progress=job.progress,
            current_stage=stage,
            error=job.error_message,
        ),
    )
