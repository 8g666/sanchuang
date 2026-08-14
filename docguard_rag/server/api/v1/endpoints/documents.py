import json
import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from docguard_rag.server.core.security import Principal, get_principal
from docguard_rag.server.database.connection import SessionLocal, get_db
from docguard_rag.server.repositories.chunk_repository import ChunkRepository
from docguard_rag.server.repositories.document_repository import DocumentRepository
from docguard_rag.server.repositories.job_repository import JobRepository
from docguard_rag.server.schemas.common import ApiResponse
from docguard_rag.server.schemas.document import (
    DocumentBlockItem,
    DocumentBlocksResult,
    DocumentChunkItem,
    DocumentChunksResult,
    DocumentDeleteResult,
    DocumentStatus,
    DocumentUploadResult,
)
from docguard_rag.server.services.document_lifecycle_service import DocumentLifecycleService
from docguard_rag.server.services.ingestion_service import IngestionError, IngestionService
from docguard_rag.server.services.task_dispatcher import dispatcher

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])

document_repository = DocumentRepository()
chunk_repository = ChunkRepository()
job_repository = JobRepository()
ingestion_service = IngestionService()
lifecycle_service = DocumentLifecycleService(ingestion_service)


def _parse_metadata(value: str | None) -> dict | None:
    if value is None:
        return None
    try:
        metadata = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="metadata must be valid JSON") from exc
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=422, detail="metadata must be a JSON object")
    return metadata


def _run_ingestion_job(job_id: str, arguments: dict) -> dict:
    with SessionLocal() as db:
        staged_path = arguments["staged_path"]
        staged_file = ingestion_service.storage.upload_dir.joinpath(staged_path)
        job = job_repository.get(db, job_id)
        if job is not None and job.status == "COMPLETED":
            ingestion_service.storage.delete(str(staged_file))
            return json.loads(job.result_json or "{}")

        document_id = arguments["document_id"]
        existing = document_repository.get_document(db, document_id)
        if existing is not None and existing.status == "READY":
            blocks_count = len(document_repository.get_blocks(db, document_id))
            chunks_count = chunk_repository.count_chunks(db, document_id)
            result = DocumentUploadResult(
                document_id=document_id,
                status="READY",
                blocks_count=blocks_count,
                chunks_count=chunks_count,
                vectors_count=chunks_count,
            )
            result_data = result.model_dump(mode="json")
            job_repository.update(
                db,
                job_id,
                status="COMPLETED",
                progress=100,
                current_stage="COMPLETED",
                result=result_data,
            )
            ingestion_service.storage.delete(str(staged_file))
            return result_data

        job_repository.update(
            db,
            job_id,
            status="PROCESSING",
            progress=10,
            current_stage="DOCUMENT_INGESTION",
        )
        try:
            ingestion_arguments = {
                key: value for key, value in arguments.items() if key != "staged_path"
            }
            result = ingestion_service.ingest_bytes(
                db,
                content=staged_file.read_bytes(),
                **ingestion_arguments,
            )
            result_data = result.model_dump(mode="json")
            job_repository.update(
                db,
                job_id,
                status="COMPLETED",
                progress=100,
                current_stage="COMPLETED",
                result=result_data,
            )
            ingestion_service.storage.delete(str(staged_file))
            return result_data
        except Exception as exc:
            db.rollback()
            logger.exception("Document ingestion job %s failed", job_id)
            job_repository.update(
                db,
                job_id,
                status="FAILED",
                progress=100,
                current_stage="FAILED",
                error=str(exc),
            )
            raise


@router.post(
    "",
    response_model=ApiResponse[DocumentUploadResult],
    summary="Upload document",
)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    collection_id: str | None = Form(default=None),
    enable_review: bool = Form(default=False),
    async_mode: bool = Form(default=False, alias="async"),
    metadata: str | None = Form(default=None),
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> ApiResponse[DocumentUploadResult]:
    filename = file.filename or "document"
    content = await file.read()
    arguments = {
        "filename": filename,
        "collection_id": collection_id,
        "enable_review": enable_review,
        "metadata": _parse_metadata(metadata),
        "tenant_id": principal.tenant_id,
        "owner_id": principal.owner_id,
    }

    if async_mode:
        document_id = f"doc_{uuid4().hex[:12]}"
        job_id = f"job_{uuid4().hex[:12]}"
        arguments["document_id"] = document_id
        staged_path = ingestion_service.storage.save_bytes(filename, content)
        arguments["staged_path"] = ingestion_service.storage.upload_dir.joinpath(staged_path).name
        job_repository.create(
            db,
            job_id=job_id,
            tenant_id=principal.tenant_id,
            job_type="DOCUMENT_INGESTION",
            payload={
                "document_id": document_id,
                "filename": filename,
                "collection_id": collection_id,
            },
        )
        try:
            dispatcher.submit(_run_ingestion_job, job_id, arguments)
        except Exception as exc:
            job_repository.update(
                db,
                job_id,
                status="FAILED",
                progress=100,
                current_stage="DISPATCH_FAILED",
                error=str(exc),
            )
            ingestion_service.storage.delete(staged_path)
            raise HTTPException(status_code=503, detail="Task queue unavailable") from exc
        result = DocumentUploadResult(
            document_id=document_id,
            job_id=job_id,
            status="PROCESSING",
            blocks_count=0,
            chunks_count=0,
            vectors_count=0,
        )
    else:
        arguments["content"] = content
        try:
            result = ingestion_service.ingest_bytes(db, **arguments)
        except IngestionError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return ApiResponse(request_id=request.state.request_id, data=result)


@router.put(
    "/{document_id}",
    response_model=ApiResponse[DocumentUploadResult],
    summary="Replace document content",
)
async def replace_document(
    document_id: str,
    request: Request,
    file: UploadFile = File(...),
    collection_id: str | None = Form(default=None),
    enable_review: bool = Form(default=False),
    metadata: str | None = Form(default=None),
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> ApiResponse[DocumentUploadResult]:
    try:
        result = lifecycle_service.replace_bytes(
            db,
            document_id,
            filename=file.filename or "document",
            content=await file.read(),
            collection_id=collection_id,
            enable_review=enable_review,
            metadata=_parse_metadata(metadata),
            tenant_id=principal.tenant_id,
            owner_id=principal.owner_id,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return ApiResponse(request_id=request.state.request_id, data=result)


@router.post(
    "/{document_id}/reindex",
    response_model=ApiResponse[DocumentUploadResult],
    summary="Rebuild document index",
)
def reindex_document(
    document_id: str,
    request: Request,
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> ApiResponse[DocumentUploadResult]:
    try:
        result = lifecycle_service.reindex(
            db,
            document_id,
            tenant_id=principal.tenant_id,
            owner_id=principal.owner_id,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return ApiResponse(request_id=request.state.request_id, data=result)


@router.delete(
    "/{document_id}",
    response_model=ApiResponse[DocumentDeleteResult],
    summary="Delete document",
)
def delete_document(
    document_id: str,
    request: Request,
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> ApiResponse[DocumentDeleteResult]:
    try:
        result = lifecycle_service.delete(db, document_id, tenant_id=principal.tenant_id)
    except IngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return ApiResponse(request_id=request.state.request_id, data=result)


@router.get(
    "/{document_id}",
    response_model=ApiResponse[DocumentStatus],
    summary="Get document status",
)
def get_document_status(
    document_id: str,
    request: Request,
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> ApiResponse[DocumentStatus]:
    document = document_repository.get_document(db, document_id, principal.tenant_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    blocks = document_repository.get_blocks(db, document_id)
    pages = [block.page_number for block in blocks if block.page_number is not None]
    return ApiResponse(
        request_id=request.state.request_id,
        data=DocumentStatus(
            document_id=document.document_id,
            filename=document.filename,
            file_type=document.file_type,
            status=document.status,
            page_count=max(pages, default=None),
            chunk_count=chunk_repository.count_chunks(db, document_id),
            sentence_count=sum(block.block_type == "sentence" for block in blocks),
        ),
    )


@router.get(
    "/{document_id}/blocks",
    response_model=ApiResponse[DocumentBlocksResult],
)
def get_document_blocks(
    document_id: str,
    request: Request,
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> ApiResponse[DocumentBlocksResult]:
    if document_repository.get_document(db, document_id, principal.tenant_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    blocks = document_repository.get_blocks(db, document_id)
    items = [
        DocumentBlockItem(
            block_id=block.block_id,
            block_type=block.block_type,
            text=block.text,
            page_number=block.page_number,
            section_path=json.loads(block.section_path or "[]"),
            paragraph_index=block.paragraph_index,
            sentence_index=block.sentence_index,
            start_offset=block.start_offset,
            end_offset=block.end_offset,
        )
        for block in blocks
    ]
    return ApiResponse(
        request_id=request.state.request_id,
        data=DocumentBlocksResult(
            document_id=document_id,
            blocks=items,
            blocks_count=len(items),
        ),
    )


@router.get(
    "/{document_id}/chunks",
    response_model=ApiResponse[DocumentChunksResult],
)
def get_document_chunks(
    document_id: str,
    request: Request,
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> ApiResponse[DocumentChunksResult]:
    if document_repository.get_document(db, document_id, principal.tenant_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = chunk_repository.get_chunks(db, document_id)
    return ApiResponse(
        request_id=request.state.request_id,
        data=DocumentChunksResult(
            chunks=[
                DocumentChunkItem(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    block_id=chunk.block_id,
                    text=chunk.text,
                    chunk_index=chunk.chunk_index,
                )
                for chunk in chunks
            ]
        ),
    )
