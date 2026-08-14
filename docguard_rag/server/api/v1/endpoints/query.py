from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from docguard_rag.server.core.security import Principal, get_principal
from docguard_rag.server.database.connection import get_db
from docguard_rag.server.repositories.document_repository import DocumentRepository
from docguard_rag.server.schemas.common import ApiResponse
from docguard_rag.server.schemas.query import QueryRequest, QueryResult
from docguard_rag.server.services.query_service import QueryService

router = APIRouter(prefix="/query", tags=["Query"])


query_service = QueryService()
document_repository = DocumentRepository()


@router.post("", response_model=ApiResponse[QueryResult])
def query_document(
    payload: QueryRequest,
    request: Request,
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> ApiResponse[QueryResult]:
    accessible = document_repository.get_documents(
        db, payload.document_ids, tenant_id=principal.tenant_id
    )
    if payload.document_ids and len(accessible) != len(set(payload.document_ids)):
        raise HTTPException(status_code=404, detail="One or more documents were not found")
    return ApiResponse(
        request_id=request.state.request_id,
        data=query_service.query(
            db,
            payload,
            tenant_id=principal.tenant_id,
            request_id=request.state.request_id,
        ),
    )
