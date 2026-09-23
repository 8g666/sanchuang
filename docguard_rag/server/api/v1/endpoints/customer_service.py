from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from docguard_rag.server.core.security import Principal, get_principal
from docguard_rag.server.database.connection import get_db
from docguard_rag.server.schemas.common import ApiResponse
from docguard_rag.server.schemas.customer_service import (
    CustomerServiceRequest,
    CustomerServiceResponse,
)
from docguard_rag.server.services.customer_service import CustomerServiceService

router = APIRouter(prefix="/support", tags=["Customer Service"])
customer_service = CustomerServiceService()


@router.post("/chat", response_model=ApiResponse[CustomerServiceResponse])
def chat(
    payload: CustomerServiceRequest,
    request: Request,
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> ApiResponse[CustomerServiceResponse]:
    return ApiResponse(
        request_id=request.state.request_id,
        data=customer_service.chat(
            db,
            payload,
            tenant_id=principal.tenant_id,
            request_id=request.state.request_id,
        ),
    )


@router.delete("/sessions/{session_id}")
def reset_session(
    session_id: str,
    request: Request,
    principal: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> ApiResponse[dict]:
    # 200 even for an unknown session: callers treat this as fire-and-forget, and `found`
    # distinguishes "reset" from "never existed" without adding a failure mode.
    found = customer_service.reset(db, session_id, tenant_id=principal.tenant_id)
    return ApiResponse(
        request_id=request.state.request_id,
        data={"session_id": session_id, "status": "RESET", "found": found},
    )
