from fastapi import APIRouter, Depends, Request

from docguard_rag.server.core.security import Principal, get_principal
from docguard_rag.server.schemas.common import ApiResponse
from docguard_rag.server.schemas.correction import CorrectionRequest, CorrectionResult
from docguard_rag.server.services.correction_service import CorrectionService

router = APIRouter(prefix="/correct", tags=["Corrections"])
service = CorrectionService()


@router.post("", response_model=ApiResponse[CorrectionResult])
def correct_text(
    payload: CorrectionRequest,
    request: Request,
    _principal: Principal = Depends(get_principal),
) -> ApiResponse[CorrectionResult]:
    return ApiResponse(
        request_id=request.state.request_id,
        data=service.correct(payload),
    )
