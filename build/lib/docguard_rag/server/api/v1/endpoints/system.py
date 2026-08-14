from fastapi import APIRouter, Request
from pydantic import BaseModel

from docguard_rag.server.core.config import get_settings
from docguard_rag.server.schemas.common import ApiResponse

router = APIRouter(prefix="/system", tags=["System"])


class SystemInfo(BaseModel):
    app_name: str
    app_version: str
    environment: str


@router.get(
    "/info",
    response_model=ApiResponse[SystemInfo],
    summary="Get system information",
)
def get_system_info(request: Request) -> ApiResponse[SystemInfo]:
    settings = get_settings()

    return ApiResponse(
        request_id=request.state.request_id,
        data=SystemInfo(
            app_name=settings.app_name,
            app_version=settings.app_version,
            environment=settings.environment,
        ),
    )
