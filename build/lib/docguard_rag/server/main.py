import logging
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from docguard_rag.server.api.v1.router import api_router
from docguard_rag.server.core.config import get_settings
from docguard_rag.server.core.provider_errors import ProviderError
from docguard_rag.server.database.connection import SessionLocal
from docguard_rag.server.database.init_db import init_database

API_PREFIX = "/v1"
logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _response(
    request: Request,
    *,
    status_code: int,
    message: str,
    data: Any = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "code": 0 if status_code < 400 else status_code,
            "message": message,
            "request_id": _request_id(request),
            "data": jsonable_encoder(data),
        },
    )


def create_application(*, initialize_database: bool = True) -> FastAPI:
    """Create a configured FastAPI application."""

    settings = get_settings()
    if initialize_database:
        init_database()

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        description="General-purpose document RAG and quality review service",
        redirect_slashes=False,
    )
    application.include_router(api_router, prefix=API_PREFIX)

    @application.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        return _response(
            request,
            status_code=exc.status_code,
            message=str(exc.detail),
        )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return _response(
            request,
            status_code=422,
            message="Request validation failed",
            data={"errors": exc.errors()},
        )

    @application.exception_handler(ProviderError)
    async def provider_exception_handler(
        request: Request,
        exc: ProviderError,
    ) -> JSONResponse:
        return _response(request, status_code=502, message=str(exc))

    @application.middleware("http")
    async def add_request_id(request: Request, call_next):
        request.state.request_id = f"req_{uuid4().hex}"
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @application.get("/health", tags=["System"], summary="Service liveness check")
    def health_check(request: Request) -> JSONResponse:
        return _response(
            request,
            status_code=200,
            message="success",
            data={"status": "healthy"},
        )

    @application.get("/ready", tags=["System"], summary="Service readiness check")
    def readiness_check(request: Request) -> JSONResponse:
        try:
            with SessionLocal() as db:
                db.execute(text("SELECT 1"))
        except SQLAlchemyError:
            logger.exception("Database readiness check failed")
            return _response(
                request,
                status_code=503,
                message="database unavailable",
                data={"status": "not_ready"},
            )
        return _response(
            request,
            status_code=200,
            message="success",
            data={"status": "ready"},
        )

    return application


app = create_application()
