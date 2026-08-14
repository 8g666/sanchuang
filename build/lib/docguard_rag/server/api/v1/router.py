from fastapi import APIRouter

from docguard_rag.server.api.v1.endpoints import (
    corrections,
    documents,
    jobs,
    query,
    reviews,
    system,
)

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(documents.router)
api_router.include_router(query.router)
api_router.include_router(reviews.router)
api_router.include_router(jobs.router)
api_router.include_router(corrections.router)
