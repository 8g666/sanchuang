import json
import mimetypes
from pathlib import Path
from typing import Any

import httpx

from docguard_rag.errors import DocGuardAPIError
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


class DocumentsClient:
    def __init__(self, client: "DocGuardClient"):
        self._client = client

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
        if not path.is_file():
            raise FileNotFoundError(path)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        form = {
            "enable_review": str(enable_review).lower(),
            "async": str(async_mode).lower(),
        }
        if collection_id is not None:
            form["collection_id"] = collection_id
        if metadata is not None:
            form["metadata"] = json.dumps(metadata, ensure_ascii=False)
        with path.open("rb") as file:
            data = self._client._request(
                "POST",
                "/documents",
                data=form,
                files={"file": (path.name, file, content_type)},
            )
        return DocumentResult.model_validate(data)

    def get(self, document_id: str) -> DocumentStatus:
        data = self._client._request("GET", f"/documents/{document_id}")
        return DocumentStatus.model_validate(data)

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
        form = {"enable_review": str(enable_review).lower()}
        if collection_id is not None:
            form["collection_id"] = collection_id
        if metadata is not None:
            form["metadata"] = json.dumps(metadata, ensure_ascii=False)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with path.open("rb") as file:
            data = self._client._request(
                "PUT",
                f"/documents/{document_id}",
                data=form,
                files={"file": (path.name, file, content_type)},
            )
        return DocumentResult.model_validate(data)

    def reindex(self, document_id: str) -> DocumentResult:
        data = self._client._request("POST", f"/documents/{document_id}/reindex")
        return DocumentResult.model_validate(data)

    def delete(self, document_id: str) -> DocumentDeleteResult:
        data = self._client._request("DELETE", f"/documents/{document_id}")
        return DocumentDeleteResult.model_validate(data)

    def blocks(self, document_id: str) -> list[dict]:
        data = self._client._request("GET", f"/documents/{document_id}/blocks")
        return data["blocks"]

    def chunks(self, document_id: str) -> list[dict]:
        data = self._client._request("GET", f"/documents/{document_id}/chunks")
        return data["chunks"]


class ReviewsClient:
    def __init__(self, client: "DocGuardClient"):
        self._client = client

    def create(
        self,
        *,
        document_ids: list[str],
        check_types: list[str] | None = None,
        scope: str = "FULL_DOCUMENT",
        mode: str = "BALANCED",
        async_mode: bool = False,
    ) -> ReviewResult:
        data = self._client._request(
            "POST",
            "/reviews",
            json={
                "document_ids": document_ids,
                "check_types": check_types or ["TYPO", "SEMANTIC_UNCLEAR"],
                "scope": scope,
                "mode": mode,
                "async": async_mode,
            },
        )
        return ReviewResult.model_validate(data)

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
        params = {
            "issue_type": issue_type,
            "min_confidence": min_confidence,
            "page": page,
            "status": status,
            "offset": offset,
            "limit": limit,
        }
        data = self._client._request(
            "GET",
            f"/reviews/{review_id}/issues",
            params={key: value for key, value in params.items() if value is not None},
        )
        return IssuePage.model_validate(data)


class IssuesClient:
    def __init__(self, client: "DocGuardClient"):
        self._client = client

    def update(
        self,
        issue_id: str,
        *,
        status: str,
        comment: str | None = None,
    ) -> Issue:
        data = self._client._request(
            "PATCH",
            f"/issues/{issue_id}",
            json={"status": status, "comment": comment},
        )
        return Issue.model_validate(data)


class JobsClient:
    def __init__(self, client: "DocGuardClient"):
        self._client = client

    def get(self, job_id: str) -> JobStatus:
        data = self._client._request("GET", f"/jobs/{job_id}")
        return JobStatus.model_validate(data)


class DocGuardClient:
    """Synchronous Python client for a running DocGuard REST service."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        api_prefix: str = "/v1",
        timeout: float = 30.0,
        http_client: httpx.Client | None = None,
    ):
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.api_prefix = "/" + api_prefix.strip("/")
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
        )
        self.documents = DocumentsClient(self)
        self.reviews = ReviewsClient(self)
        self.issues = IssuesClient(self)
        self.jobs = JobsClient(self)

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
        data = self._request(
            "POST",
            "/query",
            json={
                "query": query,
                "document_ids": document_ids or [],
                "collection_id": collection_id,
                "top_k": top_k,
                "generate_answer": generate_answer,
                "include_sources": include_sources,
                "filters": filters or {},
                "corrective": corrective,
                "max_retries": max_retries,
            },
        )
        return QueryResult.model_validate(data)

    def correct(
        self,
        text: str,
        *,
        check_types: list[str] | None = None,
        mode: str = "BALANCED",
        apply_changes: bool = True,
    ) -> CorrectionResult:
        data = self._request(
            "POST",
            "/correct",
            json={
                "text": text,
                "check_types": check_types or ["TYPO", "GRAMMAR", "SEMANTIC_UNCLEAR"],
                "mode": mode,
                "apply_changes": apply_changes,
            },
        )
        return CorrectionResult.model_validate(data)

    def health(self) -> dict:
        return self._request("GET", "/health", versioned=False)

    def ready(self) -> dict:
        return self._request("GET", "/ready", versioned=False)

    def _request(
        self,
        method: str,
        path: str,
        *,
        versioned: bool = True,
        **kwargs: Any,
    ) -> Any:
        request_path = f"{self.api_prefix}{path}" if versioned else path
        response = self._http.request(method, request_path, **kwargs)
        try:
            body = response.json()
        except ValueError as exc:
            raise DocGuardAPIError(
                "DocGuard returned a non-JSON response",
                status_code=response.status_code,
            ) from exc
        if response.is_error or body.get("code", 0) != 0:
            raise DocGuardAPIError(
                str(body.get("message", "DocGuard request failed")),
                status_code=response.status_code,
                code=body.get("code"),
                request_id=body.get("request_id"),
            )
        return body.get("data")

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> "DocGuardClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
