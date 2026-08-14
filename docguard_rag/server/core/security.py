from dataclasses import dataclass

from fastapi import Header, HTTPException

from docguard_rag.server.core.config import get_settings


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    owner_id: str | None = None


def get_principal(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> Principal:
    settings = get_settings()
    if not settings.auth_enabled:
        return Principal(tenant_id=settings.default_tenant_id)

    api_key = x_api_key
    if not api_key and authorization and authorization.lower().startswith("bearer "):
        api_key = authorization[7:].strip()
    identity = settings.api_keys.get(api_key or "")
    if identity is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return Principal(
        tenant_id=str(identity["tenant_id"]),
        owner_id=identity.get("owner_id"),
    )
