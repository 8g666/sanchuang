import pytest
from fastapi import HTTPException

from docguard_rag.server.core import security
from docguard_rag.server.core.config import Settings


def test_api_key_maps_to_tenant_and_owner(monkeypatch):
    settings = Settings(
        _env_file=None,
        auth_enabled=True,
        api_keys_json=(
            '{"secret-a":{"tenant_id":"tenant-a","owner_id":"owner-a"},"secret-b":"tenant-b"}'
        ),
    )
    monkeypatch.setattr(security, "get_settings", lambda: settings)

    bearer = security.get_principal("Bearer secret-a", None)
    header = security.get_principal(None, "secret-b")

    assert (bearer.tenant_id, bearer.owner_id) == ("tenant-a", "owner-a")
    assert (header.tenant_id, header.owner_id) == ("tenant-b", None)


def test_api_key_rejects_invalid_credentials(monkeypatch):
    settings = Settings(
        _env_file=None,
        auth_enabled=True,
        api_keys_json='{"secret":"tenant"}',
    )
    monkeypatch.setattr(security, "get_settings", lambda: settings)

    with pytest.raises(HTTPException) as exc_info:
        security.get_principal("Bearer wrong", None)

    assert exc_info.value.status_code == 401
