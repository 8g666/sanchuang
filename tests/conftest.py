import os
import shutil
import sys
from pathlib import Path
from uuid import uuid4

# Settings reads .env from the working directory, so a developer's real model
# credentials would otherwise make the suite call out to the network and change the
# vector width mid-run. Environment variables outrank dotenv values in pydantic-settings,
# so pinning them here keeps every test on the offline stubs.
OFFLINE_DEFAULTS = {
    "DOCGUARD_VECTOR_STORE_PROVIDER": "local",
    "DOCGUARD_EMBEDDING_PROVIDER": "hash",
    "DOCGUARD_EMBEDDING_DIMENSION": "128",
    "DOCGUARD_RERANKER_ENABLED": "true",
    "DOCGUARD_RERANKER_PROVIDER": "overlap",
    "DOCGUARD_LLM_PROVIDER": "extractive",
    "DOCGUARD_LLM_REVIEW_ENABLED": "false",
    "DOCGUARD_CORRECTIVE_RAG_ENABLED": "true",
    "DOCGUARD_TASK_BACKEND": "local",
}


def pytest_configure(config) -> None:
    """Use an isolated temp root to avoid Windows ACL conflicts between runners."""

    if config.option.basetemp is None:
        base_temp = Path.cwd() / f".pytest-tmp-{uuid4().hex}"
        config.option.basetemp = str(base_temp)
        config._docguard_generated_basetemp = base_temp

    # Not inside basetemp: pytest rm_rf's a given basetemp on the first tmp_path request,
    # which would delete the database file out from under a live connection.
    isolation = Path.cwd() / f".pytest-tmp-db-{uuid4().hex}"
    isolation.mkdir(parents=True)
    config._docguard_isolation_dir = isolation

    # setdefault throughout, so an operator-supplied value still wins.
    for name, value in OFFLINE_DEFAULTS.items():
        os.environ.setdefault(name, value)
    os.environ.setdefault("DOCGUARD_UPLOAD_DIR", str(isolation / "uploads"))
    # database/connection.py binds its engine at import time, so this has to be set
    # before collection or the suite opens the git-tracked ./docguard.db.
    os.environ.setdefault("DOCGUARD_DATABASE_URL", f"sqlite:///{isolation / 'pytest-global.db'}")


def pytest_sessionfinish(session, exitstatus) -> None:
    # The pooled connection keeps the SQLite file open; without disposing it rmtree
    # silently leaves the directory behind on Windows.
    connection = sys.modules.get("docguard_rag.server.database.connection")
    if connection is not None:
        connection.engine.dispose()
    base_temp = getattr(session.config, "_docguard_generated_basetemp", None)
    if base_temp is not None:
        shutil.rmtree(base_temp, ignore_errors=True)
    isolation = getattr(session.config, "_docguard_isolation_dir", None)
    if isolation is not None:
        shutil.rmtree(isolation, ignore_errors=True)
