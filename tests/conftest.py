import shutil
from pathlib import Path
from uuid import uuid4


def pytest_configure(config) -> None:
    """Use an isolated temp root to avoid Windows ACL conflicts between runners."""

    if config.option.basetemp is not None:
        return
    base_temp = Path.cwd() / f".pytest-tmp-{uuid4().hex}"
    config.option.basetemp = str(base_temp)
    config._docguard_generated_basetemp = base_temp


def pytest_sessionfinish(session, exitstatus) -> None:
    base_temp = getattr(session.config, "_docguard_generated_basetemp", None)
    if base_temp is not None:
        shutil.rmtree(base_temp, ignore_errors=True)
