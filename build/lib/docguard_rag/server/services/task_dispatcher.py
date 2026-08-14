from concurrent.futures import Future, ThreadPoolExecutor
from importlib import import_module
from typing import Any

from docguard_rag.server.core.config import get_settings

TASK_NAME = "docguard.execute"


def _callable_path(function) -> str:
    module = getattr(function, "__module__", None)
    qualname = getattr(function, "__qualname__", None)
    if not module or not qualname or "<locals>" in qualname or "." in qualname:
        raise ValueError("Distributed tasks must use an importable top-level function")
    return f"{module}:{qualname}"


def _resolve_callable(path: str):
    module_name, function_name = path.split(":", 1)
    return getattr(import_module(module_name), function_name)


class InProcessTaskDispatcher:
    """Thread-backed dispatcher for local and single-node deployments."""

    def __init__(self, max_workers: int | None = None):
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers or get_settings().task_workers,
            thread_name_prefix="docguard-task",
        )

    def submit(self, function, *args, **kwargs) -> Future:
        return self.executor.submit(function, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait, cancel_futures=False)


class CeleryTaskDispatcher:
    """Durable Redis/Celery dispatcher with retry and late acknowledgement."""

    def __init__(self, celery_application):
        self.app = celery_application

    def submit(self, function, *args, **kwargs):
        return self.app.send_task(
            TASK_NAME,
            args=[_callable_path(function), list(args), kwargs],
        )

    def shutdown(self, wait: bool = True) -> None:
        return None


def create_celery_app():
    try:
        from celery import Celery
    except ImportError as exc:
        raise RuntimeError(
            "Celery backend requires: pip install 'docguard-rag[distributed]'"
        ) from exc

    settings = get_settings()
    application = Celery(
        "docguard_rag",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
    )
    application.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        broker_connection_retry_on_startup=True,
    )

    @application.task(
        bind=True,
        name=TASK_NAME,
        max_retries=settings.celery_task_max_retries,
        acks_late=True,
        reject_on_worker_lost=True,
    )
    def execute(self, callable_path: str, args: list[Any], kwargs: dict[str, Any]):
        try:
            return _resolve_callable(callable_path)(*args, **kwargs)
        except Exception as exc:
            raise self.retry(
                exc=exc,
                countdown=settings.celery_task_retry_delay,
            ) from exc

    return application


def create_dispatcher():
    settings = get_settings()
    if settings.task_backend == "local":
        return InProcessTaskDispatcher()
    if settings.task_backend == "celery":
        return CeleryTaskDispatcher(celery_app)
    raise ValueError(f"Unsupported task backend: {settings.task_backend}")


celery_app = create_celery_app() if get_settings().task_backend == "celery" else None
dispatcher = create_dispatcher()
