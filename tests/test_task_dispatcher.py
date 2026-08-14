import pytest

from docguard_rag.server.services.task_dispatcher import TASK_NAME, CeleryTaskDispatcher


def sample_task(value, *, enabled=False):
    return value if enabled else None


class FakeCelery:
    def __init__(self):
        self.calls = []

    def send_task(self, name, *, args):
        self.calls.append((name, args))
        return "task-result"


def test_celery_dispatcher_serializes_importable_task():
    app = FakeCelery()
    dispatcher = CeleryTaskDispatcher(app)
    result = dispatcher.submit(sample_task, "value", enabled=True)

    assert result == "task-result"
    assert app.calls == [(TASK_NAME, [f"{__name__}:sample_task", ["value"], {"enabled": True}])]


def test_celery_dispatcher_rejects_nested_callable():
    dispatcher = CeleryTaskDispatcher(FakeCelery())

    def nested():
        return None

    with pytest.raises(ValueError, match="importable top-level function"):
        dispatcher.submit(nested)
