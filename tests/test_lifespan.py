import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from docguard_rag.server.core import provider_factory
from docguard_rag.server.core.config import get_settings
from docguard_rag.server.main import _lifespan
from docguard_rag.server.services.task_dispatcher import InProcessTaskDispatcher


@pytest.fixture
def compatible_embedding(monkeypatch):
    """Build a real CompatibleEmbedding through the factory, so it gets tracked."""

    monkeypatch.setenv("DOCGUARD_EMBEDDING_PROVIDER", "compatible_api")
    monkeypatch.setenv("DOCGUARD_EMBEDDING_BASE_URL", "http://model.test")
    monkeypatch.setenv("DOCGUARD_EMBEDDING_MODEL_NAME", "embed")
    get_settings.cache_clear()
    provider = provider_factory.create_embedding_provider()
    yield provider
    get_settings.cache_clear()
    provider.close()


def test_lifespan_closes_providers_built_by_the_factory(compatible_embedding):
    first_client = compatible_embedding.client
    application = FastAPI(lifespan=_lifespan)

    with TestClient(application):
        pass

    assert first_client.is_closed
    # The registry is drained, so a second pass cannot double-close.
    assert provider_factory.close_providers() == 0


def test_a_closed_provider_rebuilds_its_client(compatible_embedding):
    """pytest enters and exits one shared app's lifespan several times per process, and
    providers are import-time singletons that are never rebuilt. httpx refuses to reuse a
    closed client, so the rebuild is what keeps the second cycle working."""

    application = FastAPI(lifespan=_lifespan)
    first_client = compatible_embedding.client

    for _ in range(2):
        with TestClient(application):
            pass

    assert first_client.is_closed
    assert compatible_embedding.client is not first_client
    assert not compatible_embedding.client.is_closed


def test_the_same_provider_is_tracked_once(compatible_embedding):
    provider_factory._track(compatible_embedding)

    assert provider_factory.close_providers() == 1


def test_close_providers_survives_a_failing_close():
    class Exploding:
        def close(self) -> None:
            raise RuntimeError("boom")

    class Quiet:
        closed = False

        def close(self) -> None:
            self.closed = True

    quiet = Quiet()
    provider_factory._track(Exploding())
    provider_factory._track(quiet)

    assert provider_factory.close_providers() == 1
    assert quiet.closed


def test_in_process_dispatcher_accepts_work_after_shutdown():
    """The REST dispatcher is a singleton bound by name at import time, so a lifespan
    exit must leave it able to serve the next startup."""

    dispatcher = InProcessTaskDispatcher(max_workers=1)

    dispatcher.shutdown(wait=False)

    assert dispatcher.submit(lambda: "ok").result(timeout=5) == "ok"
    dispatcher.shutdown(wait=True)
