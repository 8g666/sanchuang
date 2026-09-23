import json

import httpx

from docguard_rag.server.embeddings.compatible_embedding import CompatibleEmbedding
from docguard_rag.server.llms.compatible_llm import CompatibleLLM
from docguard_rag.server.rerankers.compatible_reranker import CompatibleReranker


def _mocked(provider, handler):
    """Attach a MockTransport without ever building the real client."""

    provider.client = httpx.Client(
        base_url="http://model.test", transport=httpx.MockTransport(handler)
    )
    return provider


def test_compatible_embedding_provider():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embeddings"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            },
        )

    provider = CompatibleEmbedding(base_url="http://model.test", model_name="embed")
    provider.client.close()
    provider.client = httpx.Client(
        base_url="http://model.test", transport=httpx.MockTransport(handler)
    )

    assert provider.embed_texts(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    provider.close()


def test_compatible_llm_and_reranker_providers():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/chat/completions":
            return httpx.Response(
                200,
                json={
                    "model": "chat",
                    "choices": [{"message": {"content": "依据证据回答。[S1]"}}],
                    "usage": {"total_tokens": 12},
                },
            )
        return httpx.Response(
            200,
            json={"results": [{"index": 1, "relevance_score": 0.98}]},
        )

    transport = httpx.MockTransport(handler)
    llm = CompatibleLLM(base_url="http://model.test", model_name="chat")
    llm.client.close()
    llm.client = httpx.Client(base_url="http://model.test", transport=transport)
    reranker = CompatibleReranker(base_url="http://model.test", model_name="rank")
    reranker.client.close()
    reranker.client = httpx.Client(base_url="http://model.test", transport=transport)

    answer = llm.generate([{"role": "user", "content": "问题"}])
    ranked = reranker.rerank(
        "接口",
        [
            {"chunk_id": "a", "score": 0.1, "metadata": {"text": "其他"}},
            {"chunk_id": "b", "score": 0.2, "metadata": {"text": "接口"}},
        ],
        1,
    )

    assert answer["content"].endswith("[S1]")
    assert ranked[0]["chunk_id"] == "b"
    assert ranked[0]["score"] == 0.98
    llm.close()
    reranker.close()


def test_compatible_embedding_batches_and_preserves_input_order():
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        data = [
            {"index": index, "embedding": [float(len(text))]}
            for index, text in enumerate(payload["input"])
        ]
        # Reversed on purpose: the provider must reorder by index within each batch.
        return httpx.Response(200, json={"data": list(reversed(data))})

    provider = _mocked(
        CompatibleEmbedding(
            base_url="http://model.test", model_name="embed", dimensions=4, batch_size=64
        ),
        handler,
    )
    texts = [f"text-{index}" for index in range(130)]

    vectors = provider.embed_texts(texts)

    assert [len(payload["input"]) for payload in payloads] == [64, 64, 2]
    assert [payload["dimensions"] for payload in payloads] == [4, 4, 4]
    # The vector value encodes its own text's length, so this pins order across batches.
    assert [vector[0] for vector in vectors] == [float(len(text)) for text in texts]
    provider.close()


def test_compatible_embedding_omits_dimensions_when_unconfigured():
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    provider = _mocked(
        CompatibleEmbedding(base_url="http://model.test", model_name="embed"),
        handler,
    )

    assert provider.embed_query("a") == [1.0]
    assert "dimensions" not in payloads[0]
    provider.close()


def test_compatible_providers_close_idempotently_and_rebuild_the_client():
    """The FastAPI lifespan closes providers at shutdown, but the same import-time
    singletons are reused by a later TestClient in one pytest process."""

    for provider in (
        CompatibleEmbedding(base_url="http://model.test", model_name="embed"),
        CompatibleReranker(base_url="http://model.test", model_name="rank"),
        CompatibleLLM(base_url="http://model.test", model_name="chat"),
    ):
        provider.close()
        provider.close()
        assert not provider.client.is_closed
        provider.close()


def test_compatible_providers_expose_a_model_fingerprint():
    embedding = CompatibleEmbedding(
        base_url="http://model.test", model_name="embed", dimensions=2048
    )
    reranker = CompatibleReranker(base_url="http://model.test", model_name="rank")
    llm = CompatibleLLM(base_url="http://model.test", model_name="chat")

    assert embedding.fingerprint == ("compatible_api", "embed", 2048)
    assert reranker.fingerprint == ("compatible_api", "rank")
    assert llm.fingerprint == ("compatible_api", "chat")


def test_compatible_reranker_splits_oversized_candidate_lists():
    """HybridRetriever hands over the fused list, which reaches 400 documents at
    top_k=50 and exceeds the 128-document per-request cap."""

    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        count = len(payload["documents"])
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": index, "relevance_score": 1.0 - index / count}
                    for index in range(count)
                ]
            },
        )

    reranker = _mocked(
        CompatibleReranker(base_url="http://model.test", model_name="rank", batch_size=128),
        handler,
    )
    candidates = [
        {"chunk_id": f"c{index}", "score": 0.0, "metadata": {"text": f"文档 {index}"}}
        for index in range(300)
    ]

    ranked = reranker.rerank("查询", candidates, 5)

    assert [len(payload["documents"]) for payload in payloads] == [128, 128, 44]
    assert all(payload["top_n"] == len(payload["documents"]) for payload in payloads)
    # Stable merge: the three batch leaders tie at 1.0 and keep batch order.
    assert [item["chunk_id"] for item in ranked] == ["c0", "c128", "c256", "c1", "c129"]
    reranker.close()


def test_compatible_reranker_clamps_top_n_to_the_candidate_count():
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}]})

    reranker = _mocked(
        CompatibleReranker(base_url="http://model.test", model_name="rank"),
        handler,
    )

    ranked = reranker.rerank(
        "查询", [{"chunk_id": "a", "score": 0.0, "metadata": {"text": "a"}}], 5
    )

    assert payloads[0]["top_n"] == 1
    assert len(ranked) == 1
    reranker.close()


def test_compatible_reranker_returns_empty_without_calling_the_provider():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("rerank must not call the provider for empty candidates")

    reranker = _mocked(
        CompatibleReranker(base_url="http://model.test", model_name="rank"),
        handler,
    )

    assert reranker.rerank("查询", [], 5) == []
    reranker.close()
