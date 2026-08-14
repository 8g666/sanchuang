import httpx

from docguard_rag.server.embeddings.compatible_embedding import CompatibleEmbedding
from docguard_rag.server.llms.compatible_llm import CompatibleLLM
from docguard_rag.server.rerankers.compatible_reranker import CompatibleReranker


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
