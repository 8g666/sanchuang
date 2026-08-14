from docguard_rag.server.embeddings.fake_embedding import FakeEmbedding


def test_fake_embedding():

    model = FakeEmbedding(dimension=8)

    vectors = model.embed_texts(["RAG是一种技术"])

    assert len(vectors) == 1

    assert len(vectors[0]) == 8
