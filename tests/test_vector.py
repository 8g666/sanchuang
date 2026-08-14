from docguard_rag.server.vectorstores.simple_vector_store import SimpleVectorStore


def test_vector_search():

    store = SimpleVectorStore()

    store.add("chunk1", [1, 0, 0], {})

    store.add("chunk2", [0, 1, 0], {})

    result = store.search([0.9, 0.1, 0], top_k=1)

    assert result[0]["chunk_id"] == "chunk1"
