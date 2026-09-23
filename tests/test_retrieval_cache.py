import json
from pathlib import Path

from docguard_rag import DocGuard
from docguard_rag.server.core.retrieval_index import index_for_engine
from docguard_rag.server.database.models import DocumentChunkTable
from docguard_rag.server.models.embedding import EmbeddingRecord
from docguard_rag.server.retrievers.lexical_retriever import LexicalRetriever


def write(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def guard_for(tmp_path: Path) -> DocGuard:
    return DocGuard(
        database_url=f"sqlite:///{tmp_path / 'cache.db'}",
        upload_dir=tmp_path / "uploads",
    )


def test_repeated_queries_reuse_one_cached_snapshot(tmp_path):
    with guard_for(tmp_path) as guard:
        guard.documents.upload(write(tmp_path, "kepu.md", "光合作用发生在叶绿体中，需要光能。"))

        first = guard.query("光合作用", generate_answer=False)
        index = index_for_engine(guard._engine)
        assert index.rebuilds == 1

        second = guard.query("叶绿体", generate_answer=False)
        third = guard.query("光能", generate_answer=False)

    assert index.rebuilds == 1
    assert first.sources and second.sources and third.sources


def test_later_ingestion_invalidates_the_cache(tmp_path):
    with guard_for(tmp_path) as guard:
        guard.documents.upload(write(tmp_path, "first.md", "光合作用发生在叶绿体中。"))

        guard.query("光合作用", generate_answer=False)
        index = index_for_engine(guard._engine)
        rebuilds_before = index.rebuilds

        guard.documents.upload(write(tmp_path, "second.md", "细胞呼吸发生在线粒体中。"))
        result = guard.query("细胞呼吸", generate_answer=False)

    assert index.rebuilds == rebuilds_before + 1
    assert any("线粒体" in source.text for source in result.sources)


def test_deletion_is_visible_to_an_independent_retrieval_stack(tmp_path):
    with guard_for(tmp_path) as guard:
        document = guard.documents.upload(write(tmp_path, "kepu.md", "光合作用发生在叶绿体中。"))
        factory = guard._session_factory
        second_stack = LexicalRetriever(session_factory=factory)

        assert second_stack.retrieve("光合作用")

        guard.documents.delete(document.document_id)

        assert second_stack.retrieve("光合作用") == []


def test_replacement_stops_serving_the_previous_content(tmp_path):
    with guard_for(tmp_path) as guard:
        document = guard.documents.upload(write(tmp_path, "kepu.md", "光合作用发生在叶绿体中。"))

        assert any("叶绿体" in source.text for source in guard.query("光合作用").sources)

        guard.documents.update(
            document.document_id, write(tmp_path, "new.md", "细胞呼吸发生在线粒体中。")
        )
        results = guard.query("光合作用 细胞呼吸", generate_answer=False)

    assert results.sources
    assert all("叶绿体" not in source.text for source in results.sources)
    assert any("线粒体" in source.text for source in results.sources)


def test_rolled_back_vectors_never_reach_the_cache(tmp_path):
    with guard_for(tmp_path) as guard:
        guard.documents.upload(write(tmp_path, "real.md", "光合作用发生在叶绿体中。"))
        factory = guard._session_factory

        with factory() as db:
            db.add(
                DocumentChunkTable(
                    chunk_id="ghost-chunk",
                    document_id="ghost-doc",
                    block_id="ghost-block",
                    text="细胞呼吸发生在线粒体中。",
                    char_count=12,
                    token_count=12,
                    chunk_index=0,
                    split_method="test",
                )
            )
            guard._vector_service.add_vectors(
                [
                    EmbeddingRecord(
                        chunk_id="ghost-chunk",
                        vector=[0.1] * 128,
                        metadata={
                            "document_id": "ghost-doc",
                            "text": "细胞呼吸发生在线粒体中。",
                        },
                    )
                ],
                db=db,
            )
            db.rollback()

        guard._vector_service.search([0.1] * 128, top_k=5)

        assert LexicalRetriever(session_factory=factory).retrieve("细胞呼吸") == []
        assert "ghost-chunk" not in guard._vector_service.store.vectors


def test_in_place_reembedding_invalidates_the_cache(tmp_path):
    """Overwriting an existing chunk_id is an UPDATE, which moves neither COUNT(id) nor
    MAX(id). Only the created_at stamp tells the fingerprint that the corpus changed."""

    with guard_for(tmp_path) as guard:
        guard.documents.upload(write(tmp_path, "kepu.md", "光合作用发生在叶绿体中。"))
        factory = guard._session_factory

        guard.query("光合作用", generate_answer=False)
        index = index_for_engine(guard._engine)
        assert index.rebuilds == 1

        with factory() as db:
            chunk_id = db.query(DocumentChunkTable.chunk_id).first()[0]
        original = guard._vector_service.store.vectors[chunk_id]
        fresh = [0.9] * 128
        assert original["vector"] != fresh

        with factory() as db:
            guard._vector_service.add_vectors(
                [
                    EmbeddingRecord(
                        chunk_id=chunk_id,
                        vector=fresh,
                        metadata=original["metadata"],
                    )
                ],
                db=db,
            )
            db.commit()

        guard.query("光合作用", generate_answer=False)

        assert index.rebuilds == 2
        replaced = guard._vector_service.store.vectors[chunk_id]
        assert replaced["vector"] == fresh
        assert replaced["metadata"] == original["metadata"]


def test_document_filter_keeps_bm25_statistics_scoped_to_the_subset(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    alone = tmp_path / "alone"
    alone.mkdir()
    photosynthesis = "光合作用发生在叶绿体中。"
    noisy = "光合作用光合作用光合作用，细胞呼吸与光合作用无关。"

    with guard_for(shared) as both:
        kept = both.documents.upload(write(shared, "kept.md", photosynthesis))
        both.documents.upload(write(shared, "noisy.md", noisy))
        both_factory = both._session_factory
        filtered = LexicalRetriever(session_factory=both_factory).retrieve(
            "光合作用",
            top_k=5,
            filters={"document_id": [kept.document_id]},
        )

    with DocGuard(
        database_url=f"sqlite:///{alone / 'cache.db'}",
        upload_dir=alone / "uploads",
    ) as solo:
        solo.documents.upload(write(alone, "kept.md", photosynthesis))
        unfiltered = LexicalRetriever(session_factory=solo._session_factory).retrieve(
            "光合作用",
            top_k=5,
        )

    assert filtered
    assert [item["score"] for item in filtered] == [item["score"] for item in unfiltered]
    assert [item["metadata"]["text"] for item in filtered] == [
        item["metadata"]["text"] for item in unfiltered
    ]


def test_snapshot_metadata_is_not_mutated_by_scoring(tmp_path):
    with guard_for(tmp_path) as guard:
        guard.documents.upload(
            write(tmp_path, "kepu.md", "光合作用发生在叶绿体中。"),
            collection_id="kepu",
            metadata={"topic": "biology"},
        )
        retriever = LexicalRetriever(session_factory=guard._session_factory)

        first = retriever.retrieve("光合作用", top_k=5, filters={"collection_id": "kepu"})
        index = index_for_engine(guard._engine)
        before = json.dumps(index.snapshot.vectors, ensure_ascii=False, sort_keys=True)

        second = retriever.retrieve("光合作用", top_k=5, filters={"collection_id": "kepu"})
        after = json.dumps(index.snapshot.vectors, ensure_ascii=False, sort_keys=True)

    assert first and second
    assert before == after
