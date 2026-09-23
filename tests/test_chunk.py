from docguard_rag.server.chunkers.text_chunker import TextChunker
from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.models.document_chunk import DocumentChunk
from docguard_rag.server.services.chunk_service import ChunkService


def _block(
    block_id: str,
    text: str,
    *,
    block_type: str = "paragraph",
    section_path: list[str] | None = None,
    page_number: int | None = None,
    start_offset: int | None = None,
    document_id: str = "doc001",
) -> DocumentBlock:
    return DocumentBlock(
        block_id=block_id,
        document_id=document_id,
        block_type=block_type,
        text=text,
        section_path=section_path or [],
        page_number=page_number,
        start_offset=start_offset,
        end_offset=start_offset + len(text) if start_offset is not None else None,
    )


def _service(chunk_size: int = 500, overlap: int = 100) -> ChunkService:
    return ChunkService(chunker=TextChunker(chunk_size=chunk_size, overlap=overlap))


def test_document_chunk():
    chunk = DocumentChunk(
        chunk_id="chunk_001",
        document_id="doc_001",
        block_id="blk_001",
        text="RAG是一种检索增强生成技术",
        char_count=15,
        chunk_index=0,
        split_method="recursive",
    )

    assert chunk.chunk_id == "chunk_001"

    assert chunk.document_id == "doc_001"

    assert chunk.chunk_index == 0


def test_text_chunker():
    block = DocumentBlock(
        block_id="blk001", document_id="doc001", block_type="paragraph", text="A" * 1200
    )

    chunker = TextChunker(chunk_size=500, overlap=100)

    chunks = chunker.split(block)

    assert len(chunks) == 3

    assert chunks[0].char_count == 500

    assert chunks[1].chunk_index == 1


def test_short_blocks_in_one_section_merge_into_a_single_chunk():
    """The case the corpus is actually made of: many small blocks, one answer between them."""

    blocks = [_block(f"b{i}", f"积分可以通过完成任务获得，第{i}条规则。") for i in range(8)]

    chunks = _service().create_chunks(blocks)

    assert len(chunks) == 1
    assert chunks[0].char_count == sum(len(block.text) for block in blocks) + 7
    assert [line for line in chunks[0].text.split("\n")] == [block.text for block in blocks]


def test_merging_stops_at_the_configured_chunk_size():
    blocks = [_block(f"b{i}", "字" * 120) for i in range(10)]

    chunks = _service(chunk_size=500).create_chunks(blocks)

    assert len(chunks) > 1
    assert all(chunk.char_count <= 500 for chunk in chunks)
    # 4 blocks of 120 plus 3 separators is 483; a fifth would make 604.
    assert chunks[0].char_count == 483


def test_a_block_larger_than_chunk_size_is_still_split_not_dropped():
    blocks = [_block("small", "短块"), _block("big", "A" * 1200)]

    chunks = _service(chunk_size=500, overlap=100).create_chunks(blocks)

    assert [chunk.block_id for chunk in chunks] == ["small", "big", "big", "big"]
    assert chunks[0].block_ids == ["small"]
    assert all(chunk.block_ids == ["big"] for chunk in chunks[1:])


def test_merging_does_not_cross_a_heading():
    """section_path is the chunk's citation; merging past it would make the citation a lie."""

    blocks = [
        _block("b0", "第一节的内容", section_path=["第一章"]),
        _block("b1", "第一节更多内容", section_path=["第一章"]),
        _block("h", "第二章", block_type="heading", section_path=["第二章"]),
        _block("b2", "第二节的内容", section_path=["第二章"]),
    ]

    chunks = _service().create_chunks(blocks)

    assert len(chunks) == 2
    assert chunks[0].section_path == ["第一章"]
    assert chunks[1].section_path == ["第二章"]
    # The heading itself is not indexable, so its text never becomes a chunk.
    assert all("第二章" != chunk.text for chunk in chunks)


def test_merging_does_not_cross_a_page_break():
    blocks = [
        _block("b0", "第一页的内容", page_number=1),
        _block("b1", "第二页的内容", page_number=2),
    ]

    chunks = _service().create_chunks(blocks)

    assert len(chunks) == 2
    assert [chunk.page_start for chunk in chunks] == [1, 2]


def test_sentence_blocks_are_skipped_and_do_not_break_a_segment():
    blocks = [
        _block("b0", "第一段"),
        _block("s0", "第一段", block_type="sentence"),
        _block("b1", "第二段"),
    ]

    chunks = _service().create_chunks(blocks)

    assert len(chunks) == 1
    assert chunks[0].text == "第一段\n第二段"
    assert chunks[0].block_ids == ["b0", "b1"]


def test_merged_chunk_records_every_member_but_persists_only_the_first():
    """document_chunks has a single block_id column, so block_ids is in-memory only."""

    blocks = [_block(f"b{i}", f"内容{i}") for i in range(4)]

    chunks = _service().create_chunks(blocks)

    assert len(chunks) == 1
    assert chunks[0].block_id == "b0"
    assert chunks[0].block_ids == ["b0", "b1", "b2", "b3"]


def test_chunk_index_is_contiguous_across_segments():
    blocks = [_block(f"b{i}", "字" * 120) for i in range(10)]

    chunks = _service(chunk_size=500).create_chunks(blocks)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_offsets_point_back_into_the_source_document_after_merging():
    """QuerySource publishes these as citations, so joining blocks must not shift them."""

    source = "第一行的内容\n\n第二行的内容\n\n第三行的内容\n"
    blocks = [
        _block("b0", "第一行的内容", start_offset=0),
        _block("b1", "第二行的内容", start_offset=8),
        _block("b2", "第三行的内容", start_offset=16),
    ]

    chunks = _service().create_chunks(blocks)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.start_offset == 0
    # The merged text is 3 lines joined by single newlines, so the segment is shorter than the
    # source span; the restored end offset still has to land on the source, not on the join.
    assert source[chunk.start_offset : chunk.start_offset + len("第一行的内容")] == "第一行的内容"
    assert chunk.end_offset == 16 + len("第三行的内容")
    assert source[chunk.end_offset - len("第三行的内容") : chunk.end_offset] == "第三行的内容"


def test_offsets_inside_a_merged_chunk_stay_exact():
    """Blank lines between source lines, so merged positions and document offsets diverge."""

    source = "".join(f"第{i}行内容\n\n" for i in range(20))
    blocks = []
    position = 0
    for i in range(20):
        text = f"第{i}行内容"
        blocks.append(_block(f"b{i}", text, start_offset=position))
        position += len(text) + 2

    chunks = _service(chunk_size=100, overlap=0).create_chunks(blocks)

    assert len(chunks) > 1
    for chunk in chunks:
        first_line = chunk.text.split("\n")[0]
        assert source[chunk.start_offset : chunk.start_offset + len(first_line)] == first_line


def test_blocks_without_offsets_keep_the_chunker_s_own_numbering():
    """The docx parser sets no offsets, so there is no ground truth to restore."""

    blocks = [_block(f"b{i}", f"内容{i}") for i in range(3)]

    chunks = _service().create_chunks(blocks)

    assert len(chunks) == 1
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == len(chunks[0].text)


def test_a_single_block_is_passed_through_unchanged():
    block = _block("b0", "A" * 1200, start_offset=100)

    chunks = _service(chunk_size=500, overlap=100).create_chunks([block])

    assert len(chunks) == 3
    assert chunks[0].start_offset == 100
    assert chunks[1].start_offset == 500
    assert all(chunk.block_ids == ["b0"] for chunk in chunks)


def test_an_empty_block_list_produces_no_chunks():
    assert _service().create_chunks([]) == []


def test_documents_do_not_merge_into_each_other():
    blocks = [
        _block("a0", "甲文档内容", document_id="docA"),
        _block("b0", "乙文档内容", document_id="docB"),
    ]

    chunks = _service().create_chunks(blocks)

    assert len(chunks) == 2
    assert [chunk.document_id for chunk in chunks] == ["docA", "docB"]
