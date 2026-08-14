from docguard_rag.server.chunkers.text_chunker import TextChunker
from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.models.document_chunk import DocumentChunk


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
