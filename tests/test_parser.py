from docguard_rag.server.parsers.txt_parser import TXTParser


def test_txt_parser(tmp_path):
    parser = TXTParser()
    source = tmp_path / "test.txt"
    source.write_text("A self-contained parser test.", encoding="utf-8")
    blocks = parser.parse(str(source), "doc_test")
    assert len(blocks) > 0
    assert blocks[0].document_id == "doc_test"
    assert blocks[0].block_type == "paragraph"
