from docguard_rag.server.parsers.markdown_parser import MarkdownParser


def test_markdown_parser_preserves_structure_and_offsets(tmp_path):
    source = "# 接口设计\n\n系统使用 /v1 接口。第二句话。\n"
    path = tmp_path / "design.md"
    path.write_text(source, encoding="utf-8")
    disk_source = path.read_bytes().decode("utf-8")

    blocks = MarkdownParser().parse(str(path), "doc_markdown")

    heading = next(block for block in blocks if block.block_type == "heading")
    paragraph = next(block for block in blocks if block.block_type == "paragraph")
    sentences = [block for block in blocks if block.block_type == "sentence"]

    assert heading.section_path == ["接口设计"]
    assert paragraph.section_path == ["接口设计"]
    assert paragraph.start_offset == disk_source.index("系统")
    assert [block.text for block in sentences] == ["系统使用 /v1 接口。", "第二句话。"]
    assert sentences[1].start_offset == disk_source.index("第二句话")
