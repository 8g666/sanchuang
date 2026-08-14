from zipfile import ZIP_DEFLATED, ZipFile

from docguard_rag.server.parsers.docx_parser import DOCXParser
from docguard_rag.server.parsers.html_parser import HTMLDocumentParser


def test_html_parser_preserves_heading_context(tmp_path):
    path = tmp_path / "document.html"
    path.write_text(
        "<h1>接口设计</h1><p>服务使用 <strong>/v1</strong> 接口。</p>",
        encoding="utf-8",
    )

    blocks = HTMLDocumentParser().parse(str(path), "doc_html")
    paragraph = next(block for block in blocks if block.block_type == "paragraph")

    assert paragraph.text == "服务使用 /v1 接口。"
    assert paragraph.section_path == ["接口设计"]


def test_docx_parser_without_external_docx_dependency(tmp_path):
    path = tmp_path / "document.docx"
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>总体设计</w:t></w:r></w:p>
        <w:p><w:r><w:t>系统支持结构化解析。</w:t></w:r></w:p>
      </w:body>
    </w:document>"""
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document_xml)

    blocks = DOCXParser().parse(str(path), "doc_docx")
    paragraph = next(block for block in blocks if block.block_type == "paragraph")

    assert paragraph.text == "系统支持结构化解析。"
    assert paragraph.section_path == ["总体设计"]
