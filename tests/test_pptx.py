import io
import zipfile

import pytest

from docguard_rag.server.chunkers.text_chunker import TextChunker
from docguard_rag.server.parsers.pptx_parser import PPTXParser
from docguard_rag.server.services.chunk_service import ChunkService

A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NOTES_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide"


def _run(text: str) -> str:
    return f'<a:r><a:rPr lang="zh-CN"/><a:t>{text}</a:t></a:r>'


def _shape(name: str, paragraphs: str, placeholder: str = "") -> str:
    inner = f'<p:ph type="{placeholder}"/>' if placeholder else ""
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="1" name="{name}"/><p:cNvSpPr/>'
        f"<p:nvPr>{inner}</p:nvPr></p:nvSpPr><p:spPr/>"
        f"<p:txBody><a:bodyPr/><a:lstStyle/>{paragraphs}</p:txBody></p:sp>"
    )


def _slide(shapes: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<p:sld xmlns:a="{A_NS}" xmlns:p="{P_NS}"><p:cSld><p:spTree>{shapes}'
        "</p:spTree></p:cSld></p:sld>"
    ).encode()


def _notes(shapes: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<p:notes xmlns:a="{A_NS}" xmlns:p="{P_NS}"><p:cSld><p:spTree>{shapes}'
        "</p:spTree></p:cSld></p:notes>"
    ).encode()


def _rels(target: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{REL_NS}">'
        f'<Relationship Id="rId1" Type="{NOTES_REL}" Target="{target}"/>'
        "</Relationships>"
    ).encode()


def _build(parts: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _write(tmp_path, parts: dict[str, bytes]) -> str:
    path = tmp_path / "deck.pptx"
    path.write_bytes(_build(parts))
    return str(path)


def _blocks(path: str) -> list:
    """Parsed blocks minus the sentence sub-blocks, which repeat their parent's text."""

    return [block for block in PPTXParser().parse(path, "doc1") if block.block_type != "sentence"]


def _texts(path: str) -> list[str]:
    return [block.text for block in _blocks(path)]


def test_supports_pptx_only():
    parser = PPTXParser()

    assert parser.supports("pptx")
    assert parser.supports(".PPTX")
    assert not parser.supports("ppt")
    assert not parser.supports("docx")


def test_title_becomes_a_heading_and_body_becomes_paragraphs(tmp_path):
    slide = _slide(
        _shape("标题", f"<a:p>{_run('注册流程')}</a:p>", "title")
        + _shape("正文", f"<a:p>{_run('第一步')}</a:p><a:p>{_run('第二步')}</a:p>", "body")
    )
    path = _write(tmp_path, {"ppt/slides/slide1.xml": slide})

    blocks = PPTXParser().parse(path, "doc1")
    kinds = [(block.block_type, block.text) for block in blocks if block.block_type != "sentence"]

    assert kinds == [
        ("heading", "注册流程"),
        ("paragraph", "第一步"),
        ("paragraph", "第二步"),
    ]
    # Sentence blocks are what the review pipeline locates issues with.
    assert any(block.block_type == "sentence" for block in blocks)


def test_a_centred_title_placeholder_is_also_a_heading(tmp_path):
    slide = _slide(_shape("Title", f"<a:p>{_run('通境三创服务平台')}</a:p>", "ctrTitle"))
    path = _write(tmp_path, {"ppt/slides/slide1.xml": slide})

    blocks = PPTXParser().parse(path, "doc1")

    assert [block.block_type for block in blocks] == ["heading"]
    assert blocks[0].metadata == {"level": 1}


def test_the_title_is_not_also_emitted_as_a_paragraph(tmp_path):
    slide = _slide(
        _shape("标题", f"<a:p>{_run('平台简介')}</a:p>", "title")
        + _shape("正文", f"<a:p>{_run('正文')}</a:p>", "body")
    )
    path = _write(tmp_path, {"ppt/slides/slide1.xml": slide})

    assert _texts(path).count("平台简介") == 1


def test_cached_field_values_are_dropped(tmp_path):
    """Slide numbers, dates and footers are template noise, not authored content."""

    slide = _slide(
        _shape(
            "页码",
            '<a:p><a:fld id="{GUID}" type="slidenum"><a:rPr lang="zh-CN"/>'
            "<a:t>7</a:t></a:fld></a:p>",
            "sldNum",
        )
        + _shape("正文", f"<a:p>{_run('真实内容')}</a:p>", "body")
    )
    path = _write(tmp_path, {"ppt/slides/slide1.xml": slide})

    assert _texts(path) == ["真实内容"]


def test_a_soft_line_break_inside_a_paragraph_is_preserved(tmp_path):
    slide = _slide(
        _shape(
            "标题",
            f"<a:p>{_run('通境（TongVerse）——')}<a:br/>{_run('OPC三创平台')}</a:p>",
            "title",
        )
    )
    path = _write(tmp_path, {"ppt/slides/slide1.xml": slide})

    assert PPTXParser().parse(path, "doc1")[0].text == "通境（TongVerse）——\nOPC三创平台"


def test_empty_paragraphs_are_skipped(tmp_path):
    slide = _slide(
        _shape("正文", f"<a:p>{_run('有内容')}</a:p><a:p/><a:p><a:endParaRPr/></a:p>", "body")
    )
    path = _write(tmp_path, {"ppt/slides/slide1.xml": slide})

    assert _texts(path) == ["有内容"]


def test_slides_are_ordered_numerically_not_lexicographically(tmp_path):
    parts = {
        f"ppt/slides/slide{n}.xml": _slide(_shape("b", f"<a:p>{_run(f'第{n}页')}</a:p>", "body"))
        for n in (1, 2, 10)
    }
    path = _write(tmp_path, parts)

    blocks = _blocks(path)

    assert [block.page_number for block in blocks] == [1, 2, 10]
    assert [block.text for block in blocks] == ["第1页", "第2页", "第10页"]


def test_page_number_and_section_path_come_from_the_slide(tmp_path):
    parts = {
        "ppt/slides/slide1.xml": _slide(
            _shape("t", f"<a:p>{_run('注册流程')}</a:p>", "title")
            + _shape("b", f"<a:p>{_run('第一步')}</a:p>", "body")
        ),
        "ppt/slides/slide2.xml": _slide(_shape("b", f"<a:p>{_run('没有标题的页')}</a:p>", "body")),
    }
    path = _write(tmp_path, parts)

    heading, paragraph, untitled = _blocks(path)

    assert heading.page_number == 1
    assert paragraph.page_number == 1
    assert paragraph.section_path == ["注册流程"]
    # A slide with no title placeholder still gets a per-slide section, so the chunker treats
    # it as a boundary rather than merging it into the previous slide.
    assert untitled.page_number == 2
    assert untitled.section_path == ["幻灯片 2"]


def test_paragraph_index_runs_across_the_whole_deck(tmp_path):
    parts = {
        f"ppt/slides/slide{n}.xml": _slide(
            _shape("b", f"<a:p>{_run(f'{n}a')}</a:p><a:p>{_run(f'{n}b')}</a:p>", "body")
        )
        for n in (1, 2)
    }
    path = _write(tmp_path, parts)

    indexes = [
        block.paragraph_index
        for block in PPTXParser().parse(path, "doc1")
        if block.block_type == "paragraph"
    ]

    assert indexes == [0, 1, 2, 3]


def test_speaker_notes_are_attached_to_their_own_slide(tmp_path):
    parts = {
        "ppt/slides/slide1.xml": _slide(_shape("b", f"<a:p>{_run('幻灯片正文')}</a:p>", "body")),
        "ppt/slides/_rels/slide1.xml.rels": _rels("../notesSlides/notesSlide9.xml"),
        # The number in the part name deliberately differs from the slide's: the mapping has to
        # come from the relationship file, not from the name.
        "ppt/notesSlides/notesSlide9.xml": _notes(
            _shape("n", f"<a:p>{_run('演讲者备注')}</a:p>", "body")
        ),
    }
    path = _write(tmp_path, parts)

    blocks = _blocks(path)

    assert [(block.text, block.page_number) for block in blocks] == [
        ("幻灯片正文", 1),
        ("演讲者备注", 1),
    ]


def test_a_slide_without_a_relationships_file_yields_no_extra_blocks(tmp_path):
    parts = {
        "ppt/slides/slide1.xml": _slide(_shape("b", f"<a:p>{_run('只有正文')}</a:p>", "body")),
    }
    path = _write(tmp_path, parts)

    assert _texts(path) == ["只有正文"]


def test_a_relationship_to_a_missing_notes_part_is_ignored(tmp_path):
    parts = {
        "ppt/slides/slide1.xml": _slide(_shape("b", f"<a:p>{_run('只有正文')}</a:p>", "body")),
        "ppt/slides/_rels/slide1.xml.rels": _rels("../notesSlides/notesSlide1.xml"),
    }
    path = _write(tmp_path, parts)

    assert _texts(path) == ["只有正文"]


def test_notes_without_text_yield_no_extra_blocks(tmp_path):
    parts = {
        "ppt/slides/slide1.xml": _slide(_shape("b", f"<a:p>{_run('正文')}</a:p>", "body")),
        "ppt/slides/_rels/slide1.xml.rels": _rels("../notesSlides/notesSlide1.xml"),
        "ppt/notesSlides/notesSlide1.xml": _notes(_shape("n", "<a:p/>", "body")),
    }
    path = _write(tmp_path, parts)

    assert _texts(path) == ["正文"]


def test_table_cell_text_is_extracted(tmp_path):
    table = (
        "<p:graphicFrame><p:txBody/><a:graphic><a:graphicData>"
        "<a:tbl><a:tr><a:tc><a:txBody>"
        f"<a:p>{_run('服务费比例')}</a:p>"
        "</a:txBody></a:tc><a:tc><a:txBody>"
        f"<a:p>{_run('5%')}</a:p>"
        "</a:txBody></a:tc></a:tr></a:tbl>"
        "</a:graphicData></a:graphic></p:graphicFrame>"
    )
    path = _write(tmp_path, {"ppt/slides/slide1.xml": _slide(table)})

    assert _texts(path) == ["服务费比例", "5%"]


def test_text_inside_a_group_shape_is_extracted(tmp_path):
    grouped = (
        "<p:grpSp><p:nvGrpSpPr/><p:grpSpPr/>"
        + _shape("b", f"<a:p>{_run('组合内文字')}</a:p>", "body")
        + "</p:grpSp>"
    )
    path = _write(tmp_path, {"ppt/slides/slide1.xml": _slide(grouped)})

    assert _texts(path) == ["组合内文字"]


def test_masters_and_layouts_are_not_read(tmp_path):
    """Their placeholder text would otherwise repeat on every slide."""

    parts = {
        "ppt/slides/slide1.xml": _slide(_shape("b", f"<a:p>{_run('真实正文')}</a:p>", "body")),
        "ppt/slideMasters/slideMaster1.xml": _slide(_shape("m", f"<a:p>{_run('母版文字')}</a:p>")),
        "ppt/slideLayouts/slideLayout1.xml": _slide(
            _shape("l", f"<a:p>{_run('单击此处添加标题')}</a:p>", "title")
        ),
    }
    path = _write(tmp_path, parts)

    assert _texts(path) == ["真实正文"]


def test_a_zip_without_slides_is_rejected(tmp_path):
    path = _write(tmp_path, {"word/document.xml": b"<w:document/>"})

    with pytest.raises(ValueError, match="Invalid PPTX"):
        PPTXParser().parse(path, "doc1")


def test_a_file_that_is_not_a_zip_is_rejected(tmp_path):
    path = tmp_path / "broken.pptx"
    path.write_bytes(b"not a zip archive at all")

    with pytest.raises(ValueError, match="Invalid PPTX"):
        PPTXParser().parse(str(path), "doc1")


def test_slides_do_not_merge_into_each_other_when_chunked(tmp_path):
    """Each slide is one citation unit, so its page number has to stay true for the whole chunk."""

    parts = {
        f"ppt/slides/slide{n}.xml": _slide(
            _shape("t", f"<a:p>{_run(f'第{n}页标题')}</a:p>", "title")
            + _shape("b", f"<a:p>{_run(f'第{n}页的一句很短的正文')}</a:p>", "body")
        )
        for n in (1, 2, 3)
    }
    path = _write(tmp_path, parts)

    blocks = PPTXParser().parse(path, "doc1")
    chunks = ChunkService(chunker=TextChunker(chunk_size=500, overlap=100)).create_chunks(blocks)

    assert [chunk.page_start for chunk in chunks] == [1, 2, 3]
    assert [chunk.section_path for chunk in chunks] == [["第1页标题"], ["第2页标题"], ["第3页标题"]]
