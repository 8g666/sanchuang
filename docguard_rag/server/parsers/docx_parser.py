from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.parsers.base import DocumentParser
from docguard_rag.server.parsers.utils import new_block_id, sentence_blocks

WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": WORD_NAMESPACE}


class DOCXParser(DocumentParser):
    """Dependency-free parser for the text and heading structure in DOCX files."""

    def supports(self, file_type: str) -> bool:
        return file_type.lower().lstrip(".") == "docx"

    def parse(self, file_path: str, document_id: str) -> list[DocumentBlock]:
        try:
            with ZipFile(file_path) as archive:
                xml_content = archive.read("word/document.xml")
        except (BadZipFile, KeyError) as exc:
            raise ValueError("Invalid DOCX document") from exc

        root = ElementTree.fromstring(xml_content)
        blocks: list[DocumentBlock] = []
        section_levels: dict[int, str] = {}
        paragraph_index = 0

        for paragraph in root.findall(".//w:body/w:p", NS):
            text = "".join(node.text or "" for node in paragraph.findall(".//w:t", NS)).strip()
            if not text:
                continue
            style_node = paragraph.find("./w:pPr/w:pStyle", NS)
            style = style_node.get(f"{{{WORD_NAMESPACE}}}val", "") if style_node is not None else ""
            is_heading = style.lower().startswith("heading") or style.startswith("标题")
            if is_heading:
                digits = "".join(character for character in style if character.isdigit())
                level = int(digits or "1")
                section_levels = {
                    key: value for key, value in section_levels.items() if key < level
                }
                section_levels[level] = text
                blocks.append(
                    DocumentBlock(
                        block_id=new_block_id(),
                        document_id=document_id,
                        block_type="heading",
                        text=text,
                        section_path=[section_levels[key] for key in sorted(section_levels)],
                        metadata={"level": level},
                    )
                )
                continue

            block = DocumentBlock(
                block_id=new_block_id(),
                document_id=document_id,
                block_type="paragraph",
                text=text,
                section_path=[section_levels[key] for key in sorted(section_levels)],
                paragraph_index=paragraph_index,
            )
            blocks.append(block)
            blocks.extend(sentence_blocks(block))
            paragraph_index += 1

        return blocks
