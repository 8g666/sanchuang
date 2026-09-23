import posixpath
import re
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.parsers.base import DocumentParser
from docguard_rag.server.parsers.utils import new_block_id, sentence_blocks

DRAWING_NAMESPACE = "http://schemas.openxmlformats.org/drawingml/2006/main"
PRESENTATION_NAMESPACE = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = f"{{{DRAWING_NAMESPACE}}}"
P = f"{{{PRESENTATION_NAMESPACE}}}"

SLIDE_PART = re.compile(r"^ppt/slides/slide(\d+)\.xml$")
# ctrTitle is the centred title placeholder used by title slides.
TITLE_PLACEHOLDERS = {"title", "ctrTitle"}


class PPTXParser(DocumentParser):
    """Dependency-free parser for the text in PPTX slides and their speaker notes.

    Slide masters and layouts are deliberately not read: they hold the template's own
    placeholder text, which would repeat on every slide.
    """

    def supports(self, file_type: str) -> bool:
        return file_type.lower().lstrip(".") == "pptx"

    def parse(self, file_path: str, document_id: str) -> list[DocumentBlock]:
        try:
            with ZipFile(file_path) as archive:
                slides = self._slides(archive)
                if not slides:
                    raise ValueError("Invalid PPTX presentation")
                blocks: list[DocumentBlock] = []
                paragraph_index = 0
                for number, part in slides:
                    root = ElementTree.fromstring(archive.read(part))
                    title = self._title(root)
                    section_path = [title or f"幻灯片 {number}"]
                    if title:
                        blocks.append(
                            DocumentBlock(
                                block_id=new_block_id(),
                                document_id=document_id,
                                block_type="heading",
                                text=title,
                                page_number=number,
                                section_path=list(section_path),
                                metadata={"level": 1},
                            )
                        )
                    texts = self._paragraphs(root, self._title_paragraphs(root))
                    notes = self._notes_part(archive, part)
                    if notes is not None:
                        notes_root = ElementTree.fromstring(archive.read(notes))
                        texts += self._paragraphs(notes_root, set())
                    for text in texts:
                        blocks.append(
                            self._block(text, document_id, number, section_path, paragraph_index)
                        )
                        blocks.extend(sentence_blocks(blocks[-1]))
                        paragraph_index += 1
        except (BadZipFile, KeyError) as exc:
            raise ValueError("Invalid PPTX presentation") from exc

        return blocks

    @staticmethod
    def _slides(archive: ZipFile) -> list[tuple[int, str]]:
        """Slide parts in presentation order.

        Sorted by the number in the part name rather than lexicographically, so slide 10 does
        not land between slide 1 and slide 2.
        """

        found = []
        for name in archive.namelist():
            match = SLIDE_PART.match(name)
            if match:
                found.append((int(match.group(1)), name))
        return sorted(found)

    @staticmethod
    def _notes_part(archive: ZipFile, slide_part: str) -> str | None:
        """Resolve the slide's speaker-notes part through its relationships.

        The number in notesSlideN.xml does not have to match the slide's own, so the mapping is
        read from the relationship file instead of being guessed from the name.
        """

        rels = f"ppt/slides/_rels/{slide_part.rsplit('/', 1)[-1]}.rels"
        try:
            root = ElementTree.fromstring(archive.read(rels))
        except KeyError:
            return None
        names = set(archive.namelist())
        for relationship in root:
            target = relationship.get("Target") or ""
            if "notesSlide" not in target:
                continue
            candidate = posixpath.normpath(posixpath.join("ppt/slides", target))
            if candidate in names:
                return candidate
        return None

    @staticmethod
    def _title_shapes(root: ElementTree.Element) -> list[ElementTree.Element]:
        return [
            shape
            for shape in root.findall(f".//{P}sp")
            if (placeholder := shape.find(f"./{P}nvSpPr/{P}nvPr/{P}ph")) is not None
            and placeholder.get("type") in TITLE_PLACEHOLDERS
        ]

    @classmethod
    def _title(cls, root: ElementTree.Element) -> str | None:
        for shape in cls._title_shapes(root):
            parts = [_paragraph_text(paragraph) for paragraph in shape.findall(f".//{A}p")]
            text = " ".join(part for part in parts if part)
            if text:
                return text
        return None

    @classmethod
    def _title_paragraphs(cls, root: ElementTree.Element) -> set[int]:
        """Identity of the title shapes' paragraphs, so their text is not emitted twice."""

        return {
            id(paragraph)
            for shape in cls._title_shapes(root)
            for paragraph in shape.findall(f".//{A}p")
        }

    @staticmethod
    def _paragraphs(root: ElementTree.Element, skip: set[int]) -> list[str]:
        """Every non-empty paragraph in document order, including table cells and groups."""

        texts = []
        for paragraph in root.findall(f".//{A}p"):
            if id(paragraph) in skip:
                continue
            text = _paragraph_text(paragraph)
            if text:
                texts.append(text)
        return texts

    @staticmethod
    def _block(
        text: str,
        document_id: str,
        page_number: int,
        section_path: list[str],
        paragraph_index: int,
    ) -> DocumentBlock:
        return DocumentBlock(
            block_id=new_block_id(),
            document_id=document_id,
            block_type="paragraph",
            text=text,
            page_number=page_number,
            section_path=list(section_path),
            paragraph_index=paragraph_index,
        )


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    """Join a paragraph's runs, dropping field results.

    `a:fld` carries cached values for the slide number, date and footer, which are template
    noise rather than authored content.
    """

    parts = []
    for child in paragraph:
        if child.tag == f"{A}r":
            run = child.find(f"{A}t")
            if run is not None and run.text:
                parts.append(run.text)
        elif child.tag == f"{A}br":
            parts.append("\n")
    return "".join(parts).strip()
