from html.parser import HTMLParser

from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.parsers.base import DocumentParser
from docguard_rag.server.parsers.utils import decode_text_file, new_block_id, sentence_blocks


class _BlockHTMLParser(HTMLParser):
    block_tags = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td", "th", "caption"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.current_tag: str | None = None
        self.fragments: list[str] = []
        self.items: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.block_tags:
            self._flush()
            self.current_tag = tag

    def handle_endtag(self, tag: str) -> None:
        if tag == self.current_tag:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self.current_tag:
            self.fragments.append(data)

    def _flush(self) -> None:
        if self.current_tag:
            text = " ".join("".join(self.fragments).split())
            if text:
                self.items.append((self.current_tag, text))
        self.current_tag = None
        self.fragments = []

    def close(self) -> None:
        self._flush()
        super().close()


class HTMLDocumentParser(DocumentParser):
    def supports(self, file_type: str) -> bool:
        return file_type.lower().lstrip(".") in {"html", "htm"}

    def parse(self, file_path: str, document_id: str) -> list[DocumentBlock]:
        parser = _BlockHTMLParser()
        parser.feed(decode_text_file(file_path))
        parser.close()

        blocks: list[DocumentBlock] = []
        section_levels: dict[int, str] = {}
        paragraph_index = 0
        for tag, text in parser.items:
            if tag.startswith("h"):
                level = int(tag[1])
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

            type_by_tag = {
                "li": "list_item",
                "td": "table_row",
                "th": "table_row",
                "caption": "caption",
            }
            block = DocumentBlock(
                block_id=new_block_id(),
                document_id=document_id,
                block_type=type_by_tag.get(tag, "paragraph"),
                text=text,
                section_path=[section_levels[key] for key in sorted(section_levels)],
                paragraph_index=paragraph_index,
            )
            blocks.append(block)
            blocks.extend(sentence_blocks(block))
            paragraph_index += 1
        return blocks
