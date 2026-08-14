import re
from uuid import uuid4

from docguard_rag.server.models.document_block import DocumentBlock

SENTENCE_PATTERN = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?", re.MULTILINE)


def new_block_id() -> str:
    return f"blk_{uuid4().hex[:12]}"


def sentence_blocks(
    paragraph: DocumentBlock,
    *,
    first_sentence_index: int = 0,
) -> list[DocumentBlock]:
    """Create locatable sentence blocks from one paragraph-like block."""

    blocks: list[DocumentBlock] = []
    for relative_index, match in enumerate(SENTENCE_PATTERN.finditer(paragraph.text)):
        raw_text = match.group(0)
        sentence = raw_text.strip()
        if not sentence:
            continue
        leading = len(raw_text) - len(raw_text.lstrip())
        relative_start = match.start() + leading
        absolute_start = (
            paragraph.start_offset + relative_start
            if paragraph.start_offset is not None
            else relative_start
        )
        blocks.append(
            DocumentBlock(
                block_id=new_block_id(),
                document_id=paragraph.document_id,
                block_type="sentence",
                text=sentence,
                page_number=paragraph.page_number,
                section_path=list(paragraph.section_path),
                paragraph_index=paragraph.paragraph_index,
                sentence_index=first_sentence_index + relative_index,
                start_offset=absolute_start,
                end_offset=absolute_start + len(sentence),
                metadata={"parent_id": paragraph.block_id},
            )
        )
    return blocks


def decode_text_file(file_path: str) -> str:
    """Decode common Chinese and Unicode text files without silent data loss."""

    with open(file_path, "rb") as file:
        content = file.read()

    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("The text file is neither UTF-8 nor GB18030 encoded")
