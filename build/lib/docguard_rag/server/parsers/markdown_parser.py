import re

from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.parsers.base import DocumentParser
from docguard_rag.server.parsers.utils import decode_text_file, new_block_id, sentence_blocks


class MarkdownParser(DocumentParser):
    """Parse Markdown headings and body blocks while retaining source offsets."""

    def supports(self, file_type: str) -> bool:
        return file_type.lower().lstrip(".") in {"md", "markdown"}

    def parse(self, file_path: str, document_id: str) -> list[DocumentBlock]:
        content = decode_text_file(file_path)
        blocks: list[DocumentBlock] = []
        section_levels: dict[int, str] = {}
        paragraph_index = 0

        for match in re.finditer(r"[^\r\n]+", content):
            raw_line = match.group(0)
            line = raw_line.strip()
            if not line or line.startswith("```"):
                continue

            heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*#*$", line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                section_levels = {
                    key: value for key, value in section_levels.items() if key < level
                }
                section_levels[level] = title
                leading = raw_line.find(title)
                start = match.start() + max(leading, 0)
                blocks.append(
                    DocumentBlock(
                        block_id=new_block_id(),
                        document_id=document_id,
                        block_type="heading",
                        text=title,
                        section_path=[section_levels[key] for key in sorted(section_levels)],
                        start_offset=start,
                        end_offset=start + len(title),
                        metadata={"level": level},
                    )
                )
                continue

            block_type = "list_item" if re.match(r"^(?:[-*+] |\d+[.)]\s+)", line) else "paragraph"
            text = (
                re.sub(r"^(?:[-*+] |\d+[.)]\s+)", "", line) if block_type == "list_item" else line
            )
            start = match.start() + raw_line.find(text)
            block = DocumentBlock(
                block_id=new_block_id(),
                document_id=document_id,
                block_type=block_type,
                text=text,
                section_path=[section_levels[key] for key in sorted(section_levels)],
                paragraph_index=paragraph_index,
                start_offset=start,
                end_offset=start + len(text),
            )
            blocks.append(block)
            blocks.extend(sentence_blocks(block))
            paragraph_index += 1

        return blocks
