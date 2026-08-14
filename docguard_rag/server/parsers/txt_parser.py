import re

from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.parsers.base import DocumentParser
from docguard_rag.server.parsers.utils import decode_text_file, new_block_id, sentence_blocks


class TXTParser(DocumentParser):
    def supports(self, file_type: str) -> bool:

        return file_type.lower() in ["txt"]

    def parse(self, file_path: str, document_id: str) -> list[DocumentBlock]:

        blocks: list[DocumentBlock] = []
        content = decode_text_file(file_path)

        paragraph_index = 0
        for match in re.finditer(r"[^\r\n]+", content):
            raw_text = match.group(0)
            paragraph = raw_text.strip()
            if not paragraph:
                continue
            leading = len(raw_text) - len(raw_text.lstrip())
            start_offset = match.start() + leading
            block = DocumentBlock(
                block_id=new_block_id(),
                document_id=document_id,
                block_type="paragraph",
                text=paragraph,
                paragraph_index=paragraph_index,
                start_offset=start_offset,
                end_offset=start_offset + len(paragraph),
            )
            blocks.append(block)
            blocks.extend(sentence_blocks(block))
            paragraph_index += 1

        return blocks
