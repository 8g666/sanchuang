from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.parsers.base import DocumentParser
from docguard_rag.server.parsers.utils import new_block_id, sentence_blocks


class PDFParser(DocumentParser):
    def supports(self, file_type: str) -> bool:
        return file_type.lower().lstrip(".") == "pdf"

    def parse(self, file_path: str, document_id: str) -> list[DocumentBlock]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF parsing requires the optional 'pypdf' dependency") from exc

        try:
            reader = PdfReader(file_path)
        except Exception as exc:
            raise ValueError("Invalid or encrypted PDF document") from exc

        blocks: list[DocumentBlock] = []
        paragraph_index = 0
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            for line in page_text.splitlines():
                text = line.strip()
                if not text:
                    continue
                block = DocumentBlock(
                    block_id=new_block_id(),
                    document_id=document_id,
                    block_type="paragraph",
                    text=text,
                    page_number=page_number,
                    paragraph_index=paragraph_index,
                )
                blocks.append(block)
                blocks.extend(sentence_blocks(block))
                paragraph_index += 1
        return blocks
