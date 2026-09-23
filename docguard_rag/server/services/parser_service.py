from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.parsers.base import DocumentParser
from docguard_rag.server.parsers.docx_parser import DOCXParser
from docguard_rag.server.parsers.html_parser import HTMLDocumentParser
from docguard_rag.server.parsers.image_parser import ImageOCRParser
from docguard_rag.server.parsers.markdown_parser import MarkdownParser
from docguard_rag.server.parsers.pdf_parser import PDFParser
from docguard_rag.server.parsers.pptx_parser import PPTXParser
from docguard_rag.server.parsers.txt_parser import TXTParser


class ParserService:
    def __init__(self, parsers: list[DocumentParser] | None = None):
        self.parsers = parsers or [
            TXTParser(),
            MarkdownParser(),
            HTMLDocumentParser(),
            DOCXParser(),
            PPTXParser(),
            PDFParser(),
            ImageOCRParser(),
        ]

    def parse(
        self,
        file_path: str,
        file_type: str,
        document_id: str,
    ) -> list[DocumentBlock]:
        normalized_type = file_type.lower().lstrip(".")
        for parser in self.parsers:
            if parser.supports(normalized_type):
                return parser.parse(file_path, document_id)
        raise ValueError(f"Unsupported document type: {normalized_type}")
