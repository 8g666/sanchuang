from pathlib import Path

from docguard_rag.server.models.document_block import DocumentBlock
from docguard_rag.server.parsers.base import DocumentParser
from docguard_rag.server.parsers.utils import new_block_id, sentence_blocks

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"}


class ImageOCRParser(DocumentParser):
    """Extract searchable text from raster images with optional Tesseract OCR."""

    def supports(self, file_type: str) -> bool:
        return file_type.lower().lstrip(".") in IMAGE_EXTENSIONS

    def parse(self, file_path: str, document_id: str) -> list[DocumentBlock]:
        try:
            import pytesseract
            from PIL import Image, ImageOps
            from pytesseract import Output
        except ImportError as exc:
            raise RuntimeError(
                "Image OCR requires the optional dependencies: "
                "pip install 'docguard-rag[image]' and a Tesseract installation"
            ) from exc

        settings = self._settings()
        if settings.ocr_tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.ocr_tesseract_cmd
        try:
            with Image.open(Path(file_path)) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                data = pytesseract.image_to_data(
                    image,
                    lang=settings.ocr_lang,
                    config=settings.ocr_config,
                    output_type=Output.DICT,
                )
        except Exception as exc:
            raise ValueError(f"Unable to OCR image: {exc}") from exc

        lines: dict[tuple[int, int, int], list[tuple[str, float]]] = {}
        for index, raw_text in enumerate(data.get("text", [])):
            text = str(raw_text).strip()
            if not text:
                continue
            confidence = self._confidence(data.get("conf", [])[index])
            key = (
                int(data.get("block_num", [0])[index]),
                int(data.get("par_num", [0])[index]),
                int(data.get("line_num", [0])[index]),
            )
            lines.setdefault(key, []).append((text, confidence))

        blocks: list[DocumentBlock] = []
        for paragraph_index, words in enumerate(lines.values()):
            text = " ".join(word for word, _ in words).strip()
            if not text:
                continue
            confidence = round(sum(score for _, score in words) / len(words), 4)
            block = DocumentBlock(
                block_id=new_block_id(),
                document_id=document_id,
                block_type="paragraph",
                text=text,
                paragraph_index=paragraph_index,
                metadata={"ocr": True, "ocr_confidence": confidence},
            )
            blocks.append(block)
            blocks.extend(sentence_blocks(block))
        return blocks

    @staticmethod
    def _confidence(value) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, parsed / 100.0))

    @staticmethod
    def _settings():
        from docguard_rag.server.core.config import get_settings

        return get_settings()
