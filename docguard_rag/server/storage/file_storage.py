import hashlib
import shutil
from pathlib import Path
from uuid import uuid4

from docguard_rag.server.core.config import get_settings


class FileStorage:
    def __init__(self, upload_dir: str | Path | None = None):
        self.upload_dir = Path(upload_dir or get_settings().upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def safe_filename(filename: str) -> str:
        name = Path(filename or "document").name
        return "".join(
            character if character.isalnum() or character in {".", "-", "_"} else "_"
            for character in name
        )

    def save(self, filename: str, file_path: str) -> str:
        safe_name = self.safe_filename(filename)
        target = self.upload_dir / f"{uuid4().hex}_{safe_name}"

        shutil.copy(file_path, target)

        return str(target)

    def save_bytes(self, filename: str, content: bytes) -> str:
        safe_name = self.safe_filename(filename)
        target = self.upload_dir / f"{uuid4().hex}_{safe_name}"
        target.write_bytes(content)
        return str(target)

    def delete(self, file_path: str) -> None:
        path = Path(file_path).resolve()
        upload_root = self.upload_dir.resolve()
        if upload_root not in path.parents:
            raise ValueError("Refusing to delete a file outside the upload directory")
        path.unlink(missing_ok=True)

    def calculate_hash(self, file_path: str) -> str:
        sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)

        return sha256.hexdigest()
