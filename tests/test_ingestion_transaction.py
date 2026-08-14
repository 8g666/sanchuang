import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from docguard_rag.server.database.connection import Base
from docguard_rag.server.database.models import (
    DocumentBlockTable,
    DocumentChunkTable,
    DocumentTable,
    DocumentVectorTable,
)
from docguard_rag.server.services.ingestion_service import IngestionService
from docguard_rag.server.storage.file_storage import FileStorage


class FailingIndexService:
    def build_index(self, db, chunks):
        raise RuntimeError("embedding failed")


def test_ingestion_rolls_back_all_database_records(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'rollback.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    storage = FileStorage(tmp_path / "uploads")
    service = IngestionService(storage=storage, index_service=FailingIndexService())

    with session_factory() as db:
        with pytest.raises(RuntimeError, match="embedding failed"):
            service.ingest_bytes(
                db,
                filename="document.txt",
                content="需要回滚的内容。".encode(),
            )

        assert db.query(DocumentTable).count() == 0
        assert db.query(DocumentBlockTable).count() == 0
        assert db.query(DocumentChunkTable).count() == 0
        assert db.query(DocumentVectorTable).count() == 0

    assert list((tmp_path / "uploads").iterdir()) == []
    engine.dispose()
