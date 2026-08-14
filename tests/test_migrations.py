from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from docguard_rag.server.core.config import get_settings


def test_packaged_alembic_migration_creates_schema(tmp_path, monkeypatch):
    database_path = tmp_path / "migration.db"
    database_url = f"sqlite:///{database_path}"
    monkeypatch.setenv("DOCGUARD_DATABASE_URL", database_url)
    get_settings.cache_clear()

    migration_dir = Path(__file__).parents[1] / "docguard_rag" / "migrations"
    config = Config(str(migration_dir / "alembic.ini"))
    config.set_main_option("script_location", str(migration_dir))
    command.upgrade(config, "head")

    engine = create_engine(database_url)
    tables = set(inspect(engine).get_table_names())
    with engine.connect() as connection:
        revision = connection.execute(text("SELECT version_num FROM alembic_version")).scalar()
    engine.dispose()
    get_settings.cache_clear()

    assert revision == "0001_initial"
    assert {
        "documents",
        "document_blocks",
        "document_chunks",
        "document_vectors",
        "review_jobs",
        "document_issues",
        "jobs",
        "query_logs",
    } <= tables
