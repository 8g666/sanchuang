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
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    indexes = {
        index["name"]
        for table in ("document_chunks", "document_vectors")
        for index in inspector.get_indexes(table)
    }
    support_indexes = {
        index["name"]
        for table in ("support_sessions", "support_messages")
        for index in inspector.get_indexes(table)
    }
    with engine.connect() as connection:
        revision = connection.execute(text("SELECT version_num FROM alembic_version")).scalar()
    engine.dispose()
    get_settings.cache_clear()

    assert revision == "0003_support_sessions"
    assert {
        "documents",
        "document_blocks",
        "document_chunks",
        "document_vectors",
        "review_jobs",
        "document_issues",
        "jobs",
        "query_logs",
        "support_sessions",
        "support_messages",
    } <= tables
    assert {"ix_document_chunks_created_at", "ix_document_vectors_created_at"} <= indexes
    assert {
        "ix_support_sessions_session_id",
        "ix_support_sessions_tenant_id",
        "ix_support_messages_session_seq",
    } <= support_indexes


def test_upgrade_adds_corpus_indexes_to_a_database_created_before_them(tmp_path, monkeypatch):
    database_path = tmp_path / "existing.db"
    database_url = f"sqlite:///{database_path}"
    monkeypatch.setenv("DOCGUARD_DATABASE_URL", database_url)
    get_settings.cache_clear()

    migration_dir = Path(__file__).parents[1] / "docguard_rag" / "migrations"
    config = Config(str(migration_dir / "alembic.ini"))
    config.set_main_option("script_location", str(migration_dir))
    command.upgrade(config, "0001_initial")

    # 0001 creates the schema from SQLAlchemy metadata, so a database stamped there today
    # already has the indexes. Drop them to reproduce one stamped before they existed;
    # create_all(checkfirst=True) skips existing tables entirely, indexes included.
    engine = create_engine(database_url)
    with engine.begin() as connection:
        for name in ("ix_document_chunks_created_at", "ix_document_vectors_created_at"):
            connection.execute(text(f"DROP INDEX {name}"))
    engine.dispose()

    command.upgrade(config, "head")

    engine = create_engine(database_url)
    indexes = {
        index["name"]
        for table in ("document_chunks", "document_vectors")
        for index in inspect(engine).get_indexes(table)
    }
    with engine.connect() as connection:
        revision = connection.execute(text("SELECT version_num FROM alembic_version")).scalar()
    engine.dispose()
    get_settings.cache_clear()

    assert revision == "0003_support_sessions"
    assert {"ix_document_chunks_created_at", "ix_document_vectors_created_at"} <= indexes


def test_upgrade_adds_support_tables_to_a_database_created_before_them(tmp_path, monkeypatch):
    database_path = tmp_path / "existing.db"
    database_url = f"sqlite:///{database_path}"
    monkeypatch.setenv("DOCGUARD_DATABASE_URL", database_url)
    get_settings.cache_clear()

    migration_dir = Path(__file__).parents[1] / "docguard_rag" / "migrations"
    config = Config(str(migration_dir / "alembic.ini"))
    config.set_main_option("script_location", str(migration_dir))
    command.upgrade(config, "0002_corpus_created_at_index")

    # 0001 builds the baseline from current metadata, so a database stamped at 0002 today
    # already carries both tables. Drop them to reproduce one stamped before they existed.
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE support_messages"))
        connection.execute(text("DROP TABLE support_sessions"))
    engine.dispose()

    command.upgrade(config, "head")

    engine = create_engine(database_url)
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    columns = {column["name"] for column in inspector.get_columns("support_messages")}
    uniques = {
        constraint["name"] for constraint in inspector.get_unique_constraints("support_sessions")
    }
    with engine.connect() as connection:
        revision = connection.execute(text("SELECT version_num FROM alembic_version")).scalar()
    engine.dispose()

    assert revision == "0003_support_sessions"
    assert {"support_sessions", "support_messages"} <= tables
    assert {"session_id", "tenant_id", "role", "content", "intent", "request_id"} <= columns
    assert "uq_support_session" in uniques

    command.downgrade(config, "0002_corpus_created_at_index")

    engine = create_engine(database_url)
    tables = set(inspect(engine).get_table_names())
    engine.dispose()
    get_settings.cache_clear()

    assert not {"support_sessions", "support_messages"} & tables
