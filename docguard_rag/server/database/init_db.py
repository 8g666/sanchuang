from sqlalchemy.engine import Engine

from docguard_rag.server.database import models  # noqa: F401
from docguard_rag.server.database.connection import Base, engine


def init_database(bind: Engine = engine) -> None:
    """Create the current schema for local and development databases.

    Existing production databases must be upgraded with Alembic before the
    application starts; runtime compatibility ALTER statements are intentionally
    not performed here.
    """

    Base.metadata.create_all(bind=bind)
