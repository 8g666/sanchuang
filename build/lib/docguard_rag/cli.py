import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DocGuard-RAG API service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Apply database migrations and exit",
    )
    args = parser.parse_args()

    if args.migrate:
        from alembic import command
        from alembic.config import Config

        migration_dir = Path(__file__).with_name("migrations")
        config = Config(str(migration_dir / "alembic.ini"))
        config.set_main_option("script_location", str(migration_dir))
        command.upgrade(config, "head")
        return

    import uvicorn

    uvicorn.run(
        "docguard_rag.asgi:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
