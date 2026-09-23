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
    parser.add_argument(
        "--reembed",
        action="store_true",
        help=(
            "Re-embed every committed chunk with the configured model and exit. "
            "Operates on DOCGUARD_DATABASE_URL (default ./docguard.db)"
        ),
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

    if args.reembed:
        from docguard_rag.local import DocGuard

        with DocGuard() as guard:
            result = guard.documents.reembed()
        print(
            f"Re-embedded {result.reembedded} chunk(s) in {result.batches} batch(es) "
            f"at dimension {result.dimension}."
        )
        if result.document_ids_needing_reindex:
            print(
                f"WARNING: {result.embedded_without_offsets} chunk(s) across "
                f"{len(result.document_ids_needing_reindex)} document(s) had no stored "
                "vector to copy citation offsets from. Reindex those documents to "
                "restore start_offset/end_offset."
            )
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
