"""Read-through cache of committed chunks and vectors for the in-process retrieval stack.

The SQL tables are the only source of truth: a snapshot is rebuilt only when a cheap
fingerprint query shows the committed corpus changed. That keeps writes made by another
process visible while never surfacing vectors from a rolled-back transaction.
"""

import json
import threading
import weakref
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from docguard_rag.server.core.tokenizer import tokenize
from docguard_rag.server.core.vector_matrix import VectorMatrix, build_matrices
from docguard_rag.server.database.models import DocumentChunkTable, DocumentVectorTable

Fingerprint = tuple[int, int, object, int, int, object]


@dataclass(frozen=True)
class ChunkEntry:
    order: int
    chunk_id: str
    document_id: str
    block_id: str
    chunk_index: int
    length: int
    text: str


@dataclass(frozen=True)
class CorpusSnapshot:
    """Immutable corpus view. Replaced wholesale, never mutated in place."""

    fingerprint: Fingerprint
    vectors: dict[str, dict]
    # Pre-normalised view of `vectors`, keyed by dimension. Built once per rebuild so a
    # search never converts a stored Python list again.
    matrices: dict[int, VectorMatrix]
    chunks: dict[str, ChunkEntry]
    postings: dict[str, dict[str, int]]
    documents: dict[str, tuple[str, ...]]
    stats: dict[str, tuple[int, int]]
    total: int
    total_length: int


def _fingerprint(db: Session) -> Fingerprint:
    """One round trip: row count, max primary key and max insert time of both tables.

    All three are needed. Inserts raise the count and the max id, deletes lower the
    count, but an equal-size replacement moves neither: SQLite reuses rowids once the
    highest row is gone, so a one-chunk document replaced by another one-chunk document
    keeps COUNT(*) and MAX(id) identical. Only MAX(created_at) always advances, because
    every insert is stamped with datetime.now.
    """

    statement = select(
        select(func.count(DocumentChunkTable.id)).scalar_subquery(),
        select(func.max(DocumentChunkTable.id)).scalar_subquery(),
        select(func.max(DocumentChunkTable.created_at)).scalar_subquery(),
        select(func.count(DocumentVectorTable.id)).scalar_subquery(),
        select(func.max(DocumentVectorTable.id)).scalar_subquery(),
        select(func.max(DocumentVectorTable.created_at)).scalar_subquery(),
    )
    row = db.execute(statement).one()
    return (row[0] or 0, row[1] or 0, row[2], row[3] or 0, row[4] or 0, row[5])


def _build(db: Session, fingerprint: Fingerprint) -> CorpusSnapshot:
    chunk_rows = db.execute(
        select(
            DocumentChunkTable.chunk_id,
            DocumentChunkTable.document_id,
            DocumentChunkTable.block_id,
            DocumentChunkTable.chunk_index,
            DocumentChunkTable.text,
        ).order_by(DocumentChunkTable.id)
    ).all()
    vector_rows = db.execute(
        select(
            DocumentVectorTable.chunk_id,
            DocumentVectorTable.vector,
            DocumentVectorTable.metadata_json,
        )
    ).all()

    vectors: dict[str, dict] = {}
    for chunk_id, vector, metadata_json in vector_rows:
        metadata = json.loads(metadata_json or "{}")
        metadata.setdefault("tenant_id", "default")
        vectors[chunk_id] = {"vector": json.loads(vector), "metadata": metadata}

    chunks: dict[str, ChunkEntry] = {}
    postings: dict[str, dict[str, int]] = {}
    documents: dict[str, list[str]] = {}
    stats: dict[str, list[int]] = {}
    total_length = 0

    for order, (chunk_id, document_id, block_id, chunk_index, text) in enumerate(chunk_rows):
        tokens = tokenize(text)
        length = len(tokens)
        chunks[chunk_id] = ChunkEntry(
            order, chunk_id, document_id, block_id, chunk_index, length, text
        )
        documents.setdefault(document_id, []).append(chunk_id)
        counters = stats.setdefault(document_id, [0, 0])
        counters[0] += 1
        counters[1] += length
        total_length += length
        for term, frequency in Counter(tokens).items():
            postings.setdefault(term, {})[chunk_id] = frequency

    return CorpusSnapshot(
        fingerprint=fingerprint,
        vectors=vectors,
        matrices=build_matrices(vectors),
        chunks=chunks,
        postings=postings,
        documents={key: tuple(value) for key, value in documents.items()},
        stats={key: (value[0], value[1]) for key, value in stats.items()},
        total=len(chunks),
        total_length=total_length,
    )


class RetrievalIndex:
    def __init__(self):
        self.snapshot: CorpusSnapshot | None = None
        self.rebuilds = 0
        self._lock = threading.Lock()

    def ensure_fresh(self, db: Session) -> CorpusSnapshot:
        fingerprint = _fingerprint(db)
        cached = self.snapshot
        if cached is not None and cached.fingerprint == fingerprint:
            return cached

        with self._lock:
            # Re-read under the lock so concurrent rebuilders collapse into one and a
            # snapshot built while we waited is not overwritten with a staler one.
            fingerprint = _fingerprint(db)
            cached = self.snapshot
            if cached is not None and cached.fingerprint == fingerprint:
                return cached
            built = _build(db, fingerprint)
            self.snapshot = built
            self.rebuilds += 1
            return built


_REGISTRY: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_REGISTRY_LOCK = threading.Lock()


def _engine_of(bind):
    return getattr(bind, "engine", bind)


def _index_for(db: Session) -> RetrievalIndex:
    """Resolve the index shared by every retrieval stack bound to this engine."""

    engine = _engine_of(db.get_bind())
    index = _REGISTRY.get(engine)
    if index is None:
        with _REGISTRY_LOCK:
            index = _REGISTRY.get(engine)
            if index is None:
                index = RetrievalIndex()
                _REGISTRY[engine] = index
    return index


def snapshot_for(session_factory) -> CorpusSnapshot:
    with session_factory() as db:
        return _index_for(db).ensure_fresh(db)


def index_for_engine(engine) -> RetrievalIndex | None:
    return _REGISTRY.get(_engine_of(engine))
