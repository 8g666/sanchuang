import json
import math
import re
from collections import Counter

from docguard_rag.server.database.connection import SessionLocal
from docguard_rag.server.database.models import DocumentChunkTable, DocumentVectorTable
from docguard_rag.server.retrievers.base import Retriever


def tokenize(text: str) -> list[str]:
    """Tokenize English terms and Chinese unigrams/bigrams without external models."""

    normalized = text.lower()
    words = re.findall(r"[a-z0-9_./-]+", normalized)
    chinese_runs = re.findall(r"[\u3400-\u9fff]+", normalized)
    chinese_tokens: list[str] = []
    for run in chinese_runs:
        chinese_tokens.extend(run)
        chinese_tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return words + chinese_tokens


class LexicalRetriever(Retriever):
    """Small in-process BM25 retriever backed by persisted chunks."""

    def __init__(self, session_factory=None):
        self.session_factory = session_factory

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        if top_k <= 0:
            return []

        filters = filters or {}
        with (self.session_factory or SessionLocal)() as db:
            statement = db.query(DocumentChunkTable)
            document_ids = filters.get("document_id") or filters.get("document_ids")
            if document_ids:
                if isinstance(document_ids, str):
                    document_ids = [document_ids]
                statement = statement.filter(DocumentChunkTable.document_id.in_(document_ids))
            chunks = statement.all()
            vector_rows = {row.chunk_id: row for row in db.query(DocumentVectorTable).all()}

        if not chunks:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []
        tokenized = [tokenize(chunk.text) for chunk in chunks]
        document_frequency = Counter(token for tokens in tokenized for token in set(tokens))
        average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized) or 1.0
        total = len(chunks)
        results: list[dict] = []

        for chunk, tokens in zip(chunks, tokenized):
            frequencies = Counter(tokens)
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                frequency_count = document_frequency.get(term, 0)
                inverse_document_frequency = math.log(
                    1 + (total - frequency_count + 0.5) / (frequency_count + 0.5)
                )
                denominator = frequency + 1.5 * (0.25 + 0.75 * len(tokens) / average_length)
                score += inverse_document_frequency * frequency * 2.5 / denominator
            if score <= 0:
                continue
            vector_row = vector_rows.get(chunk.chunk_id)
            metadata = json.loads(vector_row.metadata_json or "{}") if vector_row else {}
            metadata.setdefault("document_id", chunk.document_id)
            metadata.setdefault("tenant_id", "default")
            metadata.setdefault("block_id", chunk.block_id)
            metadata.setdefault("text", chunk.text)
            metadata.setdefault("chunk_index", chunk.chunk_index)
            if not self._matches_metadata(metadata, filters):
                continue
            results.append(
                {"chunk_id": chunk.chunk_id, "score": float(score), "metadata": metadata}
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]

    @staticmethod
    def _matches_metadata(metadata: dict, filters: dict) -> bool:
        ignored = {"document_ids"}
        for key, expected in filters.items():
            if key in ignored or expected is None:
                continue
            actual = metadata.get(key)
            if isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True
