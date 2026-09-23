import math

from docguard_rag.server.core.retrieval_index import snapshot_for
from docguard_rag.server.core.tokenizer import tokenize
from docguard_rag.server.database.connection import SessionLocal
from docguard_rag.server.retrievers.base import Retriever


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
        query_terms = tokenize(query)
        if not query_terms:
            return []

        snapshot = snapshot_for(self.session_factory or SessionLocal)

        document_ids = filters.get("document_id") or filters.get("document_ids")
        if document_ids:
            if isinstance(document_ids, str):
                document_ids = [document_ids]
            # A set mirrors SQL IN semantics: repeats and unknown ids contribute nothing.
            wanted = set(document_ids)
            candidate_ids = {
                chunk_id
                for document_id in wanted
                for chunk_id in snapshot.documents.get(document_id, ())
            }
            present = [snapshot.stats[item] for item in wanted if item in snapshot.stats]
            total = sum(item[0] for item in present)
            total_length = sum(item[1] for item in present)
        else:
            candidate_ids = None
            total = snapshot.total
            total_length = snapshot.total_length

        if not total:
            return []

        # Statistics stay scoped to the candidate set so scores match a per-query rebuild.
        average_length = total_length / total or 1.0

        frequencies: dict[str, dict[str, int]] = {}
        inverse_document_frequency: dict[str, float] = {}
        for term in set(query_terms):
            posting = snapshot.postings.get(term) or {}
            if candidate_ids is not None:
                posting = {
                    chunk_id: frequency
                    for chunk_id, frequency in posting.items()
                    if chunk_id in candidate_ids
                }
            count = len(posting)
            frequencies[term] = posting
            inverse_document_frequency[term] = math.log(1 + (total - count + 0.5) / (count + 0.5))

        # Only chunks holding at least one query term can score above zero.
        matched = {chunk_id for posting in frequencies.values() for chunk_id in posting}

        results: list[dict] = []
        for chunk_id in sorted(matched, key=lambda item: snapshot.chunks[item].order):
            chunk = snapshot.chunks[chunk_id]
            score = 0.0
            for term in query_terms:
                frequency = frequencies[term].get(chunk_id, 0)
                if not frequency:
                    continue
                denominator = frequency + 1.5 * (0.25 + 0.75 * chunk.length / average_length)
                score += inverse_document_frequency[term] * frequency * 2.5 / denominator
            if score <= 0:
                continue
            vector = snapshot.vectors.get(chunk_id)
            # Copy: the snapshot is shared and immutable, and setdefault below mutates.
            metadata = dict(vector["metadata"]) if vector else {}
            metadata.setdefault("document_id", chunk.document_id)
            metadata.setdefault("tenant_id", "default")
            metadata.setdefault("block_id", chunk.block_id)
            metadata.setdefault("text", chunk.text)
            metadata.setdefault("chunk_index", chunk.chunk_index)
            if not self._matches_metadata(metadata, filters):
                continue
            results.append({"chunk_id": chunk_id, "score": float(score), "metadata": metadata})

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]

    @staticmethod
    def _matches_metadata(metadata: dict, filters: dict) -> bool:
        ignored = {"document_ids"}
        for key, expected in filters.items():
            if key in ignored or expected is None:
                continue
            actual = metadata.get(key)
            if isinstance(expected, list | tuple | set):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True
