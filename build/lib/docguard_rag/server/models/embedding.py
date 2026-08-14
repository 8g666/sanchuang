from typing import TypedDict


class EmbeddingRecord(TypedDict):
    chunk_id: str
    vector: list[float]
    metadata: dict
