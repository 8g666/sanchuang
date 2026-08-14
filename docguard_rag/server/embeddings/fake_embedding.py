import hashlib
import math
import re

from docguard_rag.server.embeddings.base import EmbeddingProvider


class FakeEmbedding(EmbeddingProvider):
    def __init__(self, dimension: int = 8):
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:

        return [self._encode(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:

        return self._encode(query)

    def _encode(self, text: str) -> list[float]:

        vector = [0.0] * self.dimension
        normalized = "".join(text.lower().split())
        ascii_words = re.findall(r"[a-z0-9_]+", normalized)
        features = list(normalized)
        features.extend(
            normalized[index : index + 2] for index in range(max(0, len(normalized) - 1))
        )
        features.extend(ascii_words)

        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self.dimension
            vector[bucket] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
