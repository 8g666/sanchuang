from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Design-document compatible alias for batch document embedding."""

        return self.embed_texts(texts)

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""

        raise NotImplementedError

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed one query."""

        raise NotImplementedError
