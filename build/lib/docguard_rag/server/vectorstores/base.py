from abc import ABC, abstractmethod


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunk_id: str, vector: list[float], metadata: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        """Delete vectors by their public chunk identifiers."""

        raise NotImplementedError
