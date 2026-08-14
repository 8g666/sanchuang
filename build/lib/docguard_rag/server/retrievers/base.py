from abc import ABC, abstractmethod


class Retriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """Retrieve ranked text candidates for a query."""

        raise NotImplementedError
