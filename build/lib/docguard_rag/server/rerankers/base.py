from abc import ABC, abstractmethod


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        raise NotImplementedError
