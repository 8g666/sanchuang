from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        response_schema: dict | None = None,
    ) -> dict:
        raise NotImplementedError
