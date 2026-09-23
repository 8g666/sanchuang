import json

import httpx

from docguard_rag.server.core.provider_errors import ProviderError
from docguard_rag.server.llms.base import LLMProvider

_SCHEMA_INSTRUCTION = (
    "\n\nRespond with one JSON object and no other text. It must match this JSON Schema "
    "exactly, using these field names:\n{schema}"
)


def _with_schema_instruction(messages: list[dict], response_schema: dict) -> list[dict]:
    """Describe the expected shape in the prompt instead of in response_format.

    Only OpenAI and a few others accept `response_format=json_schema`; the wider
    OpenAI-compatible baseline stops at `json_object`, which guarantees parseable JSON but
    says nothing about field names. Callers name their fields only in the schema, so
    without this the model guesses and every caller silently falls back to its
    deterministic path.
    """

    schema = response_schema.get("schema", response_schema)
    instruction = _SCHEMA_INSTRUCTION.format(
        schema=json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    )
    # Copied rather than mutated: callers build their message list once per request and
    # a retry would otherwise stack a second copy of the instruction on the end.
    last = messages[-1]
    return [*messages[:-1], {**last, "content": last["content"] + instruction}]


class CompatibleLLM(LLMProvider):
    """Chat provider for OpenAI-compatible HTTP APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        endpoint: str = "/chat/completions",
        timeout: float = 60.0,
        temperature: float = 0.0,
        strict_schema: bool = False,
    ):
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.model_name = model_name
        self.endpoint = endpoint
        self.temperature = temperature
        self.strict_schema = strict_schema
        self._base_url = base_url.rstrip("/")
        self._headers = headers
        self._timeout = timeout
        self._client: httpx.Client | None = None

    @property
    def fingerprint(self) -> tuple:
        """Identifies the model whose completions this provider produces."""

        return ("compatible_api", self.model_name)

    @property
    def client(self) -> httpx.Client:
        # Built on first use rather than in __init__: providers are constructed at import
        # time and closed at shutdown, and httpx refuses to reuse a closed client.
        if self._client is None:
            self._client = httpx.Client(
                base_url=self._base_url,
                headers=self._headers,
                timeout=self._timeout,
            )
        return self._client

    @client.setter
    def client(self, client: httpx.Client) -> None:
        self._client = client

    def generate(
        self,
        messages: list[dict],
        response_schema: dict | None = None,
    ) -> dict:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        if response_schema:
            if self.strict_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": response_schema,
                }
            else:
                payload["response_format"] = {"type": "json_object"}
                payload["messages"] = _with_schema_instruction(messages, response_schema)
        try:
            response = self.client.post(self.endpoint, json=payload)
            response.raise_for_status()
            body = response.json()
            message = body["choices"][0]["message"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(f"LLM provider request failed: {exc}") from exc
        return {
            "content": message.get("content", ""),
            "tool_calls": message.get("tool_calls"),
            "usage": body.get("usage", {}),
            "model": body.get("model", self.model_name),
        }

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
