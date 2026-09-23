import httpx

from docguard_rag.server.core.provider_errors import ProviderError
from docguard_rag.server.llms.base import LLMProvider


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
    ):
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.model_name = model_name
        self.endpoint = endpoint
        self.temperature = temperature
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
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": response_schema,
            }
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
