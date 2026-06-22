import json
import logging
from typing import Any, AsyncGenerator, Optional

import aiohttp
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from adapter.base import BaseLLMAdapter

logger = logging.getLogger(__name__)


class LocalModelError(RuntimeError):
    """Raised when the local model endpoint returns an unrecoverable error."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"Local model returned HTTP {status}: {body}")
        self.status = status
        self.body = body


def _is_retryable(exc: BaseException) -> bool:
    """Tenacity predicate: only 5xx and connection errors are retryable."""
    if isinstance(exc, LocalModelError):
        return exc.status >= 500
    if isinstance(exc, aiohttp.ClientConnectionError):
        return True
    return False


class LocalModelAdapter(BaseLLMAdapter):
    """
    Adapter for local LLM servers that expose an OpenAI-compatible HTTP API.

    Targets vLLM, Ollama (when started with the OpenAI-compatible shim), LM
    Studio, and any other server speaking ``POST /v1/chat/completions``.
    Designed for hardware-accelerated local environments such as Apple
    Silicon (Metal) and AWS Graviton (Graviton3/Graviton4 CPUs).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Initializes the local model adapter.

        Args:
            base_url: Root URL of the local server (e.g. ``http://localhost:11434/v1``).
            model: Local model identifier (e.g. ``llama3.1``, ``qwen2.5:7b``).
            api_key: Optional bearer token for servers that require auth (vLLM does, Ollama ignores it).
            timeout: Per-request timeout in seconds.
        """
        if not base_url:
            raise ValueError("base_url is required for LocalModelAdapter")
        if not model:
            raise ValueError("model is required for LocalModelAdapter")

        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def aclose(self) -> None:
        """Closes the underlying HTTP session. Safe to call multiple times."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Returns a lazily-initialised, shared aiohttp ClientSession."""
        if self._session is None or self._session.closed:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._session = aiohttp.ClientSession(timeout=self.timeout, headers=headers)
        return self._session

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        """Tenacity predicate: only 5xx and connection errors are retryable."""
        if isinstance(exc, LocalModelError):
            return exc.status >= 500
        if isinstance(exc, aiohttp.ClientConnectionError):
            return True
        return False

    @retry(
        retry=retry_if_exception(_is_retryable),  # type: ignore[arg-type]
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        POSTs the JSON payload to ``/v1/chat/completions`` and returns the
        decoded response body. Retries on 5xx and connection errors.
        """
        url = f"{self.base_url}/chat/completions"
        session = await self._get_session()
        async with session.post(url, json=payload, headers=self._headers()) as resp:
            body = await resp.text()
            if resp.status >= 500:
                # 5xx errors are transient — let tenacity decide.
                raise LocalModelError(resp.status, body[:500])
            if resp.status >= 400:
                # 4xx errors are caller bugs — do not retry.
                raise LocalModelError(resp.status, body[:500])
            try:
                return json.loads(body)
            except json.JSONDecodeError as e:
                raise LocalModelError(resp.status, f"invalid JSON: {e}") from e

    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a text response from the local model.

        Args:
            prompt: User prompt (sent as the single ``user`` turn).
            **kwargs: Additional OpenAI-compatible parameters (temperature, max_tokens, ...).

        Returns:
            The generated text content.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        payload.update(kwargs)

        logger.debug(f"Sending async request to local model '{self.model}' at {self.base_url}")
        response = await self._post_json(payload)
        choices = response.get("choices") or []
        if not choices:
            raise LocalModelError(200, f"empty choices in response: {response}")
        message = choices[0].get("message") or {}
        return message.get("content") or ""

    async def agenerate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """
        Streams a response from the local model using SSE framing.
        """
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        payload.update(kwargs)

        logger.debug(f"Sending async streaming request to local model '{self.model}' at {self.base_url}")
        session = await self._get_session()
        async with session.post(url, json=payload, headers=self._headers()) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise LocalModelError(resp.status, body[:500])
            # Each SSE event is one or more lines of "data: {...}", separated by blank lines.
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content

    async def get_token_count(self, text: str) -> int:
        """
        Approximate local token count.

        The whole point of routing traffic to this adapter is to skip remote
        token-counting APIs; a rough character-based heuristic is good enough
        for telemetry and windowing.
        """
        if not text:
            return 0
        return max(1, len(text) // 4)

    async def generate_with_tools(self, prompt: str) -> str:
        """
        Local models have inconsistent function-calling support, so this adapter
        deliberately does not implement it. Use a cloud adapter instead.
        """
        raise NotImplementedError(
            "LocalModelAdapter does not support tool calling. "
            "Route requests that require tools to a cloud adapter (openai/gemini)."
        )