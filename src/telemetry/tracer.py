import json
import logging
import time
from typing import Any, AsyncGenerator, Callable, Optional

from adapter.base import BaseLLMAdapter

logger = logging.getLogger(__name__)


class ObservabilityMiddleware(BaseLLMAdapter):
    """
    Middleware that intercepts all outgoing LLM requests and incoming responses.
    Logs latency, detailed token usage, caching status, and adapter configuration
    in a structured JSON format compatible with ELK or Prometheus.
    """

    def __init__(self, adapter: BaseLLMAdapter, provider: str) -> None:
        super().__init__()
        self.adapter = adapter
        self.provider = provider
        # Sync tools dictionary from the inner adapter
        self.tools = self.adapter.tools

    def register_tool(self, name: str, func: Callable[..., Any], description: str) -> None:
        """
        Delegates tool registration to the underlying adapter and syncs the tools dictionary.
        """
        self.adapter.register_tool(name, func, description)
        self.tools = self.adapter.tools

    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        start_time = time.perf_counter()
        error: Optional[str] = None
        response = ""
        try:
            response = await self.adapter.generate_response(prompt, **kwargs)
            return response
        except Exception as e:
            error = str(e)
            raise
        finally:
            latency = time.perf_counter() - start_time
            await self._log_telemetry("generate_response", prompt, response, latency, error)

    async def agenerate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        start_time = time.perf_counter()
        error: Optional[str] = None
        response_chunks = []
        try:
            async for chunk in self.adapter.agenerate_stream(prompt, **kwargs):
                response_chunks.append(chunk)
                yield chunk
        except Exception as e:
            error = str(e)
            raise
        finally:
            latency = time.perf_counter() - start_time
            response = "".join(response_chunks)
            await self._log_telemetry("agenerate_stream", prompt, response, latency, error)

    async def generate_with_tools(self, prompt: str) -> str:
        start_time = time.perf_counter()
        error: Optional[str] = None
        response = ""
        try:
            response = await self.adapter.generate_with_tools(prompt)
            return response
        except Exception as e:
            error = str(e)
            raise
        finally:
            latency = time.perf_counter() - start_time
            await self._log_telemetry("generate_with_tools", prompt, response, latency, error)

    async def get_token_count(self, text: str) -> int:
        return await self.adapter.get_token_count(text)

    async def _log_telemetry(self, operation: str, prompt: str, response: str, latency: float, error: Optional[str]) -> None:
        """
        Computes token metrics and logs the final JSON structured telemetry data.
        """
        try:
            prompt_tokens = await self.adapter.get_token_count(prompt) if prompt else 0
            completion_tokens = await self.adapter.get_token_count(response) if response else 0
            total_tokens = prompt_tokens + completion_tokens
        except Exception:
            prompt_tokens, completion_tokens, total_tokens = 0, 0, 0

        # Heuristic for cache hit: extremely low latency when a semantic cache is present
        cache_status = "DISABLED"
        if getattr(self.adapter, "semantic_cache", None) is not None:
            if latency < 0.05 and error is None:
                cache_status = "HIT"
            else:
                cache_status = "MISS"

        model_name = getattr(self.adapter, "model", getattr(self.adapter, "model_name", "unknown"))

        log_entry = {
            "event": "llm_invocation",
            "provider": self.provider,
            "model": model_name,
            "operation": operation,
            "latency_sec": round(latency, 4),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cache_status": cache_status,
            "error": error,
        }
        logger.info(json.dumps(log_entry))
