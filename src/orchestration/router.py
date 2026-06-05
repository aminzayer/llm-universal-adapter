import logging
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from src.adapter.base import BaseLLMAdapter
from src.adapter.factory import LLMAdapterFactory

logger = logging.getLogger(__name__)


class RouterManager(BaseLLMAdapter):
    """
    A router manager that wraps the LLMAdapterFactory to provide a strict failover mechanism.
    If the primary adapter fails (e.g., exhausts its tenacity retries due to APIError or RateLimitError),
    it silently intercepts the failure and re-routes the exact prompt and MCP tools 
    to a secondary fallback adapter.
    """

    def __init__(
        self,
        primary_provider: str,
        fallback_provider: str,
        primary_kwargs: Optional[Dict[str, Any]] = None,
        fallback_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the RouterManager with a primary and fallback LLM provider.

        Args:
            primary_provider (str): The name of the primary provider (e.g., 'openai').
            fallback_provider (str): The name of the fallback provider (e.g., 'gemini').
            primary_kwargs (Optional[Dict[str, Any]]): Instantiation arguments for the primary adapter.
            fallback_kwargs (Optional[Dict[str, Any]]): Instantiation arguments for the fallback adapter.
        """
        super().__init__()
        self.primary_name = primary_provider
        self.fallback_name = fallback_provider

        # Instantiate adapters via the factory wrapper
        self.primary_adapter = LLMAdapterFactory.create_adapter(primary_provider, **(primary_kwargs or {}))
        self.fallback_adapter = LLMAdapterFactory.create_adapter(fallback_provider, **(fallback_kwargs or {}))

    def register_tool(self, name: str, func: Callable[..., Any], description: str) -> None:
        """
        Registers a tool on the router and explicitly propagates it to both the 
        primary and fallback adapters to ensure seamless failover execution.
        """
        super().register_tool(name, func, description)
        self.primary_adapter.register_tool(name, func, description)
        self.fallback_adapter.register_tool(name, func, description)

    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Attempts to generate a response using the primary adapter.
        Silently falls back to the secondary adapter upon failure.
        """
        try:
            return await self.primary_adapter.generate_response(prompt, **kwargs)
        except Exception as e:
            logger.warning(f"Primary adapter '{self.primary_name}' failed with error: {e}. "
                           f"Initiating silent failover to secondary adapter '{self.fallback_name}'.")
            return await self.fallback_adapter.generate_response(prompt, **kwargs)

    async def agenerate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """
        Attempts to stream a response using the primary adapter.
        Falls back to the secondary adapter if the primary fails before yielding any chunks.
        """
        first_chunk_yielded = False
        try:
            stream = self.primary_adapter.agenerate_stream(prompt, **kwargs)
            async for chunk in stream:
                first_chunk_yielded = True
                yield chunk
        except Exception as e:
            if first_chunk_yielded:
                # If chunks were already yielded, a seamless failover is impossible
                # without duplicating the start of the stream on the client side.
                logger.error(f"Stream failed mid-flight on primary adapter '{self.primary_name}': {e}")
                raise

            logger.warning(f"Primary adapter '{self.primary_name}' failed before streaming: {e}. "
                           f"Initiating silent failover to secondary adapter '{self.fallback_name}'.")
            async for chunk in self.fallback_adapter.agenerate_stream(prompt, **kwargs):
                yield chunk

    async def generate_with_tools(self, prompt: str) -> str:
        """
        Attempts to generate a response with MCP tools using the primary adapter.
        Silently falls back to the secondary adapter upon failure.
        """
        try:
            return await self.primary_adapter.generate_with_tools(prompt)
        except Exception as e:
            logger.warning(f"Primary adapter '{self.primary_name}' failed during tool execution: {e}. "
                           f"Initiating silent failover to secondary adapter '{self.fallback_name}'.")
            return await self.fallback_adapter.generate_with_tools(prompt)

    async def get_token_count(self, text: str) -> int:
        """
        Calculates the token count. This uses the primary adapter directly without failover, 
        as tokenization is a local operation for most providers (e.g. tiktoken).
        """
        return await self.primary_adapter.get_token_count(text)
