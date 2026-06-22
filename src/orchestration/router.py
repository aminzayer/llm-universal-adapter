import json
import logging
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from adapter.base import BaseLLMAdapter
from adapter.factory import LLMAdapterFactory

logger = logging.getLogger(__name__)

# Keyword sets used by the trivial/complex heuristic. These are deliberately
# small and conservative — false positives route complex reasoning to local
# (low-stakes; we just fall back to cloud) and false negatives route trivial
# classification to cloud (slightly wasted spend, no correctness impact).
_TRIVIAL_KEYWORDS = (
    "classify",
    "categorize",
    "categorise",
    "label",
    "sentiment",
    "is this",
    "yes/no",
    "true or false",
    "spam or not",
)
_COMPLEX_KEYWORDS = (
    "reason",
    "explain",
    "plan",
    "analyze",
    "analyse",
    "compare",
    "code",
    "implement",
    "refactor",
    "design",
    "derive",
    "prove",
    "summarise",
    "summarize",
)

# Anything longer than this is treated as complex regardless of keywords.
_TRIVIAL_MAX_PROMPT_CHARS = 800


class RouterManager(BaseLLMAdapter):
    """
    A router manager that wraps the LLMAdapterFactory to provide a strict failover mechanism.

    Routing strategy:
    - If a ``local_provider`` is configured and the prompt looks trivial
      (short, classification-style), dispatch to the local adapter first.
      Local failures fall back to the primary/fallback cloud chain so a
      flaky local server never breaks user requests.
    - Complex prompts (reasoning, code, long context) and tool-calling
      requests always go through primary → fallback on the cloud adapters.
    """

    def __init__(
        self,
        primary_provider: str,
        fallback_provider: str,
        primary_kwargs: Optional[Dict[str, Any]] = None,
        fallback_kwargs: Optional[Dict[str, Any]] = None,
        local_provider: Optional[str] = None,
        local_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the RouterManager with primary, fallback, and optionally local providers.

        Args:
            primary_provider (str): Name of the primary cloud provider (e.g. 'openai').
            fallback_provider (str): Name of the fallback cloud provider (e.g. 'gemini').
            primary_kwargs (Optional[Dict[str, Any]]): Instantiation arguments for the primary adapter.
            fallback_kwargs (Optional[Dict[str, Any]]): Instantiation arguments for the fallback adapter.
            local_provider (Optional[str]): Name of the local provider (e.g. 'local'). When None, the
                router behaves exactly as before — every request goes to the cloud chain.
            local_kwargs (Optional[Dict[str, Any]]): Instantiation arguments for the local adapter
                (typically ``base_url`` and ``model``).
        """
        super().__init__()
        self.primary_name = primary_provider
        self.fallback_name = fallback_provider
        self.local_name = local_provider

        # Instantiate adapters via the factory wrapper
        self.primary_adapter = LLMAdapterFactory.create_adapter(primary_provider, **(primary_kwargs or {}))
        self.fallback_adapter = LLMAdapterFactory.create_adapter(fallback_provider, **(fallback_kwargs or {}))

        self.local_adapter: Optional[BaseLLMAdapter] = None
        if local_provider:
            self.local_adapter = LLMAdapterFactory.create_adapter(local_provider, **(local_kwargs or {}))

    def register_tool(self, name: str, func: Callable[..., Any], description: str) -> None:
        """
        Registers a tool on the router and explicitly propagates it to the
        primary, fallback, and local adapters so the router's tool list
        always matches the inner adapters.
        """
        super().register_tool(name, func, description)
        self.primary_adapter.register_tool(name, func, description)
        self.fallback_adapter.register_tool(name, func, description)
        if self.local_adapter is not None:
            self.local_adapter.register_tool(name, func, description)

    def _is_trivial(self, prompt: str) -> bool:
        """
        Returns True if the prompt looks like a cheap, high-volume task that
        a local model can handle without spending cloud API budget.

        This is a lightweight heuristic — full intent detection lives in
        ``SwarmOrchestrator.ClassifierAgent`` for the swarm path; the router
        only needs to recognise the obvious cases.
        """
        text = prompt

        # ``main.py`` serialises multi-turn requests as a JSON array of message
        # dicts. Flatten that to the joined content so keyword matching sees
        # the actual user-visible text rather than the JSON structure.
        stripped = text.strip()
        if stripped.startswith("["):
            try:
                messages = json.loads(stripped)
                if isinstance(messages, list):
                    text = " ".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
            except (ValueError, TypeError):
                pass

        if len(text) > _TRIVIAL_MAX_PROMPT_CHARS:
            return False

        lowered = text.lower()
        if any(keyword in lowered for keyword in _COMPLEX_KEYWORDS):
            return False
        if any(keyword in lowered for keyword in _TRIVIAL_KEYWORDS):
            return True
        # Very short prompts without reasoning cues are probably cheap.
        return len(text) < 120

    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Attempts to generate a response using the most appropriate adapter.

        Trivial prompts try the local adapter first; any failure falls
        through to the existing primary → fallback chain. Complex prompts
        skip the local adapter entirely.
        """
        if self.local_adapter is not None and self._is_trivial(prompt):
            try:
                logger.debug(f"Routing trivial prompt to local adapter '{self.local_name}'")
                return await self.local_adapter.generate_response(prompt, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Local adapter '{self.local_name}' failed for trivial prompt: {e}. "
                    f"Falling back to primary cloud adapter '{self.primary_name}'."
                )

        try:
            return await self.primary_adapter.generate_response(prompt, **kwargs)
        except Exception as e:
            logger.warning(f"Primary adapter '{self.primary_name}' failed with error: {e}. "
                           f"Initiating silent failover to secondary adapter '{self.fallback_name}'.")
            return await self.fallback_adapter.generate_response(prompt, **kwargs)

    async def agenerate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """
        Attempts to stream a response using the most appropriate adapter.

        Trivial prompts try the local adapter first; on local failure
        before yielding, falls through to the cloud chain (primary →
        fallback). Mid-stream local failures are re-raised because seamless
        mid-stream failover is impossible. Complex prompts go straight to
        the cloud chain.
        """
        if self.local_adapter is not None and self._is_trivial(prompt):
            first_chunk_yielded = False
            try:
                stream = self.local_adapter.agenerate_stream(prompt, **kwargs)
                async for chunk in stream:
                    first_chunk_yielded = True
                    yield chunk
                return
            except Exception as e:
                if first_chunk_yielded:
                    logger.error(f"Local stream failed mid-flight on adapter '{self.local_name}': {e}")
                    raise
                logger.warning(
                    f"Local adapter '{self.local_name}' failed before streaming trivial prompt: {e}. "
                    f"Falling back to primary cloud adapter '{self.primary_name}'."
                )

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

        Tool-calling requests always stay on the cloud chain — local models
        have inconsistent function-calling support, and ``LocalModelAdapter``
        raises ``NotImplementedError`` for this method.
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