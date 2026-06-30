from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, AsyncGenerator


class BaseLLMAdapter(ABC):
    """
    Abstract base class for all LLM providers.
    Ensures a consistent interface across different models.
    """

    def __init__(self) -> None:
        # Correctly type the nested dictionary structure
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.redis_client: Any = None

    def register_tool(self, name: str, func: Callable[..., Any], description: str, requires_approval: bool = False) -> None:
        """
        Registers a local Python function to be exposed to the LLM via MCP.
        """
        self.tools[name] = {
            "function": func,
            "description": description,
            "requires_approval": requires_approval
        }

    def set_redis_client(self, redis_client: Any) -> None:
        """
        Sets the active Redis client on this adapter.
        """
        self.redis_client = redis_client

    async def resume_with_tools(self, state: Any) -> str:
        """
        Resumes a suspended tool execution using the saved state.
        """
        raise NotImplementedError("This adapter does not support tool execution resumption.")

    @abstractmethod
    async def generate_with_tools(self, prompt: str) -> str:
        """
        Forces the specific provider implementation to handle function calling
        using the registered tools.
        """
        pass

    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a text response from the underlying LLM.
        """
        pass

    @abstractmethod
    def agenerate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """
        Asynchronously generates a streamed text response from the LLM.
        """
        pass

    @abstractmethod
    async def get_token_count(self, text: str) -> int:
        """
        Calculates the number of tokens for the given text.
        """
        pass

