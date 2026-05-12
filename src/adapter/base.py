from abc import ABC, abstractmethod
from typing import Callable, Dict, Any


class BaseLLMAdapter(ABC):
    """
    Abstract base class for all LLM providers.
    Ensures a consistent interface across different models.
    """

    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def register_tool(self, name: str, func: Callable, description: str) -> None:
        """
        Registers a local Python function to be exposed to the LLM via MCP.
        """
        self.tools[name] = {"function": func, "description": description}

    @abstractmethod
    def generate_with_tools(self, prompt: str) -> str:
        """
        Forces the specific provider implementation to handle function calling
        using the registered tools.
        """
        pass

    @abstractmethod
    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a text response from the underlying LLM.
        """
        pass

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """
        Calculates the number of tokens for the given text.
        """
        pass
