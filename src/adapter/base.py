from abc import ABC, abstractmethod
from typing import Any


class BaseLLMAdapter(ABC):
    """
    Abstract base class for all LLM providers.
    Ensures a consistent interface across different models.
    """

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
