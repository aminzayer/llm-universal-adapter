import logging
from typing import Any, Optional

import openai
import tiktoken
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseLLMAdapter
from config import settings

logger = logging.getLogger(__name__)


class OpenAIAdapter(BaseLLMAdapter):
    """
    Adapter for OpenAI's language models.
    Handles interactions with the OpenAI API, including generating responses
    and calculating token counts using tiktoken.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o") -> None:
        """
        Initializes the OpenAI adapter.

        Args:
            api_key (Optional[str]): The API key for authenticating with OpenAI. Defaults to settings.
            model (str): The model to use for generation (default: 'gpt-4o').
        """
        api_key = api_key or settings.openai_api_key
        if not api_key:
            raise ValueError("OpenAI API key is missing. Please set OPENAI_API_KEY environment variable or pass it directly.")

        self.client = openai.Client(api_key=api_key)
        self.model = model

    @retry(
        retry=retry_if_exception_type((
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.InternalServerError,
        )),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a text response from the OpenAI model.
        Uses exponential backoff for handling rate limits and connection errors.

        Args:
            prompt (str): The user's input prompt.
            **kwargs (Any): Additional parameters to pass to the API (e.g., temperature, max_tokens).

        Returns:
            str: The generated textual response.
        """
        if "temperature" not in kwargs:
            kwargs["temperature"] = settings.default_temperature

        logger.debug(f"Sending request to OpenAI using model: {self.model}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def get_token_count(self, text: str) -> int:
        """
        Calculates the number of tokens for the given text using tiktoken.

        Args:
            text (str): The text to tokenize.

        Returns:
            int: The number of tokens in the text.
        """
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fallback encoding if the exact model is not found
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))

    def generate_with_tools(self, prompt: str) -> str:
        """
        Executes a prompt allowing the LLM to utilize registered tools.
        """
        raise NotImplementedError("Tool calling logic is not yet implemented for this adapter.")
