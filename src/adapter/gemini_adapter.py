import logging
from typing import Any, Optional

from google import genai
from google.genai import types  # pyright: ignore[reportMissingImports]
from google.genai.errors import APIError  # pyright: ignore[reportMissingImports]
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from adapter.base import BaseLLMAdapter
from config import settings

logger = logging.getLogger(__name__)


class GeminiAdapter(BaseLLMAdapter):
    """
    Adapter for Google's Gemini language models.
    Handles interactions using the new official google-genai SDK architecture.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash") -> None:
        """
        Initializes the Gemini adapter.

        Args:
            api_key (Optional[str]): The API key for authenticating. Defaults to settings.
            model (str): The model to use for generation (default: 'gemini-2.5-flash').
        """
        super().__init__()
        api_key = api_key or settings.gemini_api_key
        if not api_key:
            raise ValueError("Gemini API key is missing. Please set GEMINI_API_KEY environment variable or pass it directly.")

        # The new SDK instantiates an isolated Client rather than mutating global state
        self.client = genai.Client(api_key=api_key)
        self.model_name = model

    @retry(
        retry=retry_if_exception_type((APIError,)),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a text response from the Gemini model.
        Uses exponential backoff to handle API errors and quota exhaustions.

        Args:
            prompt (str): The user's input prompt.
            **kwargs (Any): Additional parameters to pass to the API configuration.

        Returns:
            str: The generated textual response.
        """
        # Map dynamic kwargs to the strict SDK config structure
        generation_config = kwargs.pop("config", None)
        if not generation_config:
            temperature = kwargs.pop("temperature", settings.default_temperature)
            generation_config = types.GenerateContentConfig(temperature=temperature, **kwargs)

        logger.debug(f"Sending request to Gemini using model: {self.model_name}")

        response = self.client.models.generate_content(model=self.model_name, contents=prompt, config=generation_config)

        if response.text is None:
            raise ValueError("The model returned an empty response. Check safety filters or prompt validity.")

        return response.text

    def get_token_count(self, text: str) -> int:
        """
        Calculates the number of tokens for the given text using the updated SDK.

        Args:
            text (str): The text to tokenize.

        Returns:
            int: The number of tokens in the text.
        """
        response = self.client.models.count_tokens(model=self.model_name, contents=text)
        return response.total_tokens or 0

    def generate_with_tools(self, prompt: str) -> str:
        """
        Executes a prompt allowing the LLM to utilize registered tools.
        """
        raise NotImplementedError("Tool calling logic is not yet implemented for this adapter.")
