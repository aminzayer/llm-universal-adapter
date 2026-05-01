import logging
from typing import Any

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseLLMAdapter

logger = logging.getLogger(__name__)


class GeminiAdapter(BaseLLMAdapter):
    """
    Adapter for Google's Gemini language models.
    Handles interactions with the Google Generative AI API.
    """

    def __init__(self, api_key: str, model: str = "gemini-1.5-pro") -> None:
        """
        Initializes the Gemini adapter.

        Args:
            api_key (str): The API key for authenticating with Google Generative AI.
            model (str): The model to use for generation (default: 'gemini-1.5-pro').
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=model)

    @retry(
        retry=retry_if_exception_type((
            google_exceptions.ResourceExhausted,
            google_exceptions.ServiceUnavailable,
            google_exceptions.DeadlineExceeded,
            google_exceptions.InternalServerError,
        )),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a text response from the Gemini model.
        Uses exponential backoff to handle quota exhaustions and server errors.

        Args:
            prompt (str): The user's input prompt.
            **kwargs (Any): Additional parameters to pass to the API (e.g., generation_config).

        Returns:
            str: The generated textual response.
        """
        logger.debug(f"Sending request to Gemini using model: {self.model.model_name}")
        response = self.model.generate_content(prompt, **kwargs)
        return response.text

    def get_token_count(self, text: str) -> int:
        """
        Calculates the number of tokens for the given text using the Gemini API.

        Args:
            text (str): The text to tokenize.

        Returns:
            int: The number of tokens in the text.
        """
        response = self.model.count_tokens(text)
        return response.total_tokens
