import pytest
from unittest.mock import MagicMock, patch

import openai
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from src.adapter.openai_adapter import OpenAIAdapter
from src.adapter.gemini_adapter import GeminiAdapter


@pytest.fixture(autouse=True)
def fast_retries():
    """
    Fixture to mock `time.sleep` used by `tenacity` during backoff.
    This ensures tests involving retries run instantaneously.
    """
    with patch("tenacity.nap.time.sleep"):
        yield


# -----------------------------------------------------------------------------
# OpenAIAdapter Tests
# -----------------------------------------------------------------------------


def test_openai_adapter_missing_key() -> None:
    """Ensure OpenAIAdapter raises ValueError if API key is completely missing."""
    with patch("src.config.settings.openai_api_key", None):
        with pytest.raises(ValueError, match="OpenAI API key is missing"):
            OpenAIAdapter(api_key=None)


@patch("openai.Client")
def test_openai_adapter_generate_response_success(mock_client: MagicMock) -> None:
    """Test successful response generation without retries."""
    mock_instance = mock_client.return_value
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello OpenAI!"))]
    mock_instance.chat.completions.create.return_value = mock_response

    adapter = OpenAIAdapter(api_key="fake-key")
    response = adapter.generate_response("Hello!")

    assert response == "Hello OpenAI!"
    mock_instance.chat.completions.create.assert_called_once()


@patch("openai.Client")
def test_openai_adapter_generate_response_retry_success(mock_client: MagicMock) -> None:
    """Test that a RateLimitError triggers a retry and eventually succeeds."""
    mock_instance = mock_client.return_value
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello again!"))]

    # Fail twice with RateLimitError, then succeed
    mock_error = openai.RateLimitError(message="Rate limited", response=MagicMock(), body=None)
    mock_instance.chat.completions.create.side_effect = [
        mock_error,
        mock_error,
        mock_response,
    ]

    adapter = OpenAIAdapter(api_key="fake-key")
    response = adapter.generate_response("Hi")

    assert response == "Hello again!"
    assert mock_instance.chat.completions.create.call_count == 3


@patch("openai.Client")
def test_openai_adapter_generate_response_retry_failure(mock_client: MagicMock) -> None:
    """Test that repeated failures eventually raise the underlying exception."""
    mock_instance = mock_client.return_value
    mock_error = openai.InternalServerError(message="Server Error", response=MagicMock(), body=None)
    mock_instance.chat.completions.create.side_effect = mock_error

    adapter = OpenAIAdapter(api_key="fake-key")

    with pytest.raises(openai.InternalServerError):
        adapter.generate_response("Hi")

    # Tenacity stops after 5 attempts
    assert mock_instance.chat.completions.create.call_count == 5


@patch("tiktoken.encoding_for_model")
def test_openai_adapter_get_token_count(mock_encoding_for_model: MagicMock) -> None:
    mock_encoding = MagicMock()
    mock_encoding.encode.return_value = [1, 2, 3, 4]
    mock_encoding_for_model.return_value = mock_encoding

    adapter = OpenAIAdapter(api_key="fake-key")
    assert adapter.get_token_count("test text") == 4
    mock_encoding.encode.assert_called_once_with("test text")


# -----------------------------------------------------------------------------
# GeminiAdapter Tests
# -----------------------------------------------------------------------------


@patch("google.generativeai.configure")
@patch("google.generativeai.GenerativeModel")
def test_gemini_adapter_missing_key(mock_model: MagicMock, mock_configure: MagicMock) -> None:
    with patch("src.config.settings.gemini_api_key", None):
        with pytest.raises(ValueError, match="Gemini API key is missing"):
            GeminiAdapter(api_key=None)


@patch("google.generativeai.configure")
@patch("google.generativeai.GenerativeModel")
def test_gemini_adapter_generate_response_retry_success(mock_model_cls: MagicMock, mock_configure: MagicMock) -> None:
    mock_model_instance = mock_model_cls.return_value
    mock_response = MagicMock()
    mock_response.text = "Hello Gemini!"

    error = google_exceptions.ResourceExhausted("Quota exceeded")
    mock_model_instance.generate_content.side_effect = [error, mock_response]

    adapter = GeminiAdapter(api_key="fake-key")
    response = adapter.generate_response("Hi")

    assert response == "Hello Gemini!"
    assert mock_model_instance.generate_content.call_count == 2


@patch("google.generativeai.configure")
@patch("google.generativeai.GenerativeModel")
def test_gemini_adapter_get_token_count(mock_model_cls: MagicMock, mock_configure: MagicMock) -> None:
    mock_model_instance = mock_model_cls.return_value
    mock_response = MagicMock()
    mock_response.total_tokens = 10
    mock_model_instance.count_tokens.return_value = mock_response

    adapter = GeminiAdapter(api_key="fake-key")
    assert adapter.get_token_count("test text") == 10
    mock_model_instance.count_tokens.assert_called_once_with("test text")
