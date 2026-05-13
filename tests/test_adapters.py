import pytest
from unittest.mock import MagicMock, patch

import openai
from google.genai.errors import APIError  # pyright: ignore[reportMissingImports]

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


# =============================================================================
# OpenAIAdapter Tests
# =============================================================================


def test_openai_adapter_missing_key() -> None:
    """Ensure OpenAIAdapter raises ValueError if API key is completely missing."""
    # Corrected patch path to remove 'src.'
    with patch("config.settings.openai_api_key", None):
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
    """Test token counting logic using tiktoken."""
    mock_encoding = MagicMock()
    mock_encoding.encode.return_value = [1, 2, 3, 4]
    mock_encoding_for_model.return_value = mock_encoding

    adapter = OpenAIAdapter(api_key="fake-key")
    assert adapter.get_token_count("test text") == 4
    mock_encoding.encode.assert_called_once_with("test text")


# =============================================================================
# GeminiAdapter Tests (Updated for google-genai SDK)
# =============================================================================


def test_gemini_adapter_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that initializing the adapter without an API key raises a ValueError.
    """
    # Ensure environment variable is unset for the test context
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with patch("config.settings.gemini_api_key", None):
        with pytest.raises(ValueError, match="Gemini API key is missing"):
            GeminiAdapter(api_key=None)


@patch("adapter.gemini_adapter.genai.Client")
def test_gemini_adapter_generate_response_retry_success(MockClient: MagicMock) -> None:
    """
    Test successful text generation using the new isolated Client architecture
    and verifying tenacity retry logic.
    """
    mock_instance = MockClient.return_value
    mock_response = MagicMock()
    mock_response.text = "Hello Gemini!"

    # Provide an empty dictionary for the required response_json argument to satisfy the constructor
    error = APIError("Quota exceeded", {})
    mock_instance.models.generate_content.side_effect = [error, mock_response]

    adapter = GeminiAdapter(api_key="fake-key")
    response = adapter.generate_response("Hi")

    assert response == "Hello Gemini!"
    assert mock_instance.models.generate_content.call_count == 2

@patch("adapter.gemini_adapter.genai.Client")
def test_gemini_adapter_get_token_count(MockClient: MagicMock) -> None:
    """
    Test token counting functionality using the updated genai SDK.
    """
    # Setup mock for count_tokens
    mock_instance = MockClient.return_value
    mock_response = MagicMock()
    mock_response.total_tokens = 10
    mock_instance.models.count_tokens.return_value = mock_response

    adapter = GeminiAdapter(api_key="fake-key")
    result = adapter.get_token_count("test text")

    assert result == 10
    mock_instance.models.count_tokens.assert_called_once()
