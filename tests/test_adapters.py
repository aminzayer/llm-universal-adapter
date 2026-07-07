import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import anthropic
from google.genai.errors import APIError  # pyright: ignore[reportMissingImports]

from adapter.openai_adapter import OpenAIAdapter
from adapter.gemini_adapter import GeminiAdapter
from adapter.anthropic_adapter import AnthropicAdapter


@pytest.fixture(autouse=True)
def fast_retries():
    """
    Fixture to mock `time.sleep` used by `tenacity` during backoff.
    This ensures tests involving retries run instantaneously.
    """
    with patch("tenacity.nap.time.sleep"), patch("asyncio.sleep"):
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


@pytest.mark.asyncio
@patch("src.adapter.openai_adapter.openai.AsyncClient")
async def test_openai_adapter_generate_response_success(mock_client: MagicMock) -> None:
    """Test successful response generation without retries."""
    mock_instance = mock_client.return_value
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello OpenAI!"))]
    mock_instance.chat.completions.create = AsyncMock(return_value=mock_response)

    adapter = OpenAIAdapter(api_key="fake-key")
    response = await adapter.generate_response("Hello!")

    assert response == "Hello OpenAI!"
    mock_instance.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
@patch("src.adapter.openai_adapter.openai.AsyncClient")
async def test_openai_adapter_generate_response_retry_success(mock_client: MagicMock) -> None:
    """Test that a RateLimitError triggers a retry and eventually succeeds."""
    mock_instance = mock_client.return_value
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Hello again!"))]

    # Fail twice with RateLimitError, then succeed
    mock_error = openai.RateLimitError(message="Rate limited", response=MagicMock(), body=None)
    mock_instance.chat.completions.create = AsyncMock(side_effect=[
        mock_error,
        mock_error,
        mock_response,
    ])

    adapter = OpenAIAdapter(api_key="fake-key")
    response = await adapter.generate_response("Hi")

    assert response == "Hello again!"
    assert mock_instance.chat.completions.create.call_count == 3


@pytest.mark.asyncio
@patch("src.adapter.openai_adapter.openai.AsyncClient")
async def test_openai_adapter_generate_response_retry_failure(mock_client: MagicMock) -> None:
    """Test that repeated failures eventually raise the underlying exception."""
    mock_instance = mock_client.return_value
    mock_error = openai.InternalServerError(message="Server Error", response=MagicMock(), body=None)
    mock_instance.chat.completions.create = AsyncMock(side_effect=mock_error)

    adapter = OpenAIAdapter(api_key="fake-key")

    with pytest.raises(openai.InternalServerError):
        await adapter.generate_response("Hi")

    # Tenacity stops after 5 attempts
    assert mock_instance.chat.completions.create.call_count == 5


@pytest.mark.asyncio
@patch("src.adapter.openai_adapter.tiktoken.encoding_for_model")
async def test_openai_adapter_get_token_count(mock_encoding_for_model: MagicMock) -> None:
    """Test token counting logic using tiktoken."""
    mock_encoding = MagicMock()
    mock_encoding.encode.return_value = [1, 2, 3, 4]
    mock_encoding_for_model.return_value = mock_encoding

    adapter = OpenAIAdapter(api_key="fake-key")
    assert await adapter.get_token_count("test text") == 4
    mock_encoding.encode.assert_called_once_with("test text")


# =============================================================================
# GeminiAdapter Tests (Updated for google-genai SDK)
# =============================================================================


@pytest.mark.asyncio
@patch("src.adapter.openai_adapter.openai.AsyncClient")
async def test_openai_adapter_generate_with_tools(mock_client: MagicMock) -> None:
    """Test tool execution pipeline for OpenAI."""
    import json
    mock_instance = mock_client.return_value

    # First response: returns a tool call
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.function.name = "get_weather"
    mock_tool_call.function.arguments = json.dumps({"location": "London"})

    first_message = MagicMock()
    first_message.content = None
    first_message.tool_calls = [mock_tool_call]

    first_response = MagicMock()
    first_response.choices = [MagicMock(message=first_message)]

    # Second response: returns the final string
    second_message = MagicMock()
    second_message.content = "The weather in London is sunny."
    second_message.tool_calls = None

    second_response = MagicMock()
    second_response.choices = [MagicMock(message=second_message)]

    mock_instance.chat.completions.create = AsyncMock(side_effect=[first_response, second_response])

    adapter = OpenAIAdapter(api_key="fake-key")

    # Register a test tool
    def get_weather(location: str) -> str:
        return f"Weather in {location} is sunny"

    adapter.register_tool("get_weather", get_weather, "Get the weather for a location")

    response = await adapter.generate_with_tools("What is the weather in London?")

    assert response == "The weather in London is sunny."
    assert mock_instance.chat.completions.create.call_count == 2


def test_gemini_adapter_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that initializing the adapter without an API key raises a ValueError.
    """
    # Ensure environment variable is unset for the test context
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with patch("config.settings.gemini_api_key", None):
        with pytest.raises(ValueError, match="Gemini API key is missing"):
            GeminiAdapter(api_key=None)


@pytest.mark.asyncio
@patch("src.adapter.gemini_adapter.genai.Client")
async def test_gemini_adapter_generate_response_retry_success(MockClient: MagicMock) -> None:
    """
    Test successful text generation using the new isolated Client architecture
    and verifying tenacity retry logic.
    """
    mock_instance = MockClient.return_value
    mock_response = MagicMock()
    mock_response.text = "Hello Gemini!"

    # Provide an empty dictionary for the required response_json argument to satisfy the constructor
    error = APIError(429, "Quota exceeded")
    mock_instance.aio.models.generate_content = AsyncMock(side_effect=[error, mock_response])

    adapter = GeminiAdapter(api_key="fake-key")
    response = await adapter.generate_response("Hi")

    assert response == "Hello Gemini!"
    assert mock_instance.aio.models.generate_content.call_count == 2


@pytest.mark.asyncio
@patch("src.adapter.gemini_adapter.genai.Client")
async def test_gemini_adapter_get_token_count(MockClient: MagicMock) -> None:
    """
    Test token counting functionality using the updated genai SDK.
    """
    # Setup mock for count_tokens
    mock_instance = MockClient.return_value
    mock_response = MagicMock()
    mock_response.total_tokens = 10
    mock_instance.aio.models.count_tokens = AsyncMock(return_value=mock_response)

    adapter = GeminiAdapter(api_key="fake-key")
    result = await adapter.get_token_count("test text")

    assert result == 10
    mock_instance.aio.models.count_tokens.assert_awaited_once()


@pytest.mark.asyncio
@patch("src.adapter.gemini_adapter.genai.Client")
async def test_gemini_adapter_generate_with_tools(MockClient: MagicMock) -> None:
    """Test tool execution pipeline for Gemini."""
    from google.genai import types
    mock_instance = MockClient.return_value

    # First response: returns a tool call
    mock_tool_call = MagicMock()
    mock_tool_call.name = "get_weather"
    mock_tool_call.args = {"location": "London"}

    first_response = MagicMock()
    first_response.text = None
    first_response.function_calls = [mock_tool_call]
    first_response.candidates = [MagicMock(content=types.Content(role="model", parts=[]))]

    # Second response: returns the final string
    second_response = MagicMock()
    second_response.text = "The weather in London is sunny."
    second_response.function_calls = None

    mock_instance.aio.models.generate_content = AsyncMock(side_effect=[first_response, second_response])

    adapter = GeminiAdapter(api_key="fake-key")

    # Register a test tool
    def get_weather(location: str) -> str:
        return f"Weather in {location} is sunny"

    adapter.register_tool("get_weather", get_weather, "Get the weather for a location")

    response = await adapter.generate_with_tools("What is the weather in London?")

    assert response == "The weather in London is sunny."
    assert mock_instance.aio.models.generate_content.call_count == 2


# =============================================================================
# AnthropicAdapter Tests
# =============================================================================


def test_anthropic_adapter_missing_key() -> None:
    """Ensure AnthropicAdapter raises ValueError if API key is completely missing."""
    with patch("config.settings.anthropic_api_key", None):
        with pytest.raises(ValueError, match="Anthropic API key is missing"):
            AnthropicAdapter(api_key=None)


@pytest.mark.asyncio
@patch("src.adapter.anthropic_adapter.anthropic.AsyncAnthropic")
async def test_anthropic_adapter_generate_response_success(mock_client: MagicMock) -> None:
    """Test successful response generation without retries."""
    mock_instance = mock_client.return_value
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="Hello Anthropic!")]
    mock_instance.messages.create = AsyncMock(return_value=mock_response)

    adapter = AnthropicAdapter(api_key="fake-key")
    response = await adapter.generate_response("Hello!")

    assert response == "Hello Anthropic!"
    mock_instance.messages.create.assert_awaited_once()


@pytest.mark.asyncio
@patch("src.adapter.anthropic_adapter.anthropic.AsyncAnthropic")
async def test_anthropic_adapter_generate_response_retry_success(mock_client: MagicMock) -> None:
    """Test that a RateLimitError triggers a retry and eventually succeeds."""
    mock_instance = mock_client.return_value
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="Hello again!")]

    # Fail twice with RateLimitError, then succeed
    mock_error = anthropic.RateLimitError(message="Rate limited", response=MagicMock(), body=None)
    mock_instance.messages.create = AsyncMock(side_effect=[
        mock_error,
        mock_error,
        mock_response,
    ])

    adapter = AnthropicAdapter(api_key="fake-key")
    response = await adapter.generate_response("Hi")

    assert response == "Hello again!"
    assert mock_instance.messages.create.call_count == 3


@pytest.mark.asyncio
@patch("src.adapter.anthropic_adapter.anthropic.AsyncAnthropic")
async def test_anthropic_adapter_generate_response_retry_failure(mock_client: MagicMock) -> None:
    """Test that repeated failures eventually raise the underlying exception."""
    mock_instance = mock_client.return_value
    mock_error = anthropic.InternalServerError(message="Server Error", response=MagicMock(), body=None)
    mock_instance.messages.create = AsyncMock(side_effect=mock_error)

    adapter = AnthropicAdapter(api_key="fake-key")

    with pytest.raises(anthropic.InternalServerError):
        await adapter.generate_response("Hi")

    # Tenacity stops after 5 attempts
    assert mock_instance.messages.create.call_count == 5


@pytest.mark.asyncio
@patch("src.adapter.anthropic_adapter.anthropic.AsyncAnthropic")
async def test_anthropic_adapter_get_token_count(mock_client: MagicMock) -> None:
    """Test token counting logic."""
    mock_instance = mock_client.return_value
    mock_response = MagicMock()
    mock_response.input_tokens = 12
    mock_instance.messages.count_tokens = AsyncMock(return_value=mock_response)

    adapter = AnthropicAdapter(api_key="fake-key")
    assert await adapter.get_token_count("test text") == 12
    mock_instance.messages.count_tokens.assert_awaited_once()


@pytest.mark.asyncio
@patch("src.adapter.anthropic_adapter.anthropic.AsyncAnthropic")
async def test_anthropic_adapter_agenerate_stream(mock_client: MagicMock) -> None:
    """Test streaming text generation."""
    mock_instance = mock_client.return_value
    
    # Mock the context manager
    mock_stream_ctx = MagicMock()
    mock_stream = AsyncMock()
    
    # Simulating stream.text_stream as an async iterator yielding chunks
    async def async_iter():
        yield "chunk1"
        yield "chunk2"
        
    mock_stream.text_stream = async_iter()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)
    
    mock_instance.messages.stream = MagicMock(return_value=mock_stream_ctx)

    adapter = AnthropicAdapter(api_key="fake-key")
    chunks = []
    async for chunk in adapter.agenerate_stream("hello"):
        chunks.append(chunk)

    assert chunks == ["chunk1", "chunk2"]
    mock_instance.messages.stream.assert_called_once()


@pytest.mark.asyncio
@patch("src.adapter.anthropic_adapter.anthropic.AsyncAnthropic")
async def test_anthropic_adapter_generate_with_tools(mock_client: MagicMock) -> None:
    """Test tool execution pipeline for Anthropic."""
    mock_instance = mock_client.return_value

    # First response: returns a tool call block
    mock_tool_block = MagicMock(type="tool_use")
    mock_tool_block.id = "toolu_123"
    mock_tool_block.name = "get_weather"
    mock_tool_block.input = {"location": "London"}

    first_response = MagicMock()
    first_response.content = [mock_tool_block]

    # Second response: returns the final text block
    mock_text_block = MagicMock(type="text", text="The weather in London is sunny.")
    second_response = MagicMock()
    second_response.content = [mock_text_block]

    mock_instance.messages.create = AsyncMock(side_effect=[first_response, second_response])

    adapter = AnthropicAdapter(api_key="fake-key")

    # Register a test tool
    def get_weather(location: str) -> str:
        return f"Weather in {location} is sunny"

    adapter.register_tool("get_weather", get_weather, "Get the weather for a location")

    response = await adapter.generate_with_tools("What is the weather in London?")

    assert response == "The weather in London is sunny."
    assert mock_instance.messages.create.call_count == 2


@pytest.mark.asyncio
@patch("src.adapter.anthropic_adapter.anthropic.AsyncAnthropic")
async def test_anthropic_adapter_generate_with_tools_hitl(mock_client: MagicMock) -> None:
    """Test HITL suspension flow for Anthropic when approval is required."""
    from orchestration.hitl import ApprovalRequiredError
    mock_instance = mock_client.return_value

    mock_tool_block = MagicMock(type="tool_use")
    mock_tool_block.id = "toolu_hitl"
    mock_tool_block.name = "secure_tool"
    mock_tool_block.input = {"data": "secret"}

    first_response = MagicMock()
    first_response.content = [mock_tool_block]

    mock_instance.messages.create = AsyncMock(return_value=first_response)

    adapter = AnthropicAdapter(api_key="fake-key")
    
    # Mock Redis client
    class MockRedis:
        def __init__(self):
            self.store = {}
        async def set(self, key, value, ex=None):
            self.store[key] = value

    mock_redis = MockRedis()
    adapter.set_redis_client(mock_redis)

    def secure_tool(data: str) -> str:
        return f"Processed {data}"

    adapter.register_tool("secure_tool", secure_tool, "A secure tool", requires_approval=True)

    with pytest.raises(ApprovalRequiredError) as exc_info:
        await adapter.generate_with_tools("Run the secure tool")

    assert exc_info.value.tool_name == "secure_tool"
    assert exc_info.value.tool_args == {"data": "secret"}
    assert exc_info.value.tool_call_id == "toolu_hitl"
    assert len(mock_redis.store) == 1


@pytest.mark.asyncio
@patch("src.adapter.anthropic_adapter.anthropic.AsyncAnthropic")
async def test_anthropic_adapter_resume_with_tools(mock_client: MagicMock) -> None:
    """Test resuming a suspended tool execution for Anthropic."""
    mock_instance = mock_client.return_value

    mock_text_block = MagicMock(type="text", text="Tool execution approved. Result: Success.")
    second_response = MagicMock()
    second_response.content = [mock_text_block]

    mock_instance.messages.create = AsyncMock(return_value=second_response)

    adapter = AnthropicAdapter(api_key="fake-key")

    def secure_tool(data: str) -> str:
        return f"Processed {data}"

    adapter.register_tool("secure_tool", secure_tool, "A secure tool", requires_approval=True)

    # Recreate the suspended state object
    from orchestration.hitl import HITLState
    state = HITLState(
        state_id="hitl_123",
        request_id="req_123",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        tool_name="secure_tool",
        tool_args={"data": "secret"},
        tool_call_id="toolu_hitl",
        messages=[{"role": "user", "content": "Run secure tool"}],
        pending_tool_calls=[{"id": "toolu_hitl", "type": "tool_use", "name": "secure_tool", "input": {"data": "secret"}}],
        status="pending"
    )

    response = await adapter.resume_with_tools(state)

    assert response == "Tool execution approved. Result: Success."
    mock_instance.messages.create.assert_awaited_once()
