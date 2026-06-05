import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.adapter.gemini_adapter import GeminiAdapter


@pytest.fixture
def mock_genai_client():

    with patch('src.adapter.gemini_adapter.genai.Client') as MockClient:
        mock_client_instance = MagicMock()
        mock_aio = MagicMock()
        mock_client_instance.aio = mock_aio
        MockClient.return_value = mock_client_instance
        yield mock_aio


@pytest.fixture
def adapter(mock_genai_client):
    with patch('src.adapter.gemini_adapter.settings') as mock_settings:
        mock_settings.gemini_api_key = "test_key"
        mock_settings.default_temperature = 0.7
        return GeminiAdapter(model="gemini-2.5-flash")


@pytest.mark.asyncio
async def test_generate_response_success(adapter, mock_genai_client):
    mock_response = MagicMock()
    mock_response.text = "Hello Async World"
    mock_genai_client.models.generate_content = AsyncMock(return_value=mock_response)

    response = await adapter.generate_response("Hello")
    assert response == "Hello Async World"
    mock_genai_client.models.generate_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_agenerate_stream_success(adapter, mock_genai_client):

    async def mock_stream():
        for word in ["Hello", " Async", " World"]:
            chunk = MagicMock()
            chunk.text = word
            yield chunk

    mock_genai_client.models.generate_content_stream = AsyncMock(return_value=mock_stream())

    chunks = []
    async for chunk in adapter.agenerate_stream("Stream this"):
        chunks.append(chunk)

    assert "".join(chunks) == "Hello Async World"
    mock_genai_client.models.generate_content_stream.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_token_count(adapter, mock_genai_client):
    mock_response = MagicMock()
    mock_response.total_tokens = 42
    mock_genai_client.models.count_tokens = AsyncMock(return_value=mock_response)

    count = await adapter.get_token_count("Tokenize me")
    assert count == 42
    mock_genai_client.models.count_tokens.assert_awaited_once()
