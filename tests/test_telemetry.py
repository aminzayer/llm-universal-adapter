import json
from typing import Any, AsyncGenerator
from unittest.mock import patch

import pytest

from adapter.base import BaseLLMAdapter
from telemetry.tracer import ObservabilityMiddleware


class DummyInnerAdapter(BaseLLMAdapter):
    """A dummy adapter to simulate LLM responses for the telemetry tests."""

    def __init__(self) -> None:
        super().__init__()
        self.model = "dummy-model"

    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        if prompt == "error":
            raise ValueError("Simulated LLM Error")
        return f"Response to {prompt}"

    async def agenerate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        yield "Stream "
        yield "Response"

    async def get_token_count(self, text: str) -> int:
        return len(text.split())

    async def generate_with_tools(self, prompt: str) -> str:
        return "Tool Response"


@pytest.fixture
def mock_logger() -> Any:
    with patch("telemetry.tracer.logger") as mock:
        yield mock


@pytest.mark.asyncio
async def test_observability_generate_response(mock_logger: Any) -> None:
    """Tests if generate_response correctly emits structured JSON telemetry."""
    adapter = DummyInnerAdapter()
    middleware = ObservabilityMiddleware(adapter, "dummy_provider")

    res = await middleware.generate_response("Hello world")
    assert res == "Response to Hello world"

    mock_logger.info.assert_called_once()
    log_json = mock_logger.info.call_args[0][0]
    log_data = json.loads(log_json)

    assert log_data["event"] == "llm_invocation"
    assert log_data["provider"] == "dummy_provider"
    assert log_data["model"] == "dummy-model"
    assert log_data["operation"] == "generate_response"
    assert log_data["prompt_tokens"] == 2
    assert log_data["completion_tokens"] == 4
    assert log_data["total_tokens"] == 6
    assert "latency_sec" in log_data
    assert log_data["error"] is None


@pytest.mark.asyncio
async def test_observability_agenerate_stream(mock_logger: Any) -> None:
    """Tests if agenerate_stream accurately captures streaming metadata."""
    adapter = DummyInnerAdapter()
    middleware = ObservabilityMiddleware(adapter, "dummy_provider")

    chunks = []
    async for chunk in middleware.agenerate_stream("Hello"):
        chunks.append(chunk)

    assert "".join(chunks) == "Stream Response"

    mock_logger.info.assert_called_once()
    log_data = json.loads(mock_logger.info.call_args[0][0])

    assert log_data["operation"] == "agenerate_stream"
    assert log_data["prompt_tokens"] == 1
    assert log_data["completion_tokens"] == 2


@pytest.mark.asyncio
async def test_observability_generate_response_error(mock_logger: Any) -> None:
    """Tests if errors are correctly caught and logged."""
    adapter = DummyInnerAdapter()
    middleware = ObservabilityMiddleware(adapter, "dummy_provider")

    with pytest.raises(ValueError, match="Simulated LLM Error"):
        await middleware.generate_response("error")

    mock_logger.info.assert_called_once()
    log_data = json.loads(mock_logger.info.call_args[0][0])
    assert log_data["error"] == "Simulated LLM Error"
