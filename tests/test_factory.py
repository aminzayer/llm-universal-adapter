import pytest
from typing import Any, AsyncGenerator

# pyrefly: ignore [missing-import]
from src.adapter.base import BaseLLMAdapter
# pyrefly: ignore [missing-import]
from src.adapter.factory import LLMAdapterFactory


class DummyAdapter(BaseLLMAdapter):
    """A dummy adapter implementation to test the factory independently."""

    # pyrefly: ignore [bad-function-definition]
    def __init__(self, api_key: str = None) -> None:
        self.api_key = api_key
        super().__init__()

    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        return "dummy response"

    async def agenerate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        yield "dummy stream"

    async def get_token_count(self, text: str) -> int:
        return len(text.split())

    async def generate_with_tools(self, prompt: str) -> str:
        return "dummy tool response"


def test_register_and_create_adapter() -> None:
    """Test that a valid adapter can be registered and instantiated."""
    LLMAdapterFactory.register_adapter("dummy", DummyAdapter)
    adapter = LLMAdapterFactory.create_adapter("dummy", api_key="secret-key")

    from telemetry.tracer import ObservabilityMiddleware
    assert isinstance(adapter, ObservabilityMiddleware)
    assert isinstance(adapter.adapter, DummyAdapter)
    assert adapter.adapter.api_key == "secret-key"


def test_create_unregistered_adapter() -> None:
    """Test that creating an unregistered adapter raises a ValueError."""
    with pytest.raises(ValueError, match="Provider 'unregistered' is not registered."):
        LLMAdapterFactory.create_adapter("unregistered")
