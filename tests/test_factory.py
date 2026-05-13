import pytest
from typing import Any

from src.adapter.base import BaseLLMAdapter
from src.adapter.factory import LLMAdapterFactory


class DummyAdapter(BaseLLMAdapter):
    """A dummy adapter implementation to test the factory independently."""

    def __init__(self, api_key: str = None) -> None:
        self.api_key = api_key

    def generate_response(self, prompt: str, **kwargs: Any) -> str:
            return "dummy response"

    def get_token_count(self, text: str) -> int:
        return len(text.split())

    def generate_with_tools(self, prompt: str) -> str:
        return "dummy tool response"


def test_register_and_create_adapter() -> None:
    """Test that a valid adapter can be registered and instantiated."""
    LLMAdapterFactory.register_adapter("dummy", DummyAdapter)
    adapter = LLMAdapterFactory.create_adapter("dummy", api_key="secret-key")

    assert isinstance(adapter, DummyAdapter)
    assert adapter.api_key == "secret-key"


def test_create_unregistered_adapter() -> None:
    """Test that creating an unregistered adapter raises a ValueError."""
    with pytest.raises(ValueError, match="Provider 'unregistered' is not registered."):
        LLMAdapterFactory.create_adapter("unregistered")
