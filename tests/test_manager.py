from typing import Any, AsyncGenerator

import pytest

# pyrefly: ignore [missing-import]
from src.adapter.base import BaseLLMAdapter
# pyrefly: ignore [missing-import]
from src.memory.manager import ConversationManager


class MockTokenAdapter(BaseLLMAdapter):
    """A mock adapter that simply equates 1 word to 1 token for testing purposes."""

    def __init__(self) -> None:
        super().__init__()

    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        return "mock response"

    async def agenerate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        yield "mock stream"

    async def generate_with_tools(self, prompt: str) -> str:
        return "mock tools"

    async def get_token_count(self, text: str) -> int:
        return len(text.split())


@pytest.mark.asyncio
async def test_manager_no_truncation() -> None:
    """Test that messages are preserved when strictly under the token threshold."""
    adapter = MockTokenAdapter()
    manager = ConversationManager(adapter, max_context_tokens=10, threshold=1.0)

    await manager.add_message("user", "Hello world")
    messages = await manager.get_messages()

    assert len(messages) == 1
    assert messages[0]["content"] == "Hello world"


@pytest.mark.asyncio
async def test_manager_sliding_window_truncation() -> None:
    """Test that the oldest message is removed when exceeding the token threshold."""
    adapter = MockTokenAdapter()
    manager = ConversationManager(adapter, max_context_tokens=10, threshold=1.0)

    await manager.add_message("user", "One Two Three")  # 3 tokens
    await manager.add_message("assistant", "Four Five Six")  # 3 tokens
    await manager.add_message("user", "Seven Eight Nine Ten Eleven")  # 5 tokens

    # Total before truncation = 11 > 10. Should discard the first user message.
    messages = await manager.get_messages()

    assert len(messages) == 2
    assert messages[0]["content"] == "Four Five Six"
    assert messages[1]["content"] == "Seven Eight Nine Ten Eleven"


@pytest.mark.asyncio
async def test_manager_preserves_system_prompt() -> None:
    """Test that the system prompt is strictly preserved during truncation."""
    adapter = MockTokenAdapter()
    manager = ConversationManager(adapter, max_context_tokens=10, threshold=1.0)

    await manager.add_message("system", "System Config")  # 2 tokens
    await manager.add_message("user", "One Two Three")  # 3 tokens
    await manager.add_message("assistant", "Four Five Six")  # 3 tokens
    await manager.add_message("user", "Seven Eight Nine")  # 3 tokens

    # Total = 11 > 10. Must bypass the system prompt and discard "One Two Three"
    messages = await manager.get_messages()

    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "System Config"
    assert messages[1]["content"] == "Four Five Six"
