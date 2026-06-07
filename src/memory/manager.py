import logging
from typing import Dict, List

from adapter.base import BaseLLMAdapter

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages multi-turn conversation states for Large Language Models.
    Implements a sliding window algorithm to ensure the conversation history 
    stays within a specified token limit while strictly preserving the system prompt.
    """

    def __init__(
        self,
        adapter: BaseLLMAdapter,
        max_context_tokens: int = 4096,
        threshold: float = 0.8,
    ) -> None:
        """
        Initializes the ConversationManager.

        Args:
            adapter (BaseLLMAdapter): The LLM adapter used to accurately calculate token counts.
            max_context_tokens (int): The absolute maximum context window size for the model.
            threshold (float): The percentage (0.0 to 1.0) of the max_context_tokens at which 
                               truncation begins.
        """
        self.adapter = adapter
        self.max_context_tokens = max_context_tokens
        self.threshold = threshold
        self.messages: List[Dict[str, str]] = []

    async def add_message(self, role: str, content: str) -> None:
        """
        Appends a new message to the conversation and enforces the sliding window limit.
        """
        self.messages.append({"role": role, "content": content})
        await self._apply_sliding_window()

    async def get_messages(self) -> List[Dict[str, str]]:
        """
        Returns the active, truncated list of conversation messages.
        """
        return self.messages

    def clear(self) -> None:
        """
        Clears the conversation history entirely.
        """
        self.messages = []

    async def _apply_sliding_window(self) -> None:
        """
        Dynamically truncates the oldest non-system messages if the total token 
        count exceeds the configurable threshold.
        """
        limit = int(self.max_context_tokens * self.threshold)

        while len(self.messages) > 1:
            total_tokens = await self._calculate_total_tokens()
            if total_tokens <= limit:
                break

            # Ensure the system prompt at the beginning of the conversation is strictly preserved
            removal_index = 1 if self.messages[0]["role"] == "system" else 0

            if removal_index < len(self.messages):
                removed_msg = self.messages.pop(removal_index)
                logger.debug(f"Truncated message from '{removed_msg['role']}' to stay within token limits.")
            else:
                break

    async def _calculate_total_tokens(self) -> int:
        """Calculates the aggregate token count of the current conversation."""
        total = 0
        for msg in self.messages:
            total += await self.adapter.get_token_count(msg["content"])
        return total
