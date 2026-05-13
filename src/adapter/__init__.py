from src.adapter.base import BaseLLMAdapter
from src.adapter.factory import LLMAdapterFactory
from src.adapter.gemini_adapter import GeminiAdapter
from src.adapter.openai_adapter import OpenAIAdapter

# Register all available adapters with the factory
LLMAdapterFactory.register_adapter("openai", OpenAIAdapter)
LLMAdapterFactory.register_adapter("gemini", GeminiAdapter)

__all__ = [
    "BaseLLMAdapter",
    "LLMAdapterFactory",
    "OpenAIAdapter",
    "GeminiAdapter",
]
