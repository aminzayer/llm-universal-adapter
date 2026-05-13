from .base import BaseLLMAdapter
from .factory import LLMAdapterFactory
from .gemini_adapter import GeminiAdapter
from .openai_adapter import OpenAIAdapter

# Register all available adapters with the factory
LLMAdapterFactory.register_adapter("openai", OpenAIAdapter)
LLMAdapterFactory.register_adapter("gemini", GeminiAdapter)

__all__ = [
    "BaseLLMAdapter",
    "LLMAdapterFactory",
    "OpenAIAdapter",
    "GeminiAdapter",
]
