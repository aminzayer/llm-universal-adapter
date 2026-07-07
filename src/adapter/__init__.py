from .base import BaseLLMAdapter
from .factory import LLMAdapterFactory
from .gemini_adapter import GeminiAdapter
from .local_adapter import LocalModelAdapter
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter

# Register all available adapters with the factory
LLMAdapterFactory.register_adapter("openai", OpenAIAdapter)
LLMAdapterFactory.register_adapter("gemini", GeminiAdapter)
LLMAdapterFactory.register_adapter("local", LocalModelAdapter)
LLMAdapterFactory.register_adapter("anthropic", AnthropicAdapter)

__all__ = [
    "BaseLLMAdapter",
    "LLMAdapterFactory",
    "OpenAIAdapter",
    "GeminiAdapter",
    "LocalModelAdapter",
    "AnthropicAdapter",
]
