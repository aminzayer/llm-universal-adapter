from typing import Type, Dict, Any
from src.adapter.base import BaseLLMAdapter


class LLMAdapterFactory:
    """
    Factory class to instantiate the appropriate LLM adapter
    based on provider name.
    """
    _adapters: Dict[str, Type[BaseLLMAdapter]] = {}

    @classmethod
    def register_adapter(cls, provider_name: str, adapter_class: Type[BaseLLMAdapter]) -> None:
        """
        Registers a new LLM provider adapter.
        """
        cls._adapters[provider_name.lower()] = adapter_class

    @classmethod
    def create_adapter(cls, provider_name: str, **kwargs: Any) -> BaseLLMAdapter:
        """
        Creates and returns an instance of the requested LLM adapter.
        Raises ValueError if the provider is not registered.
        """
        adapter_class = cls._adapters.get(provider_name.lower())
        if not adapter_class:
            raise ValueError(f"Provider '{provider_name}' is not registered.")
        return adapter_class(**kwargs)
