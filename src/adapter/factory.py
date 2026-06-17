from typing import Type, Dict, Any
from .base import BaseLLMAdapter


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
    def create_adapter(
        cls,
        provider_name: str,
        *,
        enable_guardrail: bool = False,
        block_on_pii: bool = False,
        **kwargs: Any,
    ) -> BaseLLMAdapter:
        """
        Creates and returns an instance of the requested LLM adapter.
        Raises ValueError if the provider is not registered.

        The returned object is wrapped in :class:`ObservabilityMiddleware` for
        structured logging. When ``enable_guardrail=True`` an additional
        :class:`InputGuardrailMiddleware` layer is added on the outside so the
        prompt is screened *before* telemetry or the cache see it.

        Args:
            provider_name: Registered provider identifier (e.g. ``"openai"``).
            enable_guardrail: Opt-in switch to wrap the adapter with the
                input guardrail middleware. Defaults to ``False`` so existing
                call sites are unaffected.
            block_on_pii: If ``True``, prompts that contain masked PII are
                refused via :class:`SecurityViolationError` instead of being
                silently sanitized. Only meaningful when ``enable_guardrail``
                is also set.
            **kwargs: Forwarded to the provider adapter's constructor.
        """
        adapter_class = cls._adapters.get(provider_name.lower())
        if not adapter_class:
            raise ValueError(f"Provider '{provider_name}' is not registered.")

        adapter_instance = adapter_class(**kwargs)

        from telemetry.tracer import ObservabilityMiddleware
        wrapped: BaseLLMAdapter = ObservabilityMiddleware(
            adapter=adapter_instance, provider=provider_name
        )

        if enable_guardrail:
            from security.guardrail import InputGuardrailMiddleware
            wrapped = InputGuardrailMiddleware(
                wrapped, block_on_pii=block_on_pii
            )

        return wrapped
