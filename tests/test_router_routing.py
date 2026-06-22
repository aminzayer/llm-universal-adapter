from typing import Any, AsyncGenerator

import pytest

from adapter.base import BaseLLMAdapter
from orchestration.router import RouterManager


class EchoAdapter(BaseLLMAdapter):
    """
    Stub adapter that records its constructor args and returns a deterministic
    string from each method. Used as the local / primary / fallback adapter in
    router routing tests so we don't stand up a real HTTP transport.
    """

    def __init__(self, tag: str, **kwargs: Any) -> None:
        super().__init__()
        self.tag = tag
        self.calls: list[str] = []

    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append(prompt)
        return f"{self.tag}:{prompt}"

    async def agenerate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        self.calls.append(prompt)
        for chunk in ("a", "b", "c"):
            yield chunk

    async def get_token_count(self, text: str) -> int:
        return len(text.split())

    async def generate_with_tools(self, prompt: str) -> str:
        self.calls.append(prompt)
        return f"{self.tag}:tools:{prompt}"


class FailingAdapter(BaseLLMAdapter):
    """Stub adapter that always raises — used to exercise failover paths."""

    def __init__(self, tag: str, error: Exception | None = None, **kwargs: Any) -> None:
        super().__init__()
        self.tag = tag
        self.error = error or RuntimeError(f"{tag} down")

    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        raise self.error

    async def agenerate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        raise self.error
        yield ""  # pragma: no cover — keeps this an async generator

    async def get_token_count(self, text: str) -> int:
        return 0

    async def generate_with_tools(self, prompt: str) -> str:
        raise self.error


@pytest.fixture(autouse=True)
def register_echo_and_failing() -> Any:
    """
    Registers stub adapters under names the router uses in these tests.
    Cleans up the registry after the test to avoid polluting other suites.
    """
    from adapter.factory import LLMAdapterFactory

    LLMAdapterFactory.register_adapter("echo_primary", EchoAdapter)
    LLMAdapterFactory.register_adapter("echo_fallback", EchoAdapter)
    LLMAdapterFactory.register_adapter("echo_local", EchoAdapter)
    LLMAdapterFactory.register_adapter("failing_primary", FailingAdapter)
    LLMAdapterFactory.register_adapter("failing_fallback", FailingAdapter)
    LLMAdapterFactory.register_adapter("failing_local", FailingAdapter)
    yield
    # Unregister so test ordering cannot leak state between files.
    for name in (
        "echo_primary",
        "echo_fallback",
        "echo_local",
        "failing_primary",
        "failing_fallback",
        "failing_local",
    ):
        LLMAdapterFactory._adapters.pop(name, None)


# =============================================================================
# Trivial vs complex routing
# =============================================================================


@pytest.mark.asyncio
async def test_router_routes_trivial_to_local() -> None:
    """Short classification-style prompts must hit the local adapter."""
    router = RouterManager(
        primary_provider="echo_primary",
        fallback_provider="echo_fallback",
        primary_kwargs={"tag": "primary"},
        fallback_kwargs={"tag": "fallback"},
        local_provider="echo_local",
        local_kwargs={"tag": "local"},
    )

    response = await router.generate_response("classify this as positive or negative: 'great job'")

    # The local adapter is wrapped in ObservabilityMiddleware by the factory,
    # so its `.adapter` attribute is the underlying EchoAdapter.
    local_inner = router.local_adapter.adapter  # type: ignore[attr-defined]
    primary_inner = router.primary_adapter.adapter  # type: ignore[attr-defined]
    fallback_inner = router.fallback_adapter.adapter  # type: ignore[attr-defined]

    assert response == "local:classify this as positive or negative: 'great job'"
    assert local_inner.calls == ["classify this as positive or negative: 'great job'"]
    assert primary_inner.calls == []
    assert fallback_inner.calls == []


@pytest.mark.asyncio
async def test_router_routes_complex_to_primary() -> None:
    """Reasoning / code / long prompts must skip the local adapter."""
    router = RouterManager(
        primary_provider="echo_primary",
        fallback_provider="echo_fallback",
        primary_kwargs={"tag": "primary"},
        fallback_kwargs={"tag": "fallback"},
        local_provider="echo_local",
        local_kwargs={"tag": "local"},
    )

    complex_prompt = "Explain the difference between async and concurrent programming, then implement a producer/consumer queue."
    response = await router.generate_response(complex_prompt)

    local_inner = router.local_adapter.adapter  # type: ignore[attr-defined]
    primary_inner = router.primary_adapter.adapter  # type: ignore[attr-defined]

    assert response.startswith("primary:")
    assert local_inner.calls == []
    assert primary_inner.calls == [complex_prompt]


@pytest.mark.asyncio
async def test_router_treats_long_prompts_as_complex() -> None:
    """A prompt over the trivial length cap must skip the local adapter."""
    router = RouterManager(
        primary_provider="echo_primary",
        fallback_provider="echo_fallback",
        primary_kwargs={"tag": "primary"},
        fallback_kwargs={"tag": "fallback"},
        local_provider="echo_local",
        local_kwargs={"tag": "local"},
    )

    long_prompt = "label this: " + ("word " * 200)
    response = await router.generate_response(long_prompt)

    local_inner = router.local_adapter.adapter  # type: ignore[attr-defined]
    primary_inner = router.primary_adapter.adapter  # type: ignore[attr-defined]

    assert response.startswith("primary:")
    assert local_inner.calls == []
    assert primary_inner.calls == [long_prompt]


@pytest.mark.asyncio
async def test_router_handles_json_wrapped_messages_prompt() -> None:
    """A prompt that starts with '[' (serialised messages) must still be classified."""
    router = RouterManager(
        primary_provider="echo_primary",
        fallback_provider="echo_fallback",
        primary_kwargs={"tag": "primary"},
        fallback_kwargs={"tag": "fallback"},
        local_provider="echo_local",
        local_kwargs={"tag": "local"},
    )

    import json
    messages_prompt = json.dumps([
        {"role": "user", "content": "categorize this product as electronics or clothing"},
    ])

    response = await router.generate_response(messages_prompt)

    local_inner = router.local_adapter.adapter  # type: ignore[attr-defined]
    primary_inner = router.primary_adapter.adapter  # type: ignore[attr-defined]

    assert response.startswith("local:")
    assert local_inner.calls == [messages_prompt]
    assert primary_inner.calls == []


# =============================================================================
# Local failure fallthrough
# =============================================================================


@pytest.mark.asyncio
async def test_router_falls_back_to_cloud_when_local_fails() -> None:
    """If the local adapter raises, the router must transparently fall through to primary."""
    router = RouterManager(
        primary_provider="echo_primary",
        fallback_provider="echo_fallback",
        primary_kwargs={"tag": "primary"},
        fallback_kwargs={"tag": "fallback"},
        local_provider="failing_local",
        local_kwargs={"tag": "local"},
    )

    response = await router.generate_response("classify: spam or not")

    primary_inner = router.primary_adapter.adapter  # type: ignore[attr-defined]
    fallback_inner = router.fallback_adapter.adapter  # type: ignore[attr-defined]

    assert response.startswith("primary:")
    assert primary_inner.calls == ["classify: spam or not"]
    assert fallback_inner.calls == []


@pytest.mark.asyncio
async def test_router_full_cloud_failover_when_local_and_primary_fail() -> None:
    """Local failure + primary failure must still reach the fallback adapter."""
    router = RouterManager(
        primary_provider="failing_primary",
        fallback_provider="echo_fallback",
        primary_kwargs={"tag": "primary"},
        fallback_kwargs={"tag": "fallback"},
        local_provider="failing_local",
        local_kwargs={"tag": "local"},
    )

    response = await router.generate_response("classify: yes/no")

    fallback_inner = router.fallback_adapter.adapter  # type: ignore[attr-defined]

    assert response.startswith("fallback:")
    assert fallback_inner.calls == ["classify: yes/no"]


# =============================================================================
# No local provider configured
# =============================================================================


@pytest.mark.asyncio
async def test_router_skips_local_when_not_configured() -> None:
    """Without a local provider, even trivial prompts go to primary."""
    router = RouterManager(
        primary_provider="echo_primary",
        fallback_provider="echo_fallback",
        primary_kwargs={"tag": "primary"},
        fallback_kwargs={"tag": "fallback"},
    )

    response = await router.generate_response("classify this")

    assert router.local_adapter is None
    assert response.startswith("primary:")


# =============================================================================
# generate_with_tools ignores local
# =============================================================================


@pytest.mark.asyncio
async def test_router_generate_with_tools_ignores_local() -> None:
    """Tool-calling requests must always use the cloud chain, never local."""
    router = RouterManager(
        primary_provider="echo_primary",
        fallback_provider="echo_fallback",
        primary_kwargs={"tag": "primary"},
        fallback_kwargs={"tag": "fallback"},
        local_provider="echo_local",
        local_kwargs={"tag": "local"},
    )

    def my_tool(x: int) -> str:
        return str(x)

    router.register_tool("my_tool", my_tool, "echo an int")

    response = await router.generate_with_tools("classify something trivial")

    local_inner = router.local_adapter.adapter  # type: ignore[attr-defined]
    primary_inner = router.primary_adapter.adapter  # type: ignore[attr-defined]

    assert response.startswith("primary:tools:")
    assert local_inner.calls == []
    assert primary_inner.calls == ["classify something trivial"]


@pytest.mark.asyncio
async def test_router_generate_with_tools_propagates_failure_to_fallback() -> None:
    """Tool-calling requests must use the same primary → fallback chain as before."""
    router = RouterManager(
        primary_provider="failing_primary",
        fallback_provider="echo_fallback",
        primary_kwargs={"tag": "primary"},
        fallback_kwargs={"tag": "fallback"},
    )

    def my_tool(x: int) -> str:
        return str(x)

    router.register_tool("my_tool", my_tool, "echo an int")

    response = await router.generate_with_tools("anything")

    fallback_inner = router.fallback_adapter.adapter  # type: ignore[attr-defined]
    assert response.startswith("fallback:tools:")
    assert fallback_inner.calls == ["anything"]


# =============================================================================
# Streaming routing
# =============================================================================


@pytest.mark.asyncio
async def test_router_streams_trivial_via_local() -> None:
    """Trivial streaming requests must yield from the local adapter."""
    router = RouterManager(
        primary_provider="echo_primary",
        fallback_provider="echo_fallback",
        primary_kwargs={"tag": "primary"},
        fallback_kwargs={"tag": "fallback"},
        local_provider="echo_local",
        local_kwargs={"tag": "local"},
    )

    chunks = []
    async for chunk in router.agenerate_stream("label this as good or bad"):
        chunks.append(chunk)

    local_inner = router.local_adapter.adapter  # type: ignore[attr-defined]
    primary_inner = router.primary_adapter.adapter  # type: ignore[attr-defined]

    assert "".join(chunks) == "abc"
    assert local_inner.calls == ["label this as good or bad"]
    assert primary_inner.calls == []


@pytest.mark.asyncio
async def test_router_streams_complex_via_primary() -> None:
    """Complex streaming requests must skip the local adapter."""
    router = RouterManager(
        primary_provider="echo_primary",
        fallback_provider="echo_fallback",
        primary_kwargs={"tag": "primary"},
        fallback_kwargs={"tag": "fallback"},
        local_provider="echo_local",
        local_kwargs={"tag": "local"},
    )

    chunks = []
    async for chunk in router.agenerate_stream("plan a software architecture for a streaming service"):
        chunks.append(chunk)

    local_inner = router.local_adapter.adapter  # type: ignore[attr-defined]
    primary_inner = router.primary_adapter.adapter  # type: ignore[attr-defined]

    assert "".join(chunks) == "abc"
    assert local_inner.calls == []
    assert primary_inner.calls == ["plan a software architecture for a streaming service"]


@pytest.mark.asyncio
async def test_router_stream_falls_back_to_primary_when_local_fails_before_yield() -> None:
    """A local stream failure before yielding must fall through to the cloud chain."""
    router = RouterManager(
        primary_provider="echo_primary",
        fallback_provider="echo_fallback",
        primary_kwargs={"tag": "primary"},
        fallback_kwargs={"tag": "fallback"},
        local_provider="failing_local",
        local_kwargs={"tag": "local"},
    )

    chunks = []
    async for chunk in router.agenerate_stream("classify this"):
        chunks.append(chunk)

    assert "".join(chunks) == "abc"
    primary_inner = router.primary_adapter.adapter  # type: ignore[attr-defined]
    assert primary_inner.calls == ["classify this"]


# =============================================================================
# Tool registration propagates to local
# =============================================================================


def test_router_register_tool_propagates_to_local() -> None:
    """register_tool must reach every inner adapter, including local."""
    router = RouterManager(
        primary_provider="echo_primary",
        fallback_provider="echo_fallback",
        primary_kwargs={"tag": "primary"},
        fallback_kwargs={"tag": "fallback"},
        local_provider="echo_local",
        local_kwargs={"tag": "local"},
    )

    def my_tool() -> str:
        return "ok"

    router.register_tool("my_tool", my_tool, "does nothing")

    assert "my_tool" in router.primary_adapter.tools
    assert "my_tool" in router.fallback_adapter.tools
    assert router.local_adapter is not None
    assert "my_tool" in router.local_adapter.tools


# =============================================================================
# get_token_count still uses primary
# =============================================================================


@pytest.mark.asyncio
async def test_router_get_token_count_uses_primary() -> None:
    """Token counting stays on the primary adapter — local models aren't queried for this."""
    router = RouterManager(
        primary_provider="echo_primary",
        fallback_provider="echo_fallback",
        primary_kwargs={"tag": "primary"},
        fallback_kwargs={"tag": "fallback"},
        local_provider="echo_local",
        local_kwargs={"tag": "local"},
    )

    # EchoAdapter returns word count. The default behaviour is unchanged.
    count = await router.get_token_count("hello world")
    assert count == 2


# =============================================================================
# Constructor validation
# =============================================================================


def test_router_constructor_without_local_provider_keeps_default_behavior() -> None:
    """Constructing without local_provider preserves the original two-adapter shape."""
    router = RouterManager(
        primary_provider="echo_primary",
        fallback_provider="echo_fallback",
        primary_kwargs={"tag": "primary"},
        fallback_kwargs={"tag": "fallback"},
    )

    assert router.local_adapter is None
    assert router.local_name is None
    assert router.primary_name == "echo_primary"
    assert router.fallback_name == "echo_fallback"