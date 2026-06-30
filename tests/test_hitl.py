from typing import Any, AsyncGenerator, List, Optional
import pytest
from pydantic import ValidationError

from adapter.base import BaseLLMAdapter
from orchestration.hitl import HITLState, HITLStateManager, ApprovalRequiredError
from orchestration.router import RouterManager


class MockRedis:
    """
    In-memory mock for Redis cache.
    """
    def __init__(self) -> None:
        self.store = {}

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> None:
        self.store[key] = value

    async def get(self, key: str) -> Optional[str]:
        return self.store.get(key)

    async def delete(self, key: str) -> None:
        if key in self.store:
            del self.store[key]


class DummyToolAdapter(BaseLLMAdapter):
    """
    Mock adapter to test tool execution and state suspension.
    """
    def __init__(self, responses: Optional[List[Any]] = None) -> None:
        super().__init__()
        self.responses = responses or []
        self.call_count = 0

    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        res = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return res

    async def agenerate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        yield ""

    async def get_token_count(self, text: str) -> int:
        return len(text.split())

    async def generate_with_tools(self, prompt: str) -> str:
        # Simulate an LLM output that calls a tool
        # In a real scenario, this matches OpenAI's completion behavior.
        # We loop through registered tools.
        for name, tool_data in self.tools.items():
            if tool_data.get("requires_approval"):
                import uuid
                state_id = f"hitl_{uuid.uuid4().hex}"
                request_id = "req_test"
                messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": name, "arguments": "{}"}}]}]
                
                state = HITLState(
                    state_id=state_id,
                    request_id=request_id,
                    provider="openai",
                    model="gpt-4o",
                    tool_name=name,
                    tool_args={},
                    tool_call_id="call_1",
                    messages=messages,
                    pending_tool_calls=[{"id": "call_1", "type": "function", "function": {"name": name, "arguments": "{}"}}],
                    status="pending"
                )
                manager = HITLStateManager(self.redis_client)
                await manager.save_state(state)
                raise ApprovalRequiredError(
                    state_id=state_id,
                    tool_name=name,
                    tool_args={},
                    tool_call_id="call_1",
                    messages=messages,
                    pending_tool_calls=state.pending_tool_calls,
                )
        return "No tools executed"

    async def resume_with_tools(self, state: Any) -> str:
        # Simulate execution of the approved tool and subsequent completion
        tool_data = self.tools.get(state.tool_name)
        if not tool_data:
            return "Tool not found"
        func = tool_data["function"]
        result = func(**state.tool_args)
        return f"Resumed output with tool result: {result}"


def test_tool_registration_requires_approval() -> None:
    adapter = DummyToolAdapter([])
    
    def my_tool():
        return "success"

    adapter.register_tool("my_tool", my_tool, "A test tool", requires_approval=True)
    assert adapter.tools["my_tool"]["requires_approval"] is True
    assert adapter.tools["my_tool"]["function"] == my_tool


def test_pydantic_validation_strict() -> None:
    # Invalid missing fields should raise ValidationError
    with pytest.raises(ValidationError):
        # pyrefly: ignore [missing-argument]
        HITLState(state_id="123")  # missing required fields like request_id, provider, etc.

    # Valid initialization
    state = HITLState(
        state_id="state_123",
        request_id="req_123",
        provider="openai",
        model="gpt-4o",
        tool_name="test_tool",
        tool_args={"arg": "val"},
        status="pending"
    )
    assert state.state_id == "state_123"
    assert state.status == "pending"


@pytest.mark.asyncio
async def test_hitl_state_manager_lifecycle() -> None:
    redis_client = MockRedis()
    # pyrefly: ignore [bad-argument-type]
    manager = HITLStateManager(redis_client)

    state = HITLState(
        state_id="state_123",
        request_id="req_123",
        provider="openai",
        model="gpt-4o",
        tool_name="test_tool",
        tool_args={"arg": "val"},
        status="pending"
    )

    await manager.save_state(state)
    loaded = await manager.get_state("state_123")
    assert loaded is not None
    assert loaded.state_id == "state_123"
    assert loaded.tool_name == "test_tool"
    assert loaded.tool_args == {"arg": "val"}

    # Update status
    await manager.update_status("state_123", "approved")
    updated = await manager.get_state("state_123")
    # pyrefly: ignore [missing-attribute]
    assert updated.status == "approved"

    # Delete state
    await manager.delete_state("state_123")
    deleted = await manager.get_state("state_123")
    assert deleted is None


@pytest.mark.asyncio
async def test_tool_suspension_raises_exception() -> None:
    redis_client = MockRedis()
    adapter = DummyToolAdapter([])
    adapter.set_redis_client(redis_client)

    def my_approval_tool():
        return "secret"

    adapter.register_tool("my_approval_tool", my_approval_tool, "Description", requires_approval=True)

    with pytest.raises(ApprovalRequiredError) as exc_info:
        await adapter.generate_with_tools("Run my approval tool")

    assert exc_info.value.tool_name == "my_approval_tool"
    assert exc_info.value.state_id.startswith("hitl_")

    # Verify that the state was saved in Redis
    # pyrefly: ignore [bad-argument-type]
    manager = HITLStateManager(redis_client)
    saved_state = await manager.get_state(exc_info.value.state_id)
    assert saved_state is not None
    assert saved_state.tool_name == "my_approval_tool"
    assert saved_state.status == "pending"


@pytest.mark.asyncio
async def test_router_propagation_and_resumption() -> None:
    from adapter.factory import LLMAdapterFactory
    LLMAdapterFactory.register_adapter("primary_test", DummyToolAdapter)
    LLMAdapterFactory.register_adapter("fallback_test", DummyToolAdapter)

    redis_client = MockRedis()
    
    primary = DummyToolAdapter([])
    fallback = DummyToolAdapter([])
    
    router = RouterManager(
        primary_provider="primary_test",
        fallback_provider="fallback_test",
    )
    # Manually override the factory-created adapters with our test dummies
    router.primary_adapter = primary
    router.fallback_adapter = fallback
    router.primary_name = "primary_test"
    router.fallback_name = "fallback_test"
    
    router.set_redis_client(redis_client)
    assert primary.redis_client == redis_client
    assert fallback.redis_client == redis_client

    def secret_action(val: str = "yes"):
        return f"done_{val}"

    # Register tool propagates requires_approval
    router.register_tool("secret_action", secret_action, "Executes a secret action", requires_approval=True)
    assert primary.tools["secret_action"]["requires_approval"] is True
    assert fallback.tools["secret_action"]["requires_approval"] is True

    # Test resumption
    state = HITLState(
        state_id="state_abc",
        request_id="req_abc",
        provider="primary_test",
        model="gpt-4o",
        tool_name="secret_action",
        tool_args={"val": "approved"},
        status="approved"
    )

    res = await router.resume_with_tools(state)
    assert "done_approved" in res
