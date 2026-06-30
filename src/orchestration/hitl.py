import json
import logging
from typing import Any, Dict, List, Optional
import redis.asyncio as redis
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class HITLState(BaseModel):
    """
    Pydantic model representing the serialized execution state of a suspended tool execution.
    Enforces strict type validation on all fields.
    """
    state_id: str = Field(..., description="Unique state identifier generated for this suspension.")
    request_id: str = Field(..., description="The original request correlation ID.")
    provider: str = Field(..., description="The adapter provider name, e.g., 'openai' or 'gemini'.")
    model: str = Field(..., description="The LLM model name being used.")
    temperature: float = Field(default=0.7, description="The temperature parameter.")
    tool_name: str = Field(..., description="The name of the tool that requires approval.")
    tool_args: Dict[str, Any] = Field(..., description="The arguments passed to the tool.")
    tool_call_id: Optional[str] = Field(default=None, description="The tool call correlation ID.")
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="The message history up to the tool call.")
    pending_tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Remaining tool calls in this execution step.")
    task: Optional[Dict[str, Any]] = Field(default=None, description="Optional swarm task metadata.")
    worker_name: Optional[str] = Field(default=None, description="Optional swarm worker name.")
    status: str = Field(default="pending", description="Approval status: pending, approved, or aborted.")


class ApprovalRequiredError(Exception):
    """
    Exception raised when a tool requires human approval.
    """
    def __init__(
        self,
        state_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_call_id: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        pending_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.state_id = state_id
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_call_id = tool_call_id
        self.messages = messages or []
        self.pending_tool_calls = pending_tool_calls or []
        super().__init__(f"Tool '{tool_name}' requires approval (state_id: {state_id})")


class HITLStateManager:
    """
    Manages suspension, storage, and resumption of agent execution state in Redis.
    """
    def __init__(self, redis_client: redis.Redis) -> None:
        """
        Initializes the HITL State Manager.

        Args:
            redis_client: An active redis.asyncio.Redis connection client.
        """
        self.redis = redis_client

    async def save_state(self, state: HITLState, ttl: int = 86400) -> None:
        """
        Saves a HITLState to Redis with a TTL.

        Args:
            state: The validated HITLState model.
            ttl: Time-to-live in seconds (default: 24 hours).
        """
        key = f"hitl:state:{state.state_id}"
        await self.redis.set(key, state.model_dump_json(), ex=ttl)
        logger.info(f"Successfully saved HITL state '{state.state_id}' for tool '{state.tool_name}'")

    async def get_state(self, state_id: str) -> Optional[HITLState]:
        """
        Loads and validates a HITLState from Redis.

        Args:
            state_id: The unique state identifier.

        Returns:
            Optional[HITLState]: The validated state or None if not found.
        """
        key = f"hitl:state:{state_id}"
        raw_data = await self.redis.get(key)
        if not raw_data:
            return None

        try:
            parsed = json.loads(raw_data)
            return HITLState.model_validate(parsed)
        except (ValidationError, json.JSONDecodeError) as e:
            logger.error(f"Failed strict validation for state '{state_id}': {e}")
            raise ValueError(f"Invalid serialized HITL state: {e}") from e

    async def update_status(self, state_id: str, status: str) -> None:
        """
        Updates the status of a suspended HITL state.

        Args:
            state_id: The unique state identifier.
            status: The new status string.
        """
        state = await self.get_state(state_id)
        if not state:
            raise ValueError(f"HITL state '{state_id}' not found.")
        state.status = status
        await self.save_state(state)

    async def delete_state(self, state_id: str) -> None:
        """
        Deletes the HITL state from Redis.

        Args:
            state_id: The unique state identifier.
        """
        key = f"hitl:state:{state_id}"
        await self.redis.delete(key)
        logger.info(f"Deleted HITL state '{state_id}'")
