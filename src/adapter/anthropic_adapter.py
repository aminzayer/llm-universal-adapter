import inspect
import logging
from typing import Any, AsyncGenerator, Optional

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseLLMAdapter
from config import settings

logger = logging.getLogger(__name__)


class AnthropicAdapter(BaseLLMAdapter):
    """
    Adapter for Anthropic's Claude models.
    Handles interactions using the official anthropic SDK.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022") -> None:
        """
        Initializes the Anthropic adapter.

        Args:
            api_key (Optional[str]): The API key for authenticating. Defaults to settings.anthropic_api_key.
            model (str): The model to use for generation (default: 'claude-3-5-sonnet-20241022').
        """
        super().__init__()
        api_key = api_key or settings.anthropic_api_key
        if not api_key:
            raise ValueError("Anthropic API key is missing. Please set ANTHROPIC_API_KEY environment variable or pass it directly.")

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    @retry(
        retry=retry_if_exception_type((
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        )),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Asynchronously generates a text response from the Anthropic model.
        Uses exponential backoff to handle API errors and quota exhaustions.

        Args:
            prompt (str): The user's input prompt.
            **kwargs (Any): Additional parameters to pass to the API configuration.

        Returns:
            str: The generated textual response.
        """
        if "temperature" not in kwargs:
            kwargs["temperature"] = settings.default_temperature

        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 1024

        logger.debug(f"Sending async request to Anthropic using model: {self.model}")
        response = await self.client.messages.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            **kwargs,
        )

        text_content = ""
        # pyrefly: ignore [missing-attribute]
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_content += getattr(block, "text", "")
        return text_content

    async def agenerate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """
        Asynchronously generates a streamed text response from the Anthropic model.
        """
        if "temperature" not in kwargs:
            kwargs["temperature"] = settings.default_temperature

        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 1024

        logger.debug(f"Sending async streaming request to Anthropic using model: {self.model}")
        async with self.client.messages.stream(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        ) as stream:
            async for chunk in stream.text_stream:
                yield chunk

    async def get_token_count(self, text: str) -> int:
        """
        Asynchronously calculates the number of tokens for the given text.

        Args:
            text (str): The text to tokenize.

        Returns:
            int: The number of tokens in the text.
        """
        response = await self.client.messages.count_tokens(
            model=self.model,
            messages=[{"role": "user", "content": text}]
        )
        return response.input_tokens

    def _build_anthropic_tool_schema(self, name: str, func: Any, description: str) -> Any:
        sig = inspect.signature(func)
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            param_type = "string"  # Default
            if param.annotation is int:
                param_type = "integer"
            elif param.annotation is float:
                param_type = "number"
            elif param.annotation is bool:
                param_type = "boolean"
            elif param.annotation is list:
                param_type = "array"

            parameters["properties"][param_name] = {"type": param_type}
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)

        return {
            "name": name,
            "description": description,
            "input_schema": parameters,
        }

    async def generate_with_tools(self, prompt: str) -> str:
        """
        Asynchronously executes a prompt allowing the LLM to utilize registered tools.
        """
        if not self.tools:
            return await self.generate_response(prompt)

        anthropic_tools = [self._build_anthropic_tool_schema(name, tool_data["function"], tool_data["description"]) for name, tool_data in self.tools.items()]

        messages: list[Any] = [{"role": "user", "content": prompt}]

        logger.debug(f"Sending async request to Anthropic with tools using model: {self.model}")
        response = await self.client.messages.create(
            model=self.model,
            messages=messages,
            tools=anthropic_tools,
            temperature=settings.default_temperature,
            max_tokens=4096,
        )

        assistant_content = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                assistant_content.append({
                    "type": "text",
                    "text": getattr(block, "text", "")
                })
            elif block_type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": getattr(block, "input", {})
                })

        messages.append({
            "role": "assistant",
            "content": assistant_content
        })

        tool_calls = [block for block in response.content if getattr(block, "type", None) == "tool_use"]

        if tool_calls:
            for i, tool_call in enumerate(tool_calls):
                tc_any: Any = tool_call
                function_name = tc_any.name
                function_args = tc_any.input

                tool_data = self.tools.get(function_name)
                if tool_data and tool_data.get("requires_approval"):
                    if not getattr(self, "redis_client", None):
                        raise RuntimeError("Redis client is not configured for Human-in-the-Loop approvals.")

                    # Serialize pending tool calls (subsequent ones in the list)
                    pending_tool_calls = []
                    for pending_call in tool_calls[i:]:
                        pc_any: Any = pending_call
                        pending_tool_calls.append({
                            "id": pc_any.id,
                            "type": "tool_use",
                            "name": pc_any.name,
                            "input": pc_any.input,
                        })

                    import uuid
                    state_id = f"hitl_{uuid.uuid4().hex}"
                    request_id = f"req_{uuid.uuid4().hex}"

                    # Serialize history messages up to this point
                    serialized_messages = []
                    for m in messages:
                        if isinstance(m, dict):
                            serialized_messages.append(m)
                        else:
                            serialized_messages.append(m.model_dump() if hasattr(m, "model_dump") else dict(m))

                    from orchestration.hitl import HITLState, HITLStateManager, ApprovalRequiredError
                    state = HITLState(
                        state_id=state_id,
                        request_id=request_id,
                        provider="anthropic",
                        model=self.model,
                        temperature=settings.default_temperature,
                        tool_name=function_name,
                        tool_args=function_args,
                        tool_call_id=tc_any.id,
                        messages=serialized_messages,
                        pending_tool_calls=pending_tool_calls,
                        status="pending"
                    )

                    manager = HITLStateManager(self.redis_client)
                    await manager.save_state(state)

                    raise ApprovalRequiredError(
                        state_id=state_id,
                        tool_name=function_name,
                        tool_args=function_args,
                        tool_call_id=tc_any.id,
                        messages=serialized_messages,
                        pending_tool_calls=pending_tool_calls,
                    )

                logger.debug(f"Executing tool {function_name} with args {function_args}")
                if not tool_data:
                    tool_result_str = f"Error: Tool {function_name} not found."
                    is_error = True
                else:
                    try:
                        func = tool_data["function"]
                        if inspect.iscoroutinefunction(func):
                            tool_result = await func(**function_args)
                        else:
                            tool_result = func(**function_args)
                        tool_result_str = str(tool_result)
                        is_error = False
                    except Exception as e:
                        tool_result_str = f"Error executing {function_name}: {e}"
                        is_error = True

                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": tc_any.id,
                    "content": tool_result_str,
                }
                if is_error:
                    tool_result_block["is_error"] = True

                # Add to messages
                if messages and messages[-1].get("role") == "user":
                    last_content = messages[-1]["content"]
                    if isinstance(last_content, str):
                        messages[-1]["content"] = [
                            {"type": "text", "text": last_content},
                            tool_result_block
                        ]
                    elif isinstance(last_content, list):
                        messages[-1]["content"].append(tool_result_block)
                    else:
                        messages[-1]["content"] = [tool_result_block]
                else:
                    messages.append({
                        "role": "user",
                        "content": [tool_result_block]
                    })

            # Make a second call to get the final augmented response
            logger.debug("Sending async follow-up request to Anthropic after tool execution")
            second_response = await self.client.messages.create(
                model=self.model,
                messages=messages,
                tools=anthropic_tools,
                temperature=settings.default_temperature,
                max_tokens=4096,
            )

            text_content = ""
            for block in second_response.content:
                if getattr(block, "type", None) == "text":
                    text_content += getattr(block, "text", "")
            return text_content

        text_content = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_content += getattr(block, "text", "")
        return text_content

    async def resume_with_tools(self, state: Any) -> str:
        """
        Resumes a suspended tool execution using the saved state from Redis.
        """
        messages = list(state.messages)

        # 1. Execute the tool that was approved
        tool_data = self.tools.get(state.tool_name)
        if not tool_data:
            tool_result_str = f"Error: Tool {state.tool_name} not found."
            is_error = True
        else:
            try:
                func = tool_data["function"]
                if inspect.iscoroutinefunction(func):
                    tool_result = await func(**state.tool_args)
                else:
                    tool_result = func(**state.tool_args)
                tool_result_str = str(tool_result)
                is_error = False
            except Exception as e:
                tool_result_str = f"Error executing {state.tool_name}: {e}"
                is_error = True

        tool_result_block = {
            "type": "tool_result",
            "tool_use_id": state.tool_call_id,
            "content": tool_result_str,
        }
        if is_error:
            tool_result_block["is_error"] = True

        # Append the tool response
        if messages and messages[-1].get("role") == "user":
            last_content = messages[-1]["content"]
            if isinstance(last_content, str):
                messages[-1]["content"] = [
                    {"type": "text", "text": last_content},
                    tool_result_block
                ]
            elif isinstance(last_content, list):
                messages[-1]["content"].append(tool_result_block)
            else:
                messages[-1]["content"] = [tool_result_block]
        else:
            messages.append({
                "role": "user",
                "content": [tool_result_block]
            })

        # 2. Process remaining pending tool calls
        # Note: the first item in pending_tool_calls was the one we just executed (index 0)
        remaining_tool_calls = state.pending_tool_calls[1:] if state.pending_tool_calls else []

        for idx, pending_call in enumerate(remaining_tool_calls):
            function_name = pending_call["name"]
            function_args = pending_call["input"]
            p_tool_data = self.tools.get(function_name)

            if p_tool_data and p_tool_data.get("requires_approval"):
                # Suspend again!
                import uuid
                state_id = f"hitl_{uuid.uuid4().hex}"
                next_pending = remaining_tool_calls[idx:]

                from orchestration.hitl import HITLState, HITLStateManager, ApprovalRequiredError
                new_state = HITLState(
                    state_id=state_id,
                    request_id=state.request_id,
                    provider="anthropic",
                    model=self.model,
                    temperature=state.temperature,
                    tool_name=function_name,
                    tool_args=function_args,
                    tool_call_id=pending_call["id"],
                    messages=messages,
                    pending_tool_calls=next_pending,
                    status="pending"
                )

                manager = HITLStateManager(self.redis_client)
                await manager.save_state(new_state)

                raise ApprovalRequiredError(
                    state_id=state_id,
                    tool_name=function_name,
                    tool_args=function_args,
                    tool_call_id=pending_call["id"],
                    messages=messages,
                    pending_tool_calls=next_pending,
                )

            # If no approval is required, execute immediately
            logger.debug(f"Executing pending tool {function_name} with args {function_args}")
            if not p_tool_data:
                res_str = f"Error: Tool {function_name} not found."
                is_error = True
            else:
                try:
                    func = p_tool_data["function"]
                    if inspect.iscoroutinefunction(func):
                        res = await func(**function_args)
                    else:
                        res = func(**function_args)
                    res_str = str(res)
                    is_error = False
                except Exception as e:
                    res_str = f"Error executing {function_name}: {e}"
                    is_error = True

            p_tool_result_block = {
                "type": "tool_result",
                "tool_use_id": pending_call["id"],
                "content": res_str,
            }
            if is_error:
                p_tool_result_block["is_error"] = True

            if messages and messages[-1].get("role") == "user":
                last_content = messages[-1]["content"]
                if isinstance(last_content, str):
                    messages[-1]["content"] = [
                        {"type": "text", "text": last_content},
                        p_tool_result_block
                    ]
                elif isinstance(last_content, list):
                    messages[-1]["content"].append(p_tool_result_block)
                else:
                    messages[-1]["content"] = [p_tool_result_block]
            else:
                messages.append({
                    "role": "user",
                    "content": [p_tool_result_block]
                })

        # 3. Call follow-up to get the final response
        logger.debug("Sending async follow-up request to Anthropic after resuming tool execution")

        anthropic_tools = [self._build_anthropic_tool_schema(name, tool_data["function"], tool_data["description"]) for name, tool_data in self.tools.items()]

        second_response = await self.client.messages.create(
            model=self.model,
            messages=messages,
            tools=anthropic_tools,
            temperature=settings.default_temperature,
            max_tokens=4096,
        )

        text_content = ""
        for block in second_response.content:
            if getattr(block, "type", None) == "text":
                text_content += getattr(block, "text", "")
        return text_content
