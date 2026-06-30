import logging
from typing import Any, Optional, AsyncGenerator

import openai
import tiktoken
import json
import inspect
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseLLMAdapter
from config import settings

logger = logging.getLogger(__name__)


class OpenAIAdapter(BaseLLMAdapter):
    """
    Adapter for OpenAI's language models.
    Handles interactions with the OpenAI API, including generating responses
    and calculating token counts using tiktoken.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o") -> None:
        """
        Initializes the OpenAI adapter.

        Args:
            api_key (Optional[str]): The API key for authenticating with OpenAI. Defaults to settings.
            model (str): The model to use for generation (default: 'gpt-4o').
        """
        api_key = api_key or settings.openai_api_key
        if not api_key:
            raise ValueError("OpenAI API key is missing. Please set OPENAI_API_KEY environment variable or pass it directly.")

        self.client = openai.AsyncClient(api_key=api_key)
        self.model = model
        super().__init__()

    @retry(
        retry=retry_if_exception_type((
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.InternalServerError,
        )),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a text response from the OpenAI model.
        Uses exponential backoff for handling rate limits and connection errors.

        Args:
            prompt (str): The user's input prompt.
            **kwargs (Any): Additional parameters to pass to the API (e.g., temperature, max_tokens).

        Returns:
            str: The generated textual response.
        """
        if "temperature" not in kwargs:
            kwargs["temperature"] = settings.default_temperature

        logger.debug(f"Sending async request to OpenAI using model: {self.model}")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            **kwargs,
        )
        # pyrefly: ignore [missing-attribute]
        return response.choices[0].message.content or ""

    async def agenerate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        if "temperature" not in kwargs:
            kwargs["temperature"] = settings.default_temperature

        logger.debug(f"Sending async streaming request to OpenAI using model: {self.model}")
        stream = await self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], stream=True, **kwargs)
        # pyrefly: ignore [not-iterable]
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def get_token_count(self, text: str) -> int:
        """
        Calculates the number of tokens for the given text using tiktoken.

        Args:
            text (str): The text to tokenize.

        Returns:
            int: The number of tokens in the text.
        """
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fallback encoding if the exact model is not found
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))

    def _build_openai_tool_schema(self, name: str, func: Any, description: str) -> Any:
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
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        }

    async def generate_with_tools(self, prompt: str) -> str:
        """
        Executes a prompt allowing the LLM to utilize registered tools.
        """
        if not self.tools:
            return await self.generate_response(prompt)

        openai_tools = [self._build_openai_tool_schema(name, tool_data["function"], tool_data["description"]) for name, tool_data in self.tools.items()]

        messages: list[Any] = [{"role": "user", "content": prompt}]

        logger.debug(f"Sending async request to OpenAI with tools using model: {self.model}")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # pyright: ignore
            tools=openai_tools,
            temperature=settings.default_temperature,
        )

        message = response.choices[0].message
        messages.append(message)  # pyright: ignore

        if message.tool_calls:
            for i, tool_call in enumerate(message.tool_calls):
                function_name = tool_call.function.name  # type: ignore
                function_args = json.loads(tool_call.function.arguments or "{}")  # type: ignore

                tool_data = self.tools.get(function_name)
                if tool_data and tool_data.get("requires_approval"):
                    if not getattr(self, "redis_client", None):
                        raise RuntimeError("Redis client is not configured for Human-in-the-Loop approvals.")

                    # Serialize pending tool calls (subsequent ones in the list)
                    pending_tool_calls = []
                    for pending_call in message.tool_calls[i:]:
                        pending_tool_calls.append({
                            "id": pending_call.id,
                            "type": pending_call.type,
                            "function": {
                                "name": pending_call.function.name,  # type: ignore
                                "arguments": pending_call.function.arguments,  # type: ignore
                            }
                        })

                    import uuid
                    state_id = f"hitl_{uuid.uuid4().hex}"
                    request_id = f"req_{uuid.uuid4().hex}"

                    # Serialize history messages up to this point
                    serialized_messages = []
                    for m in messages[:-1]:
                        if isinstance(m, dict):
                            serialized_messages.append(m)
                        else:
                            # Dump the Assistant / User message
                            serialized_messages.append(m.model_dump() if hasattr(m, "model_dump") else dict(m))

                    # Append the assistant message itself (with tool calls)
                    tool_calls_list = []
                    for tc in message.tool_calls:
                        tool_calls_list.append({
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,  # type: ignore
                                "arguments": tc.function.arguments,  # type: ignore
                            }
                        })
                    serialized_messages.append({
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": tool_calls_list
                    })

                    from orchestration.hitl import HITLState, HITLStateManager, ApprovalRequiredError
                    state = HITLState(
                        state_id=state_id,
                        request_id=request_id,
                        provider="openai",
                        model=self.model,
                        temperature=settings.default_temperature,
                        tool_name=function_name,
                        tool_args=function_args,
                        tool_call_id=tool_call.id,
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
                        tool_call_id=tool_call.id,
                        messages=serialized_messages,
                        pending_tool_calls=pending_tool_calls,
                    )

                logger.debug(f"Executing tool {function_name} with args {function_args}")
                if not tool_data:
                    tool_result_str = f"Error: Tool {function_name} not found."
                else:
                    try:
                        func = tool_data["function"]
                        if inspect.iscoroutinefunction(func):
                            tool_result = await func(**function_args)
                        else:
                            tool_result = func(**function_args)
                        tool_result_str = str(tool_result)
                    except Exception as e:
                        tool_result_str = f"Error executing {function_name}: {e}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result_str,
                })

            # Make a second call to get the final augmented response
            logger.debug("Sending async follow-up request to OpenAI after tool execution")
            second_response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # pyright: ignore
                temperature=settings.default_temperature,
            )
            return second_response.choices[0].message.content or ""

        return message.content or ""

    async def resume_with_tools(self, state: Any) -> str:
        """
        Resumes a suspended tool execution using the saved state from Redis.
        """
        messages = list(state.messages)

        # 1. Execute the tool that was approved
        tool_data = self.tools.get(state.tool_name)
        if not tool_data:
            tool_result_str = f"Error: Tool {state.tool_name} not found."
        else:
            try:
                func = tool_data["function"]
                if inspect.iscoroutinefunction(func):
                    tool_result = await func(**state.tool_args)
                else:
                    tool_result = func(**state.tool_args)
                tool_result_str = str(tool_result)
            except Exception as e:
                tool_result_str = f"Error executing {state.tool_name}: {e}"

        # 2. Append the tool response
        messages.append({
            "role": "tool",
            "tool_call_id": state.tool_call_id,
            "content": tool_result_str,
        })

        # 3. Process remaining pending tool calls
        # Note: the first item in pending_tool_calls was the one we just executed (index 0)
        remaining_tool_calls = state.pending_tool_calls[1:] if state.pending_tool_calls else []

        for idx, pending_call in enumerate(remaining_tool_calls):
            function_name = pending_call["function"]["name"]
            function_args = json.loads(pending_call["function"]["arguments"] or "{}")
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
                    provider="openai",
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
            else:
                try:
                    func = p_tool_data["function"]
                    if inspect.iscoroutinefunction(func):
                        res = await func(**function_args)
                    else:
                        res = func(**function_args)
                    res_str = str(res)
                except Exception as e:
                    res_str = f"Error executing {function_name}: {e}"

            messages.append({
                "role": "tool",
                "tool_call_id": pending_call["id"],
                "content": res_str,
            })

        # 4. Make a follow-up completion call to get the final response
        logger.debug("Sending async follow-up request to OpenAI after resuming tool execution")
        second_response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # pyright: ignore
            temperature=settings.default_temperature,
        )
        return second_response.choices[0].message.content or ""
