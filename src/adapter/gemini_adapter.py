import inspect
import logging
from typing import Any, AsyncGenerator, Optional

from google import genai
from google.genai import types  # pyright: ignore[reportMissingImports]
from google.genai.errors import APIError  # pyright: ignore[reportMissingImports]
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from adapter.base import BaseLLMAdapter
from config import settings

logger = logging.getLogger(__name__)


class GeminiAdapter(BaseLLMAdapter):
    """
    Adapter for Google's Gemini language models.
    Handles interactions using the new official google-genai SDK architecture.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash") -> None:
        """
        Initializes the Gemini adapter.

        Args:
            api_key (Optional[str]): The API key for authenticating. Defaults to settings.
            model (str): The model to use for generation (default: 'gemini-2.5-flash').
        """
        super().__init__()
        api_key = api_key or settings.gemini_api_key
        if not api_key:
            raise ValueError("Gemini API key is missing. Please set GEMINI_API_KEY environment variable or pass it directly.")

        # The new SDK instantiates an isolated Client rather than mutating global state
        self.client = genai.Client(api_key=api_key)
        self.model_name = model

    @retry(
        retry=retry_if_exception_type((APIError,)),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Asynchronously generates a text response from the Gemini model.
        Uses exponential backoff to handle API errors and quota exhaustions.

        Args:
            prompt (str): The user's input prompt.
            **kwargs (Any): Additional parameters to pass to the API configuration.

        Returns:
            str: The generated textual response.
        """
        # Map dynamic kwargs to the strict SDK config structure
        generation_config = kwargs.pop("config", None)
        if not generation_config:
            temperature = kwargs.pop("temperature", settings.default_temperature)
            generation_config = types.GenerateContentConfig(temperature=temperature, **kwargs)

        logger.debug(f"Sending async request to Gemini using model: {self.model_name}")

        response = await self.client.aio.models.generate_content(model=self.model_name, contents=prompt, config=generation_config)

        if response.text is None:
            raise ValueError("The model returned an empty response. Check safety filters or prompt validity.")

        return response.text

    async def agenerate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """
        Asynchronously generates a streamed text response from the Gemini model.
        """
        generation_config = kwargs.pop("config", None)
        if not generation_config:
            temperature = kwargs.pop("temperature", settings.default_temperature)
            generation_config = types.GenerateContentConfig(temperature=temperature, **kwargs)

        logger.debug(f"Sending async streaming request to Gemini using model: {self.model_name}")
        response_stream = await self.client.aio.models.generate_content_stream(model=self.model_name, contents=prompt, config=generation_config)

        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    async def get_token_count(self, text: str) -> int:
        """
        Asynchronously calculates the number of tokens for the given text using the updated SDK.

        Args:
            text (str): The text to tokenize.

        Returns:
            int: The number of tokens in the text.
        """
        response = await self.client.aio.models.count_tokens(model=self.model_name, contents=text)
        return response.total_tokens or 0

    async def generate_with_tools(self, prompt: str) -> str:
        """
        Asynchronously executes a prompt allowing the LLM to utilize registered tools.
        """
        if not self.tools:
            return await self.generate_response(prompt)

        # The new SDK supports passing Callables directly
        tools_list = [tool_data["function"] for tool_data in self.tools.values()]

        generation_config = types.GenerateContentConfig(temperature=settings.default_temperature, tools=tools_list)

        logger.debug(f"Sending async request to Gemini with tools using model: {self.model_name}")
        response = await self.client.aio.models.generate_content(model=self.model_name, contents=prompt, config=generation_config)

        if response.function_calls:
            # We need to construct the conversation history manually to send back to the model
            # For Gemini, conversation context is maintained using a list of types.Content
            history = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)]), response.candidates[0].content if response.candidates and response.candidates[0].content else types.Content(role="model", parts=[])]

            tool_responses: list[Any] = []
            for i, tool_call in enumerate(response.function_calls):
                function_name = tool_call.name or ""
                function_args = dict(tool_call.args) if tool_call.args is not None else {}  # type: ignore

                tool_data = self.tools.get(function_name)
                if tool_data and tool_data.get("requires_approval"):
                    if not getattr(self, "redis_client", None):
                        raise RuntimeError("Redis client is not configured for Human-in-the-Loop approvals.")

                    import uuid
                    state_id = f"hitl_{uuid.uuid4().hex}"
                    request_id = f"req_{uuid.uuid4().hex}"

                    # Serialize pending tool calls
                    pending_tool_calls = []
                    for pending_call in response.function_calls[i:]:
                        pending_tool_calls.append({
                            "id": getattr(pending_call, "id", None) or f"call_{uuid.uuid4().hex}",
                            "name": pending_call.name,
                            "args": dict(pending_call.args) if pending_call.args is not None else {}
                        })

                    # Add current tool responses collected so far to history
                    if tool_responses:
                        history.append(types.Content(role="tool", parts=tool_responses))

                    serialized_history = [msg.model_dump() for msg in history]

                    from orchestration.hitl import HITLState, HITLStateManager, ApprovalRequiredError
                    state = HITLState(
                        state_id=state_id,
                        request_id=request_id,
                        provider="gemini",
                        model=self.model_name,
                        temperature=settings.default_temperature,
                        tool_name=function_name,
                        tool_args=function_args,
                        tool_call_id=getattr(tool_call, "id", None) or state_id,
                        messages=serialized_history,
                        pending_tool_calls=pending_tool_calls,
                        status="pending"
                    )

                    manager = HITLStateManager(self.redis_client)
                    await manager.save_state(state)

                    raise ApprovalRequiredError(
                        state_id=state_id,
                        tool_name=function_name,
                        tool_args=function_args,
                        tool_call_id=state.tool_call_id,
                        messages=serialized_history,
                        pending_tool_calls=pending_tool_calls,
                    )

                logger.debug(f"Executing tool {function_name} with args {function_args}")
                if not tool_data:
                    result_dict = {"error": f"Tool {function_name} not found."}
                else:
                    try:
                        func = tool_data["function"]
                        if inspect.iscoroutinefunction(func):
                            tool_result = await func(**function_args)
                        else:
                            tool_result = func(**function_args)
                        result_dict = {"result": tool_result}
                    except Exception as e:
                        result_dict = {"error": f"Error executing {function_name}: {str(e)}"}

                tool_responses.append(types.Part.from_function_response(name=function_name, response=result_dict))

            history.append(types.Content(role="tool", parts=tool_responses))

            logger.debug("Sending async follow-up request to Gemini after tool execution")
            second_response = await self.client.aio.models.generate_content(model=self.model_name, contents=history, config=types.GenerateContentConfig(temperature=settings.default_temperature))

            if second_response.text is None:
                raise ValueError("The model returned an empty response after tool execution.")

            return second_response.text

        return response.text or ""

    async def resume_with_tools(self, state: Any) -> str:
        """
        Resumes a suspended tool execution using the saved state from Redis.
        """
        # Reconstruct Gemini conversation history
        history = [types.Content.model_validate(msg) for msg in state.messages]

        # 1. Execute the approved tool call
        tool_data = self.tools.get(state.tool_name)
        if not tool_data:
            result_dict = {"error": f"Tool {state.tool_name} not found."}
        else:
            try:
                func = tool_data["function"]
                if inspect.iscoroutinefunction(func):
                    tool_result = await func(**state.tool_args)
                else:
                    tool_result = func(**state.tool_args)
                result_dict = {"result": tool_result}
            except Exception as e:
                result_dict = {"error": f"Error executing {state.tool_name}: {str(e)}"}

        tool_responses = [types.Part.from_function_response(name=state.tool_name, response=result_dict)]

        # 2. Process remaining pending tool calls
        remaining_tool_calls = state.pending_tool_calls[1:] if state.pending_tool_calls else []

        for idx, pending_call in enumerate(remaining_tool_calls):
            function_name = pending_call["name"]
            function_args = pending_call["args"]
            p_tool_data = self.tools.get(function_name)

            if p_tool_data and p_tool_data.get("requires_approval"):
                # Suspend again!
                import uuid
                state_id = f"hitl_{uuid.uuid4().hex}"
                next_pending = remaining_tool_calls[idx:]

                # Append current tool responses to history copy
                history.append(types.Content(role="tool", parts=tool_responses))
                serialized_history = [msg.model_dump() for msg in history]

                from orchestration.hitl import HITLState, HITLStateManager, ApprovalRequiredError
                new_state = HITLState(
                    state_id=state_id,
                    request_id=state.request_id,
                    provider="gemini",
                    model=self.model_name,
                    temperature=state.temperature,
                    tool_name=function_name,
                    tool_args=function_args,
                    tool_call_id=pending_call["id"],
                    messages=serialized_history,
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
                    messages=serialized_history,
                    pending_tool_calls=next_pending,
                )

            # If no approval is required, execute immediately
            logger.debug(f"Executing pending tool {function_name} with args {function_args}")
            if not p_tool_data:
                res_dict = {"error": f"Tool {function_name} not found."}
            else:
                try:
                    func = p_tool_data["function"]
                    if inspect.iscoroutinefunction(func):
                        res = await func(**function_args)
                    else:
                        res = func(**function_args)
                    res_dict = {"result": res}
                except Exception as e:
                    res_dict = {"error": f"Error executing {function_name}: {str(e)}"}

            tool_responses.append(types.Part.from_function_response(name=function_name, response=res_dict))

        history.append(types.Content(role="tool", parts=tool_responses))

        # 3. Call follow-up to get the final response
        logger.debug("Sending async follow-up request to Gemini after resuming tool execution")
        second_response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=history,
            config=types.GenerateContentConfig(temperature=settings.default_temperature)
        )
        if second_response.text is None:
            raise ValueError("The model returned an empty response after tool execution.")
        return second_response.text
