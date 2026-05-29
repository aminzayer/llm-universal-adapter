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
        return response.choices[0].message.content or ""

    async def agenerate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        if "temperature" not in kwargs:
            kwargs["temperature"] = settings.default_temperature

        logger.debug(f"Sending async streaming request to OpenAI using model: {self.model}")
        stream = await self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}], stream=True, **kwargs)
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
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name  # type: ignore
                function_args = json.loads(tool_call.function.arguments or "{}")  # type: ignore

                logger.debug(f"Executing tool {function_name} with args {function_args}")
                tool_data = self.tools.get(function_name)
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
