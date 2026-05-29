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

            tool_responses = []
            for tool_call in response.function_calls:
                function_name = tool_call.name or ""

                # Convert the arguments to a dict if it's not already
                function_args = dict(tool_call.args) if tool_call.args is not None else {}  # type: ignore

                logger.debug(f"Executing tool {function_name} with args {function_args}")
                tool_data = self.tools.get(function_name)

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

        if response.text is None:
            raise ValueError("The model returned an empty response.")

        return response.text
