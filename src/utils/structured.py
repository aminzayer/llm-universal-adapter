import json
import logging
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from adapter.base import BaseLLMAdapter

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredGenerator:
    """
    A generator that forces the underlying LLM to return responses strictly matching 
    a specified pydantic.BaseModel. It utilizes native structured output features 
    where available and implements an automatic self-correction mechanism using tenacity.
    """

    def __init__(self, adapter: BaseLLMAdapter) -> None:
        """
        Initializes the StructuredGenerator.

        Args:
            adapter (BaseLLMAdapter): The LLM adapter instance to use for generation.
        """
        self.adapter = adapter

    async def generate(self, prompt: str, schema_model: Type[T], **kwargs: Any) -> T:
        """
        Generates a structured response matching the provided Pydantic model.
        Retries automatically on validation or parsing errors, appending the error 
        to the prompt for self-correction.

        Args:
            prompt (str): The initial user prompt.
            schema_model (Type[T]): The Pydantic model class to enforce.
            **kwargs (Any): Additional generation arguments.

        Returns:
            T: An instance of the requested Pydantic model.
        """
        schema_json = schema_model.model_json_schema()
        provider = self._get_provider()

        # Utilize native structured output features if the provider is known
        if provider == "openai":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_model.__name__,
                    "schema": schema_json,
                    "strict": True
                }
            }
        elif provider == "gemini":
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = schema_json

        # Construct the initial prompt enforcing JSON and the schema structure
        current_prompt = (
            f"{prompt}\n\n"
            f"You MUST respond ONLY with a valid JSON object.\n"
            f"The JSON must strictly match the following JSON Schema:\n"
            f"{json.dumps(schema_json, indent=2)}\n"
            f"Do not include markdown formatting blocks like ```json."
        )

        # Define the retry mechanism for self-correction
        retryer = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type((json.JSONDecodeError, ValidationError)),
            reraise=True,
        )

        async for attempt in retryer:
            with attempt:
                logger.debug(f"Sending structured generation request (Attempt {attempt.retry_state.attempt_number}).")
                response_text = await self.adapter.generate_response(current_prompt, **kwargs)

                try:
                    # Strip markdown blocks to prevent JSON parsing failures
                    clean_text = response_text.strip()
                    if clean_text.startswith("```json"):
                        clean_text = clean_text[7:]
                    elif clean_text.startswith("```"):
                        clean_text = clean_text[3:]
                        
                    if clean_text.endswith("```"):
                        clean_text = clean_text[:-3]

                    parsed_json = json.loads(clean_text.strip())
                    
                    # Validate against the Pydantic model
                    return schema_model.model_validate(parsed_json)
                    
                except (json.JSONDecodeError, ValidationError) as e:
                    error_msg = str(e)
                    logger.warning(f"Structured output validation failed: {error_msg}")
                    
                    # Append the exact validation error to the prompt for self-correction
                    current_prompt += (
                        f"\n\nPrevious attempt failed with the following error:\n{error_msg}\n"
                        f"Please correct your specific mistake and return a valid JSON object strictly matching the schema."
                    )
                    raise

        # Fallback (should not be reached due to reraise=True in Retrying)
        raise RuntimeError("Failed to generate structured output after retries.")

    def _get_provider(self) -> str:
        """
        Attempts to identify the underlying LLM provider to apply native structured output configs.
        """
        if hasattr(self.adapter, "provider"):
            return str(getattr(self.adapter, "provider")).lower()
            
        inner_adapter = getattr(self.adapter, "adapter", self.adapter)
        cls_name = inner_adapter.__class__.__name__.lower()
        
        if "openai" in cls_name:
            return "openai"
        if "gemini" in cls_name:
            return "gemini"
            
        return "unknown"
