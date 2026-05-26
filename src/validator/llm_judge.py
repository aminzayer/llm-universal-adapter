import json
import logging
from typing import Any, Dict

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.adapter.base import BaseLLMAdapter

logger = logging.getLogger(__name__)


class StrictValidator:
    """
    A validator that uses an LLM to evaluate and score textual content.
    Enforces strict JSON output format.
    """

    def __init__(self, llm_adapter: BaseLLMAdapter) -> None:
        """
        Initializes the StrictValidator.

        Args:
            llm_adapter (BaseLLMAdapter): The LLM adapter instance to use for evaluation.
        """
        self.llm_adapter = llm_adapter

    @retry(
        retry=retry_if_exception_type((json.JSONDecodeError, ValueError)),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def evaluate(self, content: str, criteria: str) -> Dict[str, Any]:
        """
        Evaluates the content based on the given criteria and returns a JSON result.
        Retries automatically if the LLM fails to return parsable JSON.

        Args:
            content (str): The text content to evaluate.
            criteria (str): The criteria for evaluation.

        Returns:
            Dict[str, Any]: The evaluation result in JSON format.
        """
        prompt = (
            f"Evaluate the following content based on this criteria: {criteria}\n\n"
            f"Content:\n{content}\n\n"
            "You MUST respond ONLY with a valid JSON object containing the evaluation results. "
            "The JSON must have the exact following keys:\n"
            "- 'score': a number between 0 and 10 representing the overall score.\n"
            "- 'reasoning': a string explaining the reasoning for the score.\n"
            "- 'is_valid': a boolean indicating if the content meets the criteria.\n"
            "Do not include markdown blocks or any other text outside the JSON."
        )

        logger.debug("Sending evaluation prompt to LLM.")
        response = self.llm_adapter.generate_response(prompt)

        try:
            # Clean potential markdown formatting
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            elif clean_response.startswith("```"):
                clean_response = clean_response[3:]
            
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
                
            clean_response = clean_response.strip()

            result = json.loads(clean_response)

            # Validate required keys
            if not all(k in result for k in ("score", "reasoning", "is_valid")):
                raise ValueError("Missing required keys in LLM JSON response.")

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {response}")
            raise e
        except ValueError as e:
            logger.warning(str(e))
            raise e
