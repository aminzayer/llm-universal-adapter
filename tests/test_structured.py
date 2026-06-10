from typing import Any, AsyncGenerator

import pytest
from pydantic import BaseModel, Field, ValidationError

from adapter.base import BaseLLMAdapter
from utils.structured import StructuredGenerator


class UserProfile(BaseModel):
    """A dummy Pydantic model for structured testing."""
    name: str
    age: int = Field(gt=0)


class MockStructuredAdapter(BaseLLMAdapter):
    """Mock adapter simulating sequential LLM outputs to test retry capabilities."""
    
    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self.responses = responses
        self.call_count = 0
        self.prompts: list[str] = []

    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        res = self.responses[self.call_count]
        self.call_count += 1
        return res

    async def agenerate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        yield ""

    async def get_token_count(self, text: str) -> int:
        return len(text.split())

    async def generate_with_tools(self, prompt: str) -> str:
        return ""


@pytest.mark.asyncio
async def test_structured_generator_success() -> None:
    """Test that a valid JSON response parses into the model on the first attempt."""
    adapter = MockStructuredAdapter(['```json\n{"name": "Alice", "age": 30}\n```'])
    generator = StructuredGenerator(adapter)
    
    result = await generator.generate("Extract user details", UserProfile)
    
    assert result.name == "Alice"
    assert result.age == 30
    assert adapter.call_count == 1
    assert "You MUST respond ONLY with a valid JSON object." in adapter.prompts[0]


@pytest.mark.asyncio
async def test_structured_generator_json_error_retry() -> None:
    """Test that a JSON decoding error triggers a retry with the appended error message."""
    adapter = MockStructuredAdapter(['This is not JSON', '{"name": "Bob", "age": 25}'])
    generator = StructuredGenerator(adapter)
    
    result = await generator.generate("Extract user details", UserProfile)
    
    assert result.name == "Bob"
    assert result.age == 25
    assert adapter.call_count == 2
    assert "Previous attempt failed with the following error" in adapter.prompts[1]


@pytest.mark.asyncio
async def test_structured_generator_validation_error_exhaust() -> None:
    """Test that a persisting validation error properly exhausts retries and raises."""
    # "age" < 0 fails the Field(gt=0) check repeatedly
    adapter = MockStructuredAdapter(['{"name": "Dave", "age": -5}'] * 4)
    generator = StructuredGenerator(adapter)
    
    with pytest.raises(ValidationError):
        await generator.generate("Extract user details", UserProfile)
        
    # tenacity tries 3 times by default as specified
    assert adapter.call_count == 3
