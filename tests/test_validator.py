import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# pyrefly: ignore [missing-import]
from src.validator.llm_judge import StrictValidator


@pytest.fixture(autouse=True)
def fast_retries():
    """
    Fixture to mock `time.sleep` used by `tenacity` during backoff.
    This ensures tests involving retries run instantaneously.
    """
    with patch("tenacity.nap.time.sleep"), patch("asyncio.sleep"):
        yield


@pytest.mark.asyncio
async def test_strict_validator_success() -> None:
    """Test successful JSON parsing on the first attempt."""
    mock_adapter = MagicMock()
    mock_adapter.generate_response = AsyncMock(return_value='{"score": 8, "reasoning": "Good", "is_valid": true}')

    validator = StrictValidator(llm_adapter=mock_adapter)
    result = await validator.evaluate("test content", "test criteria")

    assert result["score"] == 8
    assert result["reasoning"] == "Good"
    assert result["is_valid"] is True
    mock_adapter.generate_response.assert_called_once()


@pytest.mark.asyncio
async def test_strict_validator_markdown_json() -> None:
    """Test JSON parsing when the LLM wraps the response in markdown blocks."""
    mock_adapter = MagicMock()
    mock_adapter.generate_response = AsyncMock(return_value='```json\n{"score": 9, "reasoning": "Excellent", "is_valid": true}\n```')

    validator = StrictValidator(llm_adapter=mock_adapter)
    result = await validator.evaluate("test content", "test criteria")

    assert result["score"] == 9
    assert result["is_valid"] is True
    mock_adapter.generate_response.assert_called_once()


@pytest.mark.asyncio
async def test_strict_validator_missing_keys() -> None:
    """Test retry mechanism triggers when required JSON keys are missing."""
    mock_adapter = MagicMock()
    # Missing 'is_valid'
    mock_adapter.generate_response = AsyncMock(return_value='{"score": 8, "reasoning": "Good"}')

    validator = StrictValidator(llm_adapter=mock_adapter)

    with pytest.raises(ValueError, match="Missing required keys in LLM JSON response."):
        await validator.evaluate("test content", "test criteria")

    # Tenacity stops after 3 attempts
    assert mock_adapter.generate_response.call_count == 3


@pytest.mark.asyncio
async def test_strict_validator_invalid_json() -> None:
    """Test retry mechanism triggers when the LLM returns invalid JSON."""
    mock_adapter = MagicMock()
    mock_adapter.generate_response = AsyncMock(return_value='This is not json.')

    validator = StrictValidator(llm_adapter=mock_adapter)

    with pytest.raises(json.JSONDecodeError):
        await validator.evaluate("test content", "test criteria")

    # Tenacity stops after 3 attempts
    assert mock_adapter.generate_response.call_count == 3
