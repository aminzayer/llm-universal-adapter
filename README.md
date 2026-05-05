# LLM Universal Adapter

A robust, model-agnostic integration layer for Large Language Models (LLMs).

This repository leverages the Factory design pattern to abstract provider-specific SDK implementations (such as OpenAI and Google Gemini) behind a strictly typed, unified interface. It natively handles rate-limiting and intermittent network failures using exponential backoff (`tenacity`), and utilizes `pydantic-settings` for secure configuration management.

This adapter is specifically engineered for high-availability systems, multi-model architectures, orchestration agents, and Agentic RAG pipelines that require seamless provider swapping and fallback logic without altering core application code.

## Architecture Overview

- **`BaseLLMAdapter`**: An abstract base class defining the required contract (`generate_response`, `get_token_count`).
- **`LLMAdapterFactory`**: A creational factory pattern implementation to instantiate and manage adapters at runtime.
- **`OpenAIAdapter` / `GeminiAdapter`**: Concrete implementations executing provider-specific API calls.
- **Configuration Management**: Environment variables dynamically injected and validated during instantiation.

## Installation & Environment Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/aminzayer/llm-universal-adapter.ok
   cd llm-universal-adapter
   ```

2. **Install dependencies:**
   Ensure you are using Python 3.9+. Install the required packages:

   ```bash
   pip install openai google-generativeai tiktoken tenacity pydantic-settings
   ```

3. **Configure Environment Variables:**
   Copy the provided `.env.example` file to create your local `.env` configuration.

   ```bash
   cp .env.example .env
   ```

   Update the `.env` file with your respective API keys:

   ```ini
   OPENAI_API_KEY=sk-...
   GEMINI_API_KEY=AIza...
   DEFAULT_TEMPERATURE=0.7
   ```

## Usage

The adapters are automatically registered with the factory upon importing from the `src.adapter` module.

### Initializing and Executing Generation

```python
from src.adapter import LLMAdapterFactory
from src.config import settings

# 1. Instantiate the adapter via the Factory
# Note: API keys are securely loaded from the environment by default.
openai_adapter = LLMAdapterFactory.create_adapter("openai", model="gpt-4o")

# 2. Execute a generation request
prompt = "Explain the mechanics of Sparse Attention in Transformers."
response = openai_adapter.generate_response(prompt, temperature=0.5)

print("Response:", response)

# 3. Utilize token counting utilities
token_count = openai_adapter.get_token_count(prompt)
print(f"Token count: {token_count}")
```

### Switching Providers

Because all adapters inherit from `BaseLLMAdapter`, swapping the underlying LLM requires only a change in the factory key:

```python
gemini_adapter = LLMAdapterFactory.create_adapter("gemini", model="gemini-1.5-pro")
gemini_response = gemini_adapter.generate_response(prompt)
```

## Extensibility: Adding a Custom Provider

The system is designed to be easily extensible. To add a new provider (e.g., Anthropic's Claude), follow these steps:

1. **Implement the Adapter Class:**
   Create a new file (e.g., `src/adapter/claude_adapter.py`). Inherit from `BaseLLMAdapter` and implement the required methods.

   ```python
   import logging
   from typing import Any, Optional
   import anthropic
   from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
   
   from .base import BaseLLMAdapter
   from ..config import settings

   logger = logging.getLogger(__name__)

   class ClaudeAdapter(BaseLLMAdapter):
       def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-opus-20240229") -> None:
           self.api_key = api_key or getattr(settings, 'anthropic_api_key', None)
           if not self.api_key:
               raise ValueError("Anthropic API key is missing.")
           self.client = anthropic.Anthropic(api_key=self.api_key)
           self.model = model

       @retry(
           retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)),
           wait=wait_exponential(multiplier=1, min=2, max=10),
           stop=stop_after_attempt(5),
           reraise=True,
       )
       def generate_response(self, prompt: str, **kwargs: Any) -> str:
           logger.debug(f"Sending request to Claude using model: {self.model}")
           response = self.client.messages.create(
               model=self.model,
               max_tokens=kwargs.get("max_tokens", 1024),
               messages=[{"role": "user", "content": prompt}]
           )
           return response.content.text

       def get_token_count(self, text: str) -> int:
           # Anthropic's tokenizer implementation
           return self.client.count_tokens(text)
   ```

2. **Register the New Adapter:**
   Update the factory registration in `src/adapter/__init__.py`.

   ```python
   from .claude_adapter import ClaudeAdapter
   
   LLMAdapterFactory.register_adapter("claude", ClaudeAdapter)
   ```

3. **Invoke the Custom Adapter:**

   ```python
   claude_adapter = LLMAdapterFactory.create_adapter("claude")
   response = claude_adapter.generate_response("Analyze this log file.")
   ```

## Testing

Run the comprehensive unit test suite utilizing `pytest`:

```bash
pytest tests/ --strict-markers
```

Mocks are configured to prevent live network requests during testing.
