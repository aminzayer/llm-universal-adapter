# LLM Universal Adapter

[![CI Pipeline](https://github.com/aminzayer/llm-universal-adapter/actions/workflows/ci.yml/badge.svg)](https://github.com/aminzayer/llm-universal-adapter/actions)

A robust, universal, and fully asynchronous Python adapter for interacting with Large Language Models (LLMs) like OpenAI (`gpt-4o`) and Google Gemini (`gemini-2.5-flash`).

## Core Features

* **Fully Asynchronous (asyncio)**: All generation, streaming, and tool execution methods use `async/await` for optimal I/O performance and non-blocking architecture.
* **Router Manager & Failover**: Automatically intercepts API and rate-limit errors from the primary provider and seamlessly re-routes the exact prompt and tools to a fallback model.
* **Semantic Caching**: Leverages embeddings to calculate cosine similarity between prompts. Short-circuits LLM calls if a semantically similar prompt exceeds the confidence threshold (e.g., 0.95) to save API costs and reduce latency.
* **Agentic Scraper**: Includes an asynchronous web crawler that leverages the LLM to evaluate the relevance of scraped technical web content.
* **Strict JSON Validator**: Uses `tenacity` for robust exponential backoff, enforcing strict JSON schema formatting and automated retries.
* **Universal MCP Tools Integration**: Standardized function calling schemas across all supported LLM providers.

## Installation

Make sure you have Python 3.11+ installed. Install the required project dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Configuration

Ensure your environment variables are configured. You can export them directly or use a `.env` file:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
```

## Running Tests

The project includes a comprehensive asynchronous test suite built with `pytest` and `pytest-asyncio`. Tests use mocks to avoid making live network requests.

To run the test suite:

```bash
pytest tests/ -v
```

## Architecture

* `src/adapter/`: Contains the Abstract Base Class and concrete implementations (`OpenAIAdapter`, `GeminiAdapter`).
* `src/orchestration/`: Includes the `RouterManager` for failover logic.
* `src/cache/`: Contains the `SemanticCache` layer and its decorators.
* `src/scraper/`: Houses the `AgenticScraper` for BFS asynchronous crawling.
* `src/validator/`: Contains the `StrictValidator` for JSON output evaluation.
