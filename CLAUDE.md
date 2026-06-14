# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup & Environment

- Install dependencies: `pip install -r requirements.txt`
- Install dev tools: `pip install pytest pytest-mock ruff mypy tiktoken`
- Copy `.env.example` to `.env` and set at least one LLM API key (`OPENAI_API_KEY` or `GEMINI_API_KEY`).

### Running the App

- Local (Bare-metal): `uwsgi --http :8000 --module src.main:app --processes 4 --threads 2`
- Containerized: `docker-compose up --build -d`
- Stop Containers: `docker-compose down`

> Note: `docker-compose.yml` and the README both reference `src.main:app` as the WSGI entry, but `src/main.py` does not yet exist in this repo. If you start the API locally, you'll need to create it (or import an existing app object) before the `uwsgi` command will work.

### Testing & Quality

- Run all tests: `pytest tests/`
- Run a single test file: `pytest tests/test_filename.py`
- Run a specific test: `pytest tests/test_filename.py::test_function`
- Lint code: `ruff check .`
- Static type checking: `mypy src/`

`pytest.ini` sets `pythonpath = src` and `asyncio_mode = auto`, so tests import modules as top-level (`from adapter.base import ...`, `from config import settings`) and async test functions are picked up automatically — no `@pytest.mark.asyncio` required.

## Architecture Overview

LLM Universal Adapter is a resilient async backend service that standardizes interactions with multiple LLM providers (OpenAI, Google Gemini) behind a single interface, with retry/failover, structured output enforcement, and observability.

### Request Flow & Key Wrapping Behavior

`LLMAdapterFactory.create_adapter(...)` in `src/adapter/factory.py` does **not** return a raw provider adapter — it wraps it in `ObservabilityMiddleware` (from `src/telemetry/tracer.py`). Every adapter retrieved through the factory already emits structured JSON telemetry (latency, prompt/completion tokens, cache status) on `generate_response`, `agenerate_stream`, and `generate_with_tools`. There is no opt-out. When adding a new provider, expect telemetry logs to fire automatically.

`RouterManager` in `src/orchestration/router.py` is itself a `BaseLLMAdapter` and **also** goes through the factory for its inner adapters, so the inner primary/fallback adapters are themselves wrapped in `ObservabilityMiddleware`. Tool registration on the router propagates to both inner adapters (so failover tool calls keep working). Token counting on the router uses the primary adapter only — no failover. For streaming, failover only kicks in if the primary fails **before** yielding any chunk; mid-stream failures are re-raised because seamless mid-stream failover is impossible.

### Adapter Layer (`src/adapter/`)

- `base.py` — `BaseLLMAdapter` ABC. Defines the interface: `generate_response`, `agenerate_stream`, `generate_with_tools`, `get_token_count`, and a `register_tool` mechanism that stores callables in `self.tools` for MCP/function calling.
- `openai_adapter.py` — uses `openai.AsyncClient`. Token counting is local via `tiktoken` (no API call). Retries on `RateLimitError`, `APIConnectionError`, `InternalServerError` (5 attempts, exponential backoff). Builds OpenAI tool schemas from Python function signatures via `inspect.signature`; supports sync and async tool functions.
- `gemini_adapter.py` — uses the new `google-genai` SDK (`genai.Client`, not the deprecated `google.generativeai`). Token counting hits the API via `client.aio.models.count_tokens`. Retries only on `APIError`. Passes tools as a list of callables directly to the SDK. Stores `model_name` (not `model`) — `ObservabilityMiddleware` checks both attributes.
- `factory.py` — registry pattern; registrations happen at import time in `src/adapter/__init__.py`.

### Cross-Cutting Components

- **Validator (`src/validator/llm_judge.py`)**: `StrictValidator` uses an LLM-as-judge to score content. Decodes markdown-wrapped JSON (strips `` ``` `` / `` ```json ``) and retries on `JSONDecodeError`/`ValueError` via `tenacity` (3 attempts, exponential backoff). Returns a dict with required keys `score`, `reasoning`, `is_valid`.
- **Memory (`src/memory/manager.py`)**: `ConversationManager` uses a sliding-window truncation strategy. System prompt at index 0 is **always preserved**; oldest non-system messages are popped when total tokens exceed `max_context_tokens * threshold` (default 80%). Token counts come from the adapter, so the choice of adapter affects how the window is measured.
- **Structured Output (`src/utils/structured.py`)**: `StructuredGenerator` uses **native** provider features where available — `response_format={"type": "json_schema", ...}` for OpenAI, `response_mime_type="application/json"` + `response_schema` for Gemini. Falls back to schema-in-prompt with self-correction (re-prompts with the validation error appended). Retries 3× on `JSONDecodeError`/`ValidationError`. Provider is inferred from `self.adapter.provider` or by class name sniffing on `self.adapter.adapter`.
- **Telemetry (`src/telemetry/tracer.py`)**: `ObservabilityMiddleware`. Logs one JSON line per invocation. The `cache_status` heuristic depends on `getattr(self.adapter, "semantic_cache", None)` — if a future adapter exposes a `semantic_cache` attribute, latency < 0.05s is reported as `HIT`, otherwise `MISS`; with no attribute present it stays `DISABLED`.
- **Semantic Cache (`src/src/cache/semantic.py`)**: `SemanticCache` class (in-memory list of `{prompt, embedding, response}` entries) plus a `with_semantic_cache` decorator intended to wrap adapter `generate_response` methods. **Not yet applied to any adapter** — the decorator checks `getattr(self, "semantic_cache", None)` at call time, so wiring it up is just assigning a `SemanticCache` instance to the adapter and decorating the method. Note the unusual `src/src/` path: when importing, use `from src.cache.semantic import ...` (or fix the path).
- **Scraper (`src/scraper/async_crawler.py`)**: `AgenticScraper` does BFS via `aiohttp` + `BeautifulSoup` and uses an LLM (via the factory) to evaluate page relevance. Fetch timeout is 10s per page.
- **Tools (`src/tools/es_discovery.py`)**: `ElasticsearchDiscoveryTool` is designed to be registered via `adapter.register_tool(...)`. Enforces source diversity (max per-domain cap) over a 5×-oversampled result set.

### Configuration (`src/config.py`)

Pydantic Settings loads from `.env`. Only `openai_api_key`, `gemini_api_key`, and `default_temperature` (default `0.7`) are exposed. `DATABASE_URL` and `REDIS_URL` are referenced in `docker-compose.yml` but not yet read by `Settings` — they exist for the (planned) `main.py` WSGI app.

### Tech Stack & Infrastructure

- **Language**: Python 3.11+
- **Application Server**: uWSGI (entry: `src.main:app`)
- **LLM SDKs**: `openai>=1.14.0`, `google-genai>=0.3.0`
- **Resilience**: `tenacity` (exponential backoff, retries on provider-specific errors)
- **Async HTTP**: `aiohttp`, `beautifulsoup4`
- **Tokenization**: `tiktoken` (OpenAI only; Gemini counts via API)
- **Database**: PostgreSQL with `pgvector` (RAG), via `ankane/pgvector` image
- **Cache/Broker**: Redis
- **Search**: Elasticsearch
- **Deployment**: Docker Compose (api + db + redis)
- **CI**: GitHub Actions in `.github/workflows/ci.yml` — runs `ruff`, `mypy`, `pytest` on push/PR to `main`
