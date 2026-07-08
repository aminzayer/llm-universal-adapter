# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup & Environment

- Install dependencies: `pip install -r requirements.txt` (dev tools — pytest, pytest-mock, ruff, mypy, tiktoken — are already pinned in `requirements.txt`). Note: `requirements.txt` has duplicate entries for `pytest`, `tiktoken`, and `mypy` with different version floors; pip resolves to the higher floor, but edit with care.
- Copy `.env.example` to `.env` and set at least one LLM API key (`OPENAI_API_KEY` or `GEMINI_API_KEY`). The example file only ships LLM keys; the connection vars below must be added by hand for full-stack local runs. A `.venv/` is already present in the repo root — activate it with `source .venv/bin/activate` instead of creating a new one.

### Running the App

- Local (bare-metal): `uvicorn src.main:app --host 0.0.0.0 --port 8000`
- Containerized: `docker-compose up --build -d`
- Stop containers: `docker-compose down`

> The app is a FastAPI ASGI app — use `uvicorn` (or any ASGI server), not `uwsgi`. The README and `docker-compose.yml` still mention `uwsgi` historically; ignore those references.

### Testing & Quality

- Run all tests: `pytest tests/`
- Run a single test file: `pytest tests/test_filename.py`
- Run a specific test: `pytest tests/test_filename.py::test_function`
- Lint code: `ruff check .`
- Static type checking: `mypy src/`

`pytest.ini` sets `pythonpath = src` and `asyncio_mode = auto`, so tests import modules as top-level (`from adapter.base import ...`, `from config import settings`) and async test functions are picked up automatically — no `@pytest.mark.asyncio` required.

## Architecture Overview

LLM Universal Adapter is a resilient async backend service that standardizes interactions with multiple LLM providers (OpenAI, Google Gemini) behind a single interface, with retry/failover, structured output enforcement, human-in-the-loop approvals, and observability.

### Request Flow & Key Wrapping Behavior

`LLMAdapterFactory.create_adapter(...)` in `src/adapter/factory.py` does **not** return a raw provider adapter — it wraps it in `ObservabilityMiddleware` (from `src/telemetry/tracer.py`). Every adapter retrieved through the factory already emits structured JSON telemetry (latency, prompt/completion tokens, cache status) on `generate_response`, `agenerate_stream`, and `generate_with_tools`. There is no opt-out. When adding a new provider, expect telemetry logs to fire automatically.

Pass `enable_guardrail=True` (and optionally `block_on_pii=True`) to additionally wrap the adapter in `InputGuardrailMiddleware` from `src/security/guardrail.py`. The wrapping order becomes `InputGuardrailMiddleware → ObservabilityMiddleware → raw adapter`, so the guardrail sees the raw prompt **before** telemetry, the cache, or the provider. This is opt-in; default behavior is unchanged, and all other call sites (`RouterManager`, `SwarmOrchestrator`, `AgenticScraper`) inherit the guardrail transparently because they reach providers through the factory.

`RouterManager` in `src/orchestration/router.py` is itself a `BaseLLMAdapter` and **also** goes through the factory for its inner adapters, so the inner primary/fallback adapters are themselves wrapped in `ObservabilityMiddleware`. Tool registration on the router propagates to both inner adapters (so failover tool calls keep working). Token counting on the router uses the primary adapter only — no failover. For streaming, failover only kicks in if the primary fails **before** yielding any chunk; mid-stream failures are re-raised because seamless mid-stream failover is impossible.

The router can also take an optional `local_provider` (e.g. `"local"`) plus `local_kwargs` (typically `base_url` and `model`). When set, an `_is_trivial(prompt)` heuristic — which flattens JSON-wrapped messages arrays the way `main.py` serialises them, then matches against a small list of cheap keywords (`classify`, `categorize`, `label`, `sentiment`, `is this`, `yes/no`, …) and rejects anything with reasoning cues (`reason`, `explain`, `plan`, `analyze`, `code`, `implement`, `design`, …) or longer than 800 characters — decides whether to send the request to the local adapter. Trivial prompts go local first; any local failure falls through to the existing primary → fallback chain with a warning, so a flaky local server never breaks user requests. `generate_with_tools` and `get_token_count` always stay on the cloud chain — local function-calling is inconsistent and local token counting is already free.

### Human-in-the-Loop (HITL) Tool Approval

The system supports suspending execution when a tool registered with `requires_approval=True` is invoked by the model. 
1. **Suspension**: When the model calls an approval-required tool, the adapter (`OpenAIAdapter` or `GeminiAdapter`) captures the execution state, saves it to Redis (using `HITLState` and `HITLStateManager` from `src/orchestration/hitl.py`), and raises `ApprovalRequiredError`.
2. **Orchestration**: `SwarmOrchestrator` catches `ApprovalRequiredError` during worker execution, enriches the Redis state with swarm task details, and re-raises it.
3. **Resumption**: A client calls `POST /v1/approval` with `approve` or `abort`. If approved, the state is retrieved, `resume_with_tools(state)` is invoked on the adapter, the tool is executed, any subsequent tool calls are completed, and a follow-up completion call returns the final answer. Telemetry tracks both suspensions and resumptions.

### Adapter Layer (`src/adapter/`)

- `base.py` — `BaseLLMAdapter` ABC. Defines the interface: `generate_response`, `agenerate_stream`, `generate_with_tools`, `get_token_count`, `register_tool` (which supports a `requires_approval` boolean), `set_redis_client`, and `resume_with_tools`.
- `openai_adapter.py` — uses `openai.AsyncClient`. Token counting is local via `tiktoken`. Retries on `RateLimitError`, `APIConnectionError`, `InternalServerError`. Suspends tool calls if marked for approval, raising `ApprovalRequiredError`. Implements `resume_with_tools` by running the approved tool, executing remaining tools in the batch, and returning the final completion.
- `gemini_adapter.py` — uses the new `google-genai` SDK (`genai.Client`). Token counting hits the API. Retries only on `APIError`. Implements tool suspension and `resume_with_tools` using Gemini's native tool response structures.
- `anthropic_adapter.py` — uses the official `anthropic` async SDK (`AsyncAnthropic`). Token counting uses the native `messages.count_tokens` endpoint. Retries on `RateLimitError`, `APIConnectionError`, `InternalServerError`. Implements tool suspension and `resume_with_tools` mapping registered tools to Claude's JSON Schema format (`input_schema`) and dynamically converting string content fields to block lists to prevent consecutive role exceptions.
- `local_adapter.py` — `LocalModelAdapter` speaks OpenAI-compatible HTTP (`POST /v1/chat/completions`) to vLLM, Ollama, or LM Studio. Optimized for hardware-accelerated local environments. Streaming is SSE-line parsed. Token counting is a local character heuristic. `generate_with_tools` raises `NotImplementedError`.
- `factory.py` — registry pattern; registrations happen at import time in `src/adapter/__init__.py`.

### Cross-Cutting Components

- **Validator (`src/validator/llm_judge.py`)**: `StrictValidator` uses an LLM-as-judge to score content. Decodes markdown-wrapped JSON (strips `` ``` `` / `` ```json ``) and retries on `JSONDecodeError`/`ValueError` via `tenacity` (3 attempts, exponential backoff). Returns a dict with required keys `score`, `reasoning`, `is_valid`.
- **Memory (`src/memory/manager.py`)**: `ConversationManager` uses a sliding-window truncation strategy. System prompt at index 0 is **always preserved**; oldest non-system messages are popped when total tokens exceed `max_context_tokens * threshold` (default 80%). Token counts come from the adapter, so the choice of adapter affects how the window is measured.
- **Structured Output (`src/utils/structured.py`)**: `StructuredGenerator` uses **native** provider features where available — `response_format={"type": "json_schema", ...}` for OpenAI, `response_mime_type="application/json"` + `response_schema` for Gemini. Falls back to schema-in-prompt with self-correction (re-prompts with the validation error appended). Retries 3× on `JSONDecodeError`/`ValidationError`. Provider is inferred from `self.adapter.provider` or by class name sniffing on `self.adapter.adapter`.
- **Telemetry (`src/telemetry/tracer.py`)**: `ObservabilityMiddleware`. Logs one JSON line per invocation. Telemetry is also captured and logged during the `resume_with_tools` step. The `cache_status` is dynamically retrieved from `self.adapter._last_cache_tier` set by the `@with_semantic_cache` decorator. It defaults to `"DISABLED"` if the cache is not configured.
- **Semantic Cache (`src/cache/semantic.py`)**: `SemanticCache` class (two-tier strategy: Layer 1 Redis for exact matches, Layer 2 PostgreSQL `pgvector` for semantic cosine similarity) plus a `with_semantic_cache` decorator. This is applied to `generate_response` in `OpenAIAdapter`, `GeminiAdapter`, and `AnthropicAdapter`. Each adapter exposes a `semantic_cache` property that lazily instantiates it using connection pools (`redis_pool` and `db_pool`) from `app_state` in `src/main.py`. For Anthropic, embedding generation is delegated to OpenAI or Gemini based on configured settings.
- **Scraper (`src/scraper/async_crawler.py`)**: `AgenticScraper` does BFS via `aiohttp` + `BeautifulSoup` and uses an LLM (via the factory) to evaluate page relevance. Fetch timeout is 10s per page.
- **Tools (`src/tools/es_discovery.py`)**: `ElasticsearchDiscoveryTool` is designed to be registered via `adapter.register_tool(...)`. Enforces source diversity (max per-domain cap) over a 5×-oversampled result set.
- **Swarm (`src/orchestration/swarm.py`)**: `SwarmOrchestrator` does lightweight intent routing. `ClassifierAgent` returns a `ClassificationDecision` (pydantic) via `StructuredGenerator`; the orchestrator looks up the named worker in its registry and dispatches a `SwarmTask` to it. Enriches `HITLState` with task metadata and worker names if `ApprovalRequiredError` is caught during worker dispatch. Built-in workers: `SearchAgent` (wraps `ElasticsearchDiscoveryTool`), `SummaryAgent` (LLM via `StructuredGenerator`). All inter-agent messages are pydantic `BaseModel` subclasses — never raw dicts. `UnknownIntentError` is raised when a classifier intent has no registered worker. Custom workers implement `BaseWorker.run(task, decision) -> BaseModel`.
- **Security (`src/security/guardrail.py`)**: `InputGuardrailMiddleware` screens every prompt before it reaches the cache or the adapter. `PIIMasker` does regex-based detection and masking for emails, phones, credit-card-like digit runs, SSN-like numbers, IPv4, OpenAI keys, GitHub PATs, AWS access keys, bearer tokens, and `api_key=` / `token=` assignments (in-place replacement with `[REDACTED_*]` placeholders, by default non-blocking). `PromptInjectionDetector` flags instruction-override, role-hijack, system-prompt extraction, jailbreak prefixes, and delimiter-injection shapes. Detection raises `SecurityViolationError(matched_pattern=...)` and halts execution. Opt-in via `LLMAdapterFactory.create_adapter(..., enable_guardrail=True)`.

### API Surface (`src/main.py`)

FastAPI app exposing an OpenAI-compatible contract. `AppState` (a plain module-level singleton) holds the asyncpg pool, redis client, and the `RouterManager`. The `lifespan` async context manager wires all three up at startup if their env vars are set, and tears them down on shutdown.

- `GET /v1/health` — basic liveness, returns `{"status": "ok"}`.
- `POST /v1/chat/completions` — OpenAI-shaped body (`model`, `messages`, `temperature`, `stream`, `max_tokens`). If tools are registered, delegates to `generate_with_tools` automatically. Yields `HTTP 202` (via `ApprovalRequiredError` exception handler) if any tools require human approval.
- `POST /v1/approval` — Resumes or aborts a suspended execution state. Accepts a JSON body containing `state_id` and `action` (`"approve"` or `"abort"`). If approved, executes the tool and resumes the agent execution to return the final chat completion response.


### Configuration (`src/config.py`)

Pydantic Settings loads from `.env`. Only `openai_api_key`, `gemini_api_key`, `anthropic_api_key`, and `default_temperature` (default `0.7`) are exposed.

`DATABASE_URL`, `REDIS_URL`, `PRIMARY_PROVIDER` (default `openai`), `FALLBACK_PROVIDER` (default `gemini`), `LOCAL_PROVIDER_BASE_URL`, `LOCAL_MODEL` (default `llama3.1`), and `LOCAL_PROVIDER_API_KEY` are **not** in `Settings` — they are read directly by `src/main.py` via `os.getenv` inside the lifespan handler. `LOCAL_PROVIDER_BASE_URL` is opt-in: if unset, the router runs in the original two-adapter shape with no local dispatch. The `.env.example` file only ships LLM keys; the connection vars and provider-selection vars must be added by hand for full-stack local runs. If `DATABASE_URL`/`REDIS_URL` are missing, `db_pool` and `redis_pool` stay `None` and the app still starts (the chat endpoint will work, but the cache/db layers won't). `mypy` strictness is relaxed for `tiktoken`, `google.*`, and `tenacity.*` in `pyproject.toml`.

### Tech Stack & Infrastructure

- **Language**: Python 3.11+
- **Web Framework**: FastAPI (entry: `src.main:app`), served via `uvicorn` (not `uwsgi`)
- **LLM SDKs**: `openai>=1.14.0`, `google-genai>=0.3.0`, `anthropic>=0.116.0`
- **Resilience**: `tenacity` (exponential backoff, retries on provider-specific errors)
- **Async HTTP**: `aiohttp`, `beautifulsoup4`
- **Tokenization**: `tiktoken` (OpenAI only; Gemini counts via API)
- **DB Driver**: `asyncpg`
- **Database**: PostgreSQL with `pgvector` (RAG), via `ankane/pgvector` image
- **Cache/Broker**: Redis (`redis>=8.0.0`)
- **Search**: Elasticsearch
- **Deployment**: Docker Compose (api + db + redis)
- **CI**: GitHub Actions in `.github/workflows/ci.yml` — triggers on push/PR to `main`, runs `ruff check .`, `mypy src/`, and `pytest tests/ --maxfail=1 --disable-warnings` on Python 3.11. Mirrors the local commands above, so a clean local run is a clean CI run.

## graphify (mandatory)

This project has a knowledge graph at `graphify-out/` (god nodes, community structure, cross-file relationships). The same rules are also enforced by an `always_on` agent rule at `.agents/rules/graphify.md`.

- For codebase questions, first run `graphify query "<question>"` when `graphify-out/graph.json` exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than `GRAPH_REPORT.md` or raw grep output.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation instead of raw source browsing.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Deeper references

- `antigravity.md` — long-form architecture reference (full `.env` variable list, system diagrams, design rationale). Reach for it when `CLAUDE.md` and `graphify` don't surface enough. `README.md` mirrors most of it for end users.
