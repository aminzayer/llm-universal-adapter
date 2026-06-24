# LLM Universal Adapter — Project Context (`antigravity.md`)

> **Generated:** 2025-06
> **Graphify Graph:** `graphify-out/` (771 nodes · 1 546 edges · 72 communities)
> **CLAUDE.md:** authoritative quick-reference for commands and gotchas

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Directory Structure](#2-directory-structure)
3. [Architecture Overview](#3-architecture-overview)
4. [Module Reference](#4-module-reference)
5. [Request Pipeline](#5-request-pipeline)
6. [Middleware Wrapping Order](#6-middleware-wrapping-order)
7. [Routing Strategy (3-Tier)](#7-routing-strategy-3-tier)
8. [Data Flow Diagrams](#8-data-flow-diagrams)
9. [Dependency Map](#9-dependency-map)
10. [Infrastructure & Deployment](#10-infrastructure--deployment)
11. [Configuration Reference](#11-configuration-reference)
12. [Development Commands](#12-development-commands)
13. [Testing Strategy](#13-testing-strategy)
14. [Known Patterns & Gotchas](#14-known-patterns--gotchas)

---

## 1. Project Summary

**LLM Universal Adapter** is a resilient, async FastAPI backend service that:

- Exposes a single **OpenAI-compatible** REST API (`POST /v1/chat/completions`)
- Normalises interactions with **multiple LLM providers** (OpenAI GPT-4o, Google Gemini 2.5 Flash, local vLLM/Ollama) behind a single `BaseLLMAdapter` interface
- Implements **automatic failover**: primary → fallback cloud, with an optional local-first path for trivial prompts
- Guards every inbound prompt with a **security middleware** (PII masking + prompt-injection detection)
- Emits **structured JSON telemetry** for every LLM call (latency, tokens, cache tier, errors)
- Stores prompt templates in a **versioned, A/B-sampled PostgreSQL registry** with LRU caching
- Provides a **two-tier semantic cache** (Redis exact-match → PostgreSQL `pgvector` cosine similarity)
- Orchestrates multi-agent workloads through a **SwarmOrchestrator** that classifies intent and routes to typed worker agents

**Language:** Python 3.11+
**Framework:** FastAPI (served via `uvicorn`, not `uwsgi`)
**License:** MIT

---

## 2. Directory Structure

```
llm-universal-adapter/
├── src/
│   ├── main.py                     # FastAPI app, lifespan, API routes
│   ├── config.py                   # Pydantic Settings (.env loader)
│   ├── adapter/
│   │   ├── __init__.py             # Registers adapters with factory
│   │   ├── base.py                 # BaseLLMAdapter ABC
│   │   ├── factory.py              # LLMAdapterFactory (registry + middleware stacking)
│   │   ├── openai_adapter.py       # OpenAIAdapter (gpt-4o default, tiktoken)
│   │   ├── gemini_adapter.py       # GeminiAdapter (gemini-2.5-flash, google-genai SDK)
│   │   └── local_adapter.py        # LocalModelAdapter (OpenAI-compat HTTP, vLLM/Ollama)
│   ├── orchestration/
│   │   ├── router.py               # RouterManager (3-tier failover, trivial heuristic)
│   │   └── swarm.py                # SwarmOrchestrator, ClassifierAgent, workers
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── registry.py             # PromptRegistry (versioned, A/B, LRU cache)
│   ├── security/
│   │   ├── __init__.py
│   │   └── guardrail.py            # InputGuardrailMiddleware, PIIMasker, InjectionDetector
│   ├── telemetry/
│   │   ├── __init__.py
│   │   └── tracer.py               # ObservabilityMiddleware (JSON telemetry)
│   ├── cache/
│   │   ├── __init__.py
│   │   └── semantic.py             # SemanticCache (Redis + pgvector), decorator
│   ├── memory/
│   │   ├── __init__.py
│   │   └── manager.py              # ConversationManager (sliding-window context)
│   ├── utils/
│   │   ├── __init__.py
│   │   └── structured.py           # StructuredGenerator (Pydantic schema enforcement)
│   ├── validator/
│   │   ├── __init__.py
│   │   └── llm_judge.py            # StrictValidator (LLM-as-judge, JSON scoring)
│   ├── scraper/
│   │   ├── __init__.py
│   │   └── async_crawler.py        # AgenticScraper (BFS + LLM relevance filter)
│   └── tools/
│       ├── __init__.py
│       └── es_discovery.py         # ElasticsearchDiscoveryTool (diverse search)
├── tests/                          # pytest test suite
├── graphify-out/                   # Knowledge graph (graph.json, GRAPH_REPORT.md, wiki/)
├── .github/workflows/ci.yml        # CI: ruff + mypy + pytest on Python 3.11
├── docker-compose.yml              # api + db (pgvector) + redis
├── Dockerfile
├── requirements.txt
├── pyproject.toml                  # ruff + mypy config
├── pytest.ini                      # pythonpath=src, asyncio_mode=auto
├── .env.example
└── CLAUDE.md                       # Developer guidance for Claude Code
```

---

## 3. Architecture Overview

The system is built around a **layered decorator chain** over a common `BaseLLMAdapter` interface.
Every component that wraps another is itself a `BaseLLMAdapter`, making the stack composable without
protocol changes.

```
Client Request
     |
     v
+-------------------------------------------------------------------+
|  FastAPI  POST /v1/chat/completions                               |
|  src/main.py -- AppState (router_manager, db_pool, redis_pool)   |
+-----------------------------+-------------------------------------+
                              |
                              v
                    +------------------+
                    |  RouterManager   |  <- itself a BaseLLMAdapter
                    | (3-tier routing) |
                    +--------+---------+
         +-------------------+--------------------+
         v                   v                    v
   local_adapter       primary_adapter      fallback_adapter
  (optional, vLLM)   (OpenAI GPT-4o)   (Gemini 2.5 Flash)
         |                   |                    |
         +-------------------+--------------------+
                             | (each inner adapter already wrapped)
                             v
            +-------------------------------+
            |  InputGuardrailMiddleware     |  <- opt-in (enable_guardrail=True)
            |  (PII mask + injection check) |
            +--------------+----------------+
                           |
                           v
            +-------------------------------+
            |  ObservabilityMiddleware      |  <- always applied by factory
            |  (JSON telemetry per call)    |
            +--------------+----------------+
                           |
                           v
            +-------------------------------+
            |  Raw Provider Adapter         |
            |  (OpenAI / Gemini / Local)    |
            +-------------------------------+
```

---

## 4. Module Reference

### 4.1 Adapter Layer (`src/adapter/`)

#### `base.py` — `BaseLLMAdapter`

Abstract base class defining the **universal interface** all providers must implement.

| Method | Signature | Purpose |
|--------|-----------|---------|
| `generate_response` | `async (prompt, **kwargs) -> str` | Single-turn generation |
| `agenerate_stream` | `async (prompt, **kwargs) -> AsyncGenerator[str, None]` | Streaming generation |
| `generate_with_tools` | `async (prompt) -> str` | MCP/function-calling generation |
| `get_token_count` | `async (text) -> int` | Token counting |
| `register_tool` | `(name, func, description)` | Register callable as an LLM tool |

Tools are stored as `Dict[str, Dict[str, Any]]` on `self.tools`. All middleware layers keep `self.tools` in sync with inner adapters.

---

#### `factory.py` — `LLMAdapterFactory`

Registry-pattern factory. Adapters register at **import time** in `src/adapter/__init__.py`:

```python
LLMAdapterFactory.register_adapter("openai", OpenAIAdapter)
LLMAdapterFactory.register_adapter("gemini", GeminiAdapter)
LLMAdapterFactory.register_adapter("local", LocalModelAdapter)
```

`create_adapter(provider_name, *, enable_guardrail=False, block_on_pii=False, **kwargs)` always
wraps the raw adapter in `ObservabilityMiddleware`. When `enable_guardrail=True`, an outer
`InputGuardrailMiddleware` layer is added on top.

**Wrapping order (outside -> inside):**
```
InputGuardrailMiddleware   [opt-in]
  |-- ObservabilityMiddleware  [always]
       |-- RawProviderAdapter
```

---

#### `openai_adapter.py` — `OpenAIAdapter`

- Uses `openai.AsyncClient`; defaults to `gpt-4o`
- Token counting via `tiktoken` (local, no API call); falls back to `cl100k_base` for unknown models
- Retries on `RateLimitError`, `APIConnectionError`, `InternalServerError` — 5 attempts, exponential backoff via `tenacity`
- Builds tool schemas from Python function signatures using `inspect.signature`; handles both sync and async tool callables
- Two-turn function-calling protocol: first call gets tool invocations, results appended as `role: tool`, second call returns final response

---

#### `gemini_adapter.py` — `GeminiAdapter`

- Uses the **new** `google-genai` SDK (`genai.Client`) — NOT deprecated `google.generativeai`
- Defaults to `gemini-2.5-flash`; stores model name as `self.model_name` (not `self.model`) — `ObservabilityMiddleware` checks both attributes
- Token counting hits the API via `client.aio.models.count_tokens`
- Retries only on `APIError`; temperature mapped to `types.GenerateContentConfig`
- Passes tools as a list of raw Python callables to the SDK (SDK auto-generates schema)

---

#### `local_adapter.py` — `LocalModelAdapter`

- Targets any **OpenAI-compatible HTTP** server: vLLM, Ollama, LM Studio
- Designed for Apple Silicon (Metal) and AWS Graviton environments
- Uses `aiohttp` with a **lazy, shared `ClientSession`** and connection pooling
- Retries 5x on 5xx and `ClientConnectionError` (tenacity); 4xx surfaces immediately as caller bugs
- Streaming via SSE-line parsing (`data: {...}` per chunk, terminated by `data: [DONE]`)
- Token counting: local character heuristic `len(text) // 4` — avoids remote API call
- `generate_with_tools` raises `NotImplementedError` — local function-calling is inconsistent across models

---

### 4.2 Orchestration (`src/orchestration/`)

#### `router.py` — `RouterManager`

A `BaseLLMAdapter` wrapping 2-3 inner adapters (primary, fallback, optional local).

**Trivial prompt heuristic (`_is_trivial`):**
- JSON-wrapped message arrays are flattened to plain text before matching
- Trivial if: prompt < 800 chars AND has a cheap keyword (`classify`, `categorize`, `label`, `sentiment`, `is this`, `yes/no`, `true or false`, `spam or not`) AND no complex reasoning keyword
- Complex keywords: `reason`, `explain`, `plan`, `analyze`, `code`, `implement`, `refactor`, `design`, `derive`, `prove`, `summarize`
- Very short prompts (<120 chars, no reasoning cues) are also trivial

**Routing table:**

| Request type | Route |
|---|---|
| Trivial (heuristic) | Local -> Primary -> Fallback |
| Complex | Primary -> Fallback |
| Tool-calling | Primary -> Fallback (never local) |
| Token counting | Primary only (no failover) |

**Streaming failover:** Only activates if primary fails before yielding any chunk. Mid-stream failures are re-raised since seamless mid-stream failover is impossible.

Tool registration on the router propagates to **all three** inner adapters.

---

#### `swarm.py` — `SwarmOrchestrator`

Multi-agent intent classification and dispatch system. All inter-agent messages are **Pydantic `BaseModel` payloads** enforced by `StructuredGenerator`.

**Key classes:**

| Class | Role |
|---|---|
| `ClassifierAgent` | Calls LLM via `StructuredGenerator`, returns `ClassificationDecision` |
| `BaseWorker` | Abstract base for domain workers |
| `SearchAgent` | Queries Elasticsearch via `ElasticsearchDiscoveryTool`, returns `SearchAgentOutput` |
| `SummaryAgent` | Calls LLM to summarize text, returns `SummaryAgentOutput` |
| `SwarmOrchestrator` | Coordinates classification + worker dispatch |
| `UnknownIntentError` | Raised when classifier intent has no registered worker |

**Pydantic schemas (inter-agent contracts):**

```
ClassificationDecision  ->  intent (str), confidence (float), reasoning (str)
SwarmTask               ->  request_id, user_input, context (Dict)
SwarmResult             ->  request_id, intent, worker, output, classification
SearchAgentOutput       ->  query, results (List[Dict]), raw_response
SummaryAgentOutput      ->  summary, key_points (List[str])
```

**Dispatch flow:**
1. `SwarmOrchestrator.dispatch(task)` calls `ClassifierAgent.classify(task)`
2. If confidence < threshold: use `default_intent` or first registered worker
3. Look up worker by intent name -> call `worker.run(task, decision)`
4. Return typed `SwarmResult`

Custom workers implement `BaseWorker.run(task, decision) -> BaseModel` and register via `orchestrator.register_worker(worker)`.

---

### 4.3 Prompt Registry (`src/prompts/`)

#### `registry.py` — `PromptRegistry`

PostgreSQL-backed, versioned prompt management with weighted A/B sampling.

**Key features:**
- **Append-only versioning:** `create_version()` inserts a new row with `MAX(version_number) + 1`
- **Weighted A/B:** `get_prompt(category)` samples among active versions using `random.choices(weights=...)`
- **LRU cache:** `cachetools.LRUCache` (default 256 entries, 10s TTL); invalidated on write, expired on stale read
- **Graceful degradation:** Falls back to `defaults` mapping (merged over `DEFAULT_PROMPTS`) when no DB

**Database schema:**
```sql
CREATE TABLE prompt_versions (
    id              BIGSERIAL PRIMARY KEY,
    category        TEXT        NOT NULL,
    version_number  INTEGER     NOT NULL,
    text            TEXT        NOT NULL,
    weight          INTEGER     NOT NULL DEFAULT 0 CHECK (weight >= 0),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (category, version_number)
);
CREATE INDEX idx_prompt_versions_category ON prompt_versions (category);
```

**Built-in default templates:**
- `classifier.system` — routing prompt for `ClassifierAgent`
- `summary.system` — summarization prompt for `SummaryAgent`

**Key methods:**

| Method | Description |
|---|---|
| `initialize()` | Idempotent schema migration |
| `create_version(category, text, weight=100)` | Append new version, invalidate cache |
| `list_versions(category)` | All versions sorted by version_number ASC |
| `get_prompt(category)` | Weighted-sample one prompt string |
| `get_category(category)` | Return `PromptCategory` (all versions) |
| `cache_stats()` | Return `{hits, misses, size}` snapshot |
| `invalidate_cache()` | Drop all cache entries |

---

### 4.4 Security (`src/security/`)

#### `guardrail.py` — `InputGuardrailMiddleware`

A `BaseLLMAdapter` middleware that screens every prompt **before** it reaches the cache or provider.

**Two-step screening (in order):**
1. **Prompt injection detection** (on raw text) — raises `SecurityViolationError` on detection
2. **PII masking** (on clean text) — masks in-place and forwards, or blocks if `block_on_pii=True`

**PII pattern coverage** (regex, in priority order):

| Type | Placeholder |
|---|---|
| Email | `[REDACTED_EMAIL]` |
| Credit card (13-19 digits) | `[REDACTED_CC]` |
| SSN (DDD-DD-DDDD) | `[REDACTED_SSN]` |
| IPv4 address | `[REDACTED_IPV4]` |
| International phone (E.164) | `[REDACTED_PHONE]` |
| OpenAI key (sk-...) | `[REDACTED_OPENAI_KEY]` |
| GitHub token (ghp_...) | `[REDACTED_GITHUB_TOKEN]` |
| AWS access key (AKIA...) | `[REDACTED_AWS_KEY]` |
| Bearer/Authorization token | `[REDACTED_BEARER_TOKEN]` |
| Generic api_key=, token= | `[REDACTED_API_KEY]` |

**Injection detection categories:**
- `instruction_override` — "ignore/disregard previous instructions"
- `role_hijack` — "you are now a...", "act as DAN"
- `system_prompt_extraction` — "reveal your system prompt"
- `delimiter_injection` — `<|im_start|>`, `<system>`, `system: ...` at line start
- `jailbreak_prefix` — "DAN mode", "jailbreak prompt", "bypass guardrails"

On detection: raises `SecurityViolationError(matched_pattern=...)`, optionally calls `on_violation` callback.
`GuardrailReport` stored on `self.last_report` for telemetry/debug.

---

### 4.5 Telemetry (`src/telemetry/`)

#### `tracer.py` — `ObservabilityMiddleware`

Always wraps raw adapters (applied automatically by `LLMAdapterFactory`). Emits one **structured JSON log line** per invocation.

**Log entry fields:**
```json
{
  "event": "llm_invocation",
  "provider": "openai",
  "model": "gpt-4o",
  "operation": "generate_response",
  "latency_sec": 0.842,
  "prompt_tokens": 120,
  "completion_tokens": 95,
  "total_tokens": 215,
  "cache_status": "MISS",
  "error": null
}
```

**Cache status heuristic:** If the inner adapter exposes `semantic_cache` and latency < 50ms with no error -> `"HIT"`; otherwise `"MISS"`; no attribute -> `"DISABLED"`.

Compatible with ELK and Prometheus (line-by-line JSON ingestion).

---

### 4.6 Semantic Cache (`src/cache/`)

#### `semantic.py` — `SemanticCache`

Two-tier caching architecture:

| Tier | Store | Match Type | Speed |
|---|---|---|---|
| Layer 1 | Redis | Exact string key | Sub-millisecond |
| Layer 2 | PostgreSQL + `pgvector` | Cosine similarity (threshold configurable) | ~10-50ms |

- `get(prompt)` returns `(response | None, tier)` where tier is `"REDIS"`, `"PGVECTOR"`, or `"MISS"`
- `set(prompt, response)` writes to both tiers; Layer 1 backfill happens on Layer 2 hits
- `with_semantic_cache` decorator checks `getattr(self, "semantic_cache", None)` — **not yet wired to any adapter**

**pgvector schema (inline-migrated on first `get` call):**
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS semantic_cache (
    id        SERIAL PRIMARY KEY,
    prompt    TEXT,
    embedding vector,
    response  TEXT
);
```

To enable: assign a `SemanticCache` instance to an adapter and decorate its `generate_response` method with `@with_semantic_cache`.

---

### 4.7 Memory (`src/memory/`)

#### `manager.py` — `ConversationManager`

Multi-turn conversation state with **sliding-window token truncation**.

- System prompt (index 0) is **always preserved**
- Oldest non-system messages are popped when total tokens exceed `max_context_tokens * threshold`
- Defaults: 4096 tokens × 0.8 = 3276 token limit
- Token counting delegates to the injected `BaseLLMAdapter` — model choice affects window measurement

**Methods:** `add_message(role, content)`, `get_messages()`, `clear()`

---

### 4.8 Structured Output (`src/utils/`)

#### `structured.py` — `StructuredGenerator`

Enforces Pydantic schema compliance from LLM responses. Uses **native provider features** where available:

| Provider | Native feature |
|---|---|
| OpenAI | `response_format={"type": "json_schema", ..., "strict": True}` |
| Gemini | `response_mime_type="application/json"` + `response_schema` |
| Other | Schema injected as text into prompt |

**Self-correction loop:** On `JSONDecodeError` or `ValidationError`, the exact error is appended to the prompt and the request retried — up to 3 attempts with exponential backoff via `tenacity`.

Provider inferred from `adapter.provider` or class-name sniffing on `adapter.adapter` (works through middleware layers).

---

### 4.9 Validator (`src/validator/`)

#### `llm_judge.py` — `StrictValidator`

Uses an LLM to score text content against arbitrary criteria. Strips markdown code fences before parsing. Retries 3x on `JSONDecodeError`/`ValueError`.

**Required JSON response keys:**
```json
{
  "score": 8.5,
  "reasoning": "...",
  "is_valid": true
}
```

---

### 4.10 Scraper (`src/scraper/`)

#### `async_crawler.py` — `AgenticScraper`

BFS web crawler with LLM relevance filtering:
1. Fetches HTML via `aiohttp` (10s per-page timeout)
2. Extracts text with `BeautifulSoup`
3. Sends first 1000 chars to LLM, checks for `"YES"` in response
4. BFS continues to `max_depth` (default 2)

Uses `LLMAdapterFactory.create_adapter(provider)` — inherits full middleware stack.

---

### 4.11 Tools (`src/tools/`)

#### `es_discovery.py` — `ElasticsearchDiscoveryTool`

Designed to be registered with `adapter.register_tool(name, func, description)`.

- Searches across `title`, `content`, `description` fields via `multi_match`
- Enforces **source diversity**: oversamples 5x top_k, then caps at `max_count_per_domain` (default 2) per domain
- Returns JSON string: `{"results": [...]}`

---

### 4.12 Configuration (`src/config.py`)

Pydantic Settings loaded from `.env` file:

| Variable | Type | Default |
|---|---|---|
| `OPENAI_API_KEY` | `Optional[str]` | `None` |
| `GEMINI_API_KEY` | `Optional[str]` | `None` |
| `DEFAULT_TEMPERATURE` | `float` | `0.7` |

> `DATABASE_URL`, `REDIS_URL`, `PRIMARY_PROVIDER`, `FALLBACK_PROVIDER`,
> `LOCAL_PROVIDER_BASE_URL`, `LOCAL_MODEL`, `LOCAL_PROVIDER_API_KEY`,
> `PROMPT_CACHE_SIZE`, and `PROMPT_CACHE_TTL` are read by `src/main.py` via
> `os.getenv` — they are **not** in `Settings`.

---

### 4.13 API Entry Point (`src/main.py`)

FastAPI application with OpenAI-compatible REST API.

**`AppState`** (module-level singleton):
```python
class AppState:
    db_pool: Optional[asyncpg.Pool] = None
    redis_pool: Optional[redis.Redis] = None
    router_manager: Optional[RouterManager] = None
    prompt_registry: Optional[PromptRegistry] = None
```

**Lifecycle (`lifespan`):** Creates all pools and managers at startup; tears down on shutdown. Missing `DATABASE_URL`/`REDIS_URL` are tolerated — app starts without DB/cache layers.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/health` | Liveness check -> `{"status": "ok"}` |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions |

**Request body (`ChatCompletionRequest`):**
```json
{
  "model": "gpt-4o",
  "messages": [{"role": "user", "content": "Hello"}],
  "temperature": 0.7,
  "stream": false,
  "max_tokens": null
}
```

**Response:** Standard OpenAI JSON or `text/event-stream` SSE (when `stream: true`). The `usage` block is always zeroed — token accounting is not surfaced at the HTTP layer.

---

## 5. Request Pipeline

```
POST /v1/chat/completions
         |
         v
   Deserialize (ChatCompletionRequest)
         |
         v
   Serialize messages -> JSON string (prompt_str = json.dumps(messages))
         |
         v
   RouterManager.generate_response(prompt_str)
   |-- _is_trivial(prompt)?
   |   |-- YES: local_adapter.generate_response()   <- if configured
   |   |         |-- SUCCESS -> return
   |   |         |-- FAIL -> log warning -> fall through
   |   |-- NO:  skip local
   |
   |-- primary_adapter.generate_response()
   |     |-- InputGuardrailMiddleware._screen()
   |     |   |-- PromptInjectionDetector.evaluate()   -> raise SecurityViolationError on hit
   |     |   |-- PIIMasker.scan()                     -> mask PII in-place
   |     |-- ObservabilityMiddleware (wraps timing, logs JSON)
   |     |     |-- OpenAIAdapter.generate_response()  -> tenacity retry on transient errors
   |     |-- SUCCESS -> return
   |     |-- FAIL -> log warning
   |
   |-- fallback_adapter.generate_response()
         |-- (same middleware stack, GeminiAdapter underneath)
         |-- SUCCESS -> return
         |-- FAIL -> raise exception
         |
         v
   Format OpenAI-compatible JSON response
         |
         v
   Return HTTP 200
```

---

## 6. Middleware Wrapping Order

```python
# With guardrail (opt-in):
LLMAdapterFactory.create_adapter("openai", enable_guardrail=True)
# Returns:
# InputGuardrailMiddleware       <- prompt screened HERE (raw, before telemetry)
#   |-- ObservabilityMiddleware  <- telemetry logged HERE (sanitized prompt)
#        |-- OpenAIAdapter       <- API call made HERE

# Without guardrail (default):
LLMAdapterFactory.create_adapter("openai")
# Returns:
# ObservabilityMiddleware
#   |-- OpenAIAdapter
```

`RouterManager` inner adapters go through the **factory**, so each is independently wrapped.
The `RouterManager` itself is NOT additionally wrapped — it is a raw `BaseLLMAdapter`.

---

## 7. Routing Strategy (3-Tier)

```
Incoming prompt
       |
       |-- local_adapter configured?
       |   |-- NO -> skip to primary
       |   |-- YES: _is_trivial(prompt)?
       |             |-- NO -> skip to primary
       |             |-- YES -> local_adapter.generate_response()
       |                         |-- SUCCESS -> return
       |                         |-- FAIL -> fall through
       |
       primary_adapter.generate_response()
         |-- SUCCESS -> return
         |-- FAIL -> log warning
       |
       fallback_adapter.generate_response()
         |-- SUCCESS -> return
         |-- FAIL -> raise exception
```

---

## 8. Data Flow Diagrams

### Swarm Orchestration

```
User Prompt
     |
     v
SwarmTask(request_id, user_input, context)
     |
     v
ClassifierAgent.classify(task)
  |-- PromptRegistry.get_prompt("classifier.system") -> template
  |-- StructuredGenerator -> LLM -> ClassificationDecision(intent, confidence, reasoning)
     |
     |-- confidence < threshold? -> use default_intent or first registered worker
     |
     v
SwarmOrchestrator.get_worker(intent)
     |
     v
worker.run(task, decision)
  |-- SearchAgent  -> ElasticsearchDiscoveryTool.search() -> SearchAgentOutput
  |-- SummaryAgent -> PromptRegistry.get_prompt("summary.system")
  |                -> StructuredGenerator -> LLM -> SummaryAgentOutput
     |
     v
SwarmResult(request_id, intent, worker, output, classification)
```

### Semantic Cache Lookup

```
generate_response(prompt)
     |
     v
Redis.get("cache:exact:{prompt}")
     |-- HIT -> return cached_response ("REDIS")
     |-- MISS
          |
          v
     embedding_func(prompt) -> vector
          |
          v
     pgvector: SELECT response WHERE cosine_sim >= threshold ORDER BY similarity DESC LIMIT 1
          |-- HIT -> backfill Redis -> return cached_response ("PGVECTOR")
          |-- MISS
               |
               v
          LLM.generate_response(prompt) -> response
               |
               v
          SemanticCache.set(prompt, response) -> write to Redis + pgvector
               |
               v
          return response ("MISS")
```

---

## 9. Dependency Map

```
src/main.py
  |-- config.py                     (Settings via pydantic-settings)
  |-- orchestration/router.py
  |     |-- adapter/factory.py
  |           |-- adapter/openai_adapter.py  -> openai>=1.14, tiktoken>=0.6, tenacity>=8.2
  |           |-- adapter/gemini_adapter.py  -> google-genai>=0.3, tenacity>=8.2
  |           |-- adapter/local_adapter.py   -> aiohttp>=3.9, tenacity>=8.2
  |           |-- telemetry/tracer.py        (always wraps)
  |           |-- security/guardrail.py      (opt-in wraps)
  |-- prompts/registry.py           -> asyncpg>=0.31, cachetools>=5.3

src/orchestration/swarm.py
  |-- adapter/base.py
  |-- prompts/registry.py
  |-- tools/es_discovery.py         -> elasticsearch>=9.4
  |-- utils/structured.py           -> pydantic>=2, tenacity>=8.2

src/cache/semantic.py               -> asyncpg>=0.31, redis>=8.0
src/memory/manager.py               -> adapter/base.py
src/validator/llm_judge.py          -> adapter/base.py, tenacity>=8.2
src/scraper/async_crawler.py        -> aiohttp>=3.9, beautifulsoup4>=4.12, adapter/factory.py

Web framework: fastapi>=0.137, served via uvicorn
```

---

## 10. Infrastructure & Deployment

### Docker Compose Services

| Service | Image | Purpose |
|---|---|---|
| `api` | Built from `Dockerfile` | FastAPI app on port 8000 |
| `db` | `ankane/pgvector` | PostgreSQL + pgvector extension |
| `redis` | `redis` | Cache and message broker |

```bash
docker-compose up --build -d   # Start all services
docker-compose down             # Stop and remove containers
```

### CI/CD (GitHub Actions)

- **File:** `.github/workflows/ci.yml`
- **Triggers:** push / PR to `main`
- **Steps:** `ruff check .` -> `mypy src/` -> `pytest tests/`
- **Python:** 3.11

---

## 11. Configuration Reference

### Full `.env` Variable List

| Variable | Used by | Default | Notes |
|---|---|---|---|
| `OPENAI_API_KEY` | `config.py` | — | Required for OpenAI |
| `GEMINI_API_KEY` | `config.py` | — | Required for Gemini |
| `DEFAULT_TEMPERATURE` | `config.py` | `0.7` | |
| `DATABASE_URL` | `main.py` lifespan | — | PostgreSQL DSN; optional |
| `REDIS_URL` | `main.py` lifespan | — | Redis DSN; optional |
| `PRIMARY_PROVIDER` | `main.py` lifespan | `openai` | |
| `FALLBACK_PROVIDER` | `main.py` lifespan | `gemini` | |
| `LOCAL_PROVIDER_BASE_URL` | `main.py` lifespan | — | Enables 3rd local tier |
| `LOCAL_MODEL` | `main.py` lifespan | `llama3.1` | |
| `LOCAL_PROVIDER_API_KEY` | `main.py` lifespan | — | Optional for vLLM |
| `PROMPT_CACHE_SIZE` | `main.py` lifespan | `256` | LRU cache entries |
| `PROMPT_CACHE_TTL` | `main.py` lifespan | `10.0` | Seconds |

> `.env.example` ships only `OPENAI_API_KEY` and `GEMINI_API_KEY`. All connection
> vars and provider-selection vars must be added manually.

---

## 12. Development Commands

```bash
# Activate existing venv (DO NOT create a new one — .venv/ is already present)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the API (bare-metal)
uvicorn src.main:app --host 0.0.0.0 --port 8000

# Run tests
pytest tests/
pytest tests/test_filename.py
pytest tests/test_filename.py::test_function

# Code quality
ruff check .      # lint
mypy src/         # type check

# Docker
docker-compose up --build -d
docker-compose down

# Knowledge graph
graphify query "how does failover work?"
graphify path "RouterManager" "BaseLLMAdapter"
graphify explain "PromptRegistry"
graphify update .    # update after code changes (AST-only, no API cost)
```

> Use `uvicorn`, not `uwsgi`. The README mentions uwsgi historically — that reference is stale.

---

## 13. Testing Strategy

- **Test location:** `tests/`
- **Runner:** pytest with config in `pytest.ini`
- **Module resolution:** `pythonpath = src` — top-level imports (`from adapter.base import ...`)
- **Async mode:** `asyncio_mode = auto` — no `@pytest.mark.asyncio` needed
- **Mocking:** `pytest-mock`
- **Coverage:** `pytest-cov`

Key test patterns:
- Adapter retry logic: mock provider errors to trigger tenacity backoff
- `StructuredGenerator` self-correction: mock LLM to return schema-violating JSON first
- `RouterManager` failover: mock primary to raise, verify fallback is called
- `InputGuardrailMiddleware`: inject PII patterns and injection strings

---

## 14. Known Patterns & Gotchas

### Adapter registration at import time
Importing `from adapter import ...` triggers `__init__.py`, which registers all three providers.
Unit tests that mock adapters must ensure the factory is populated before `create_adapter` is called.

### SemanticCache is not wired
The `SemanticCache` class and `with_semantic_cache` decorator exist but are **not yet applied to any adapter**.
Until wired, `ObservabilityMiddleware.cache_status` will always be `"DISABLED"`.
To enable: assign `SemanticCache` to an adapter instance and decorate its `generate_response`.

### RouterManager tool propagation
Registering a tool on `RouterManager` propagates to all inner adapters. If adapters are re-created
after tools are registered, tools must be re-registered.

### Gemini model attribute name
`GeminiAdapter` uses `self.model_name` (not `self.model`). `ObservabilityMiddleware` handles both.
New adapters should prefer `self.model` unless matching the Gemini pattern.

### LocalModelAdapter token counting
Returns `max(1, len(text) // 4)` — a character heuristic. `ConversationManager` windows measured
against a local model will differ from actual tokenizer behavior.

### Mid-stream streaming failover is impossible
Once the primary adapter starts yielding chunks, a mid-stream failure is re-raised. Only pre-stream
failures fall back to the fallback adapter.

### PromptRegistry inline migration
`PromptRegistry.initialize()` creates `prompt_versions` idempotently. `SemanticCache.get()` creates
its pgvector table inline on first use. No separate migration tool is required.

### requirements.txt has duplicate entries
`pytest`, `tiktoken`, and `mypy` appear twice with different version floors. `pip` resolves to the
higher constraint. Edit with care.

### graphify CLI path
Use the absolute path `/opt/homebrew/bin/graphify` if the `graphify` shell wrapper causes module
resolution errors in some environments.

---

*Generated by full repository analysis using graphify (771 nodes, 1546 edges) and direct source inspection of all 29 Python modules in `src/`.*
