# LLM Universal Adapter

[![CI Pipeline](https://github.com/aminzayer/llm-universal-adapter/actions/workflows/ci.yml/badge.svg)](https://github.com/aminzayer/llm-universal-adapter/actions)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.137+-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A resilient, async backend service that standardizes interactions with multiple LLM providers behind a single **OpenAI-compatible API**. Supports automatic failover, PII-safe prompt screening, structured JSON telemetry, versioned prompt management, and a two-tier semantic cache — all without changing your client code.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Tech Stack](#2-tech-stack)
3. [Architecture](#3-architecture)
   - [High-Level Diagram](#31-high-level-diagram)
   - [Middleware Stack](#32-middleware-stack)
   - [3-Tier Routing Strategy](#33-3-tier-routing-strategy)
   - [Swarm Orchestration](#34-swarm-orchestration)
4. [Project Structure](#4-project-structure)
5. [Prerequisites](#5-prerequisites)
6. [Environment Variables](#6-environment-variables)
7. [Installation & Setup](#7-installation--setup)
8. [Running the Application](#8-running-the-application)
9. [API Reference](#9-api-reference)
10. [Testing & Code Quality](#10-testing--code-quality)
11. [Deployment](#11-deployment)

---

## 1. Overview

**LLM Universal Adapter** provides:

- **Unified interface** — `OpenAIAdapter`, `GeminiAdapter`, and `LocalModelAdapter` all implement the same `BaseLLMAdapter` ABC, so swapping providers requires zero changes to application code.
- **Automatic failover** — requests route through a 3-tier chain: optional local model (vLLM/Ollama) → primary cloud → fallback cloud, with silent failover on any error.
- **Input security** — every prompt is screened for PII (emails, API keys, credit cards, SSNs, …) and prompt-injection attacks before reaching the LLM.
- **Observability** — structured JSON telemetry (latency, token counts, cache tier, errors) is emitted for every LLM call automatically.
- **Versioned prompts** — PostgreSQL-backed prompt templates with weighted A/B sampling and LRU caching.
- **Semantic cache** — two-tier cache: Redis exact-match (sub-millisecond) → PostgreSQL `pgvector` cosine similarity, reducing redundant LLM calls.
- **Multi-agent orchestration** — `SwarmOrchestrator` classifies user intent and routes to typed worker agents (`SearchAgent`, `SummaryAgent`, or custom workers).

---

## 2. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Web Framework | FastAPI (served by **uvicorn**) |
| LLM SDKs | `openai >= 1.14`, `google-genai >= 0.3` |
| Local LLM | vLLM / Ollama / LM Studio (OpenAI-compatible HTTP) |
| Async HTTP | `aiohttp >= 3.9` |
| Web Scraping | `beautifulsoup4 >= 4.12` |
| Resilience | `tenacity >= 8.2` (exponential backoff, retries) |
| Tokenization | `tiktoken >= 0.6` (OpenAI); API-based (Gemini) |
| Configuration | `pydantic-settings >= 2.2` |
| Database | PostgreSQL + `pgvector` (via `ankane/pgvector`) |
| DB Driver | `asyncpg >= 0.31` |
| Cache / Broker | Redis (`redis >= 8.0`) |
| Search | Elasticsearch `>= 9.4` |
| Caching Util | `cachetools >= 5.3` (LRU for prompt registry) |
| CI/CD | GitHub Actions (ruff + mypy + pytest on Python 3.11) |
| Containerization | Docker Compose |

---

## 3. Architecture

### 3.1 High-Level Diagram

```
 Client
   |
   |  POST /v1/chat/completions
   v
+----------------------------------------------------------+
|  FastAPI Application  (src/main.py)                      |
|  AppState: router_manager, db_pool, redis_pool,          |
|            prompt_registry                               |
+-----------------------------+----------------------------+
                              |
                              v
                    +-------------------+
                    |   RouterManager   |  <-- itself a BaseLLMAdapter
                    |  (3-tier routing) |
                    +--+------------+--+
          +------------+            +------------+
          v                                      v
  [Local Adapter]             [Primary Adapter] --> [Fallback Adapter]
  vLLM / Ollama               OpenAI GPT-4o          Gemini 2.5 Flash
  (trivial prompts only)      (default)               (on primary failure)

  Each adapter above is independently wrapped by the factory:

  +----------------------------+
  | InputGuardrailMiddleware   |  <- opt-in (enable_guardrail=True)
  |  PIIMasker                 |     screens raw prompt FIRST
  |  PromptInjectionDetector  |
  +----------------------------+
  | ObservabilityMiddleware    |  <- always applied by factory
  |  latency / tokens / cache  |
  +----------------------------+
  | Raw Provider Adapter       |
  |  OpenAIAdapter             |
  |  GeminiAdapter             |
  |  LocalModelAdapter         |
  +----------------------------+
```

---

### 3.2 Middleware Stack

Every adapter retrieved through `LLMAdapterFactory.create_adapter()` is automatically wrapped in layers. The wrapping order is (outermost → innermost):

```
InputGuardrailMiddleware   [opt-in via enable_guardrail=True]
  └── ObservabilityMiddleware  [always applied]
        └── RawProviderAdapter  (OpenAI / Gemini / Local)
```

**What each layer does:**

| Layer | Always? | Responsibility |
|---|---|---|
| `InputGuardrailMiddleware` | opt-in | PII masking + prompt-injection detection; raises `SecurityViolationError` on attack |
| `ObservabilityMiddleware` | always | Measures latency, counts tokens, logs structured JSON per call |
| Raw Adapter | — | Makes the actual API call (OpenAI / Gemini / local HTTP) |

The guardrail sees the **raw prompt before telemetry, the cache, or the provider** — so PII is never logged or cached.

---

### 3.3 3-Tier Routing Strategy

`RouterManager` decides which adapter to use based on a lightweight prompt heuristic:

```
Incoming prompt
       |
       +-- local adapter configured?
       |         NO  ──────────────────────────────────────────────────────+
       |         YES                                                        |
       |           +-- _is_trivial(prompt)?                                 |
       |           |         NO  ──────────────────────────────────────────+
       |           |         YES                                            |
       |           |           local_adapter.generate_response()            |
       |           |             SUCCESS -> return                          |
       |           |             FAIL -> log warning -> fall through        |
       |                                                                    |
       +-- primary_adapter.generate_response() <───────────────────────────+
                 SUCCESS -> return
                 FAIL -> log warning
       |
       +-- fallback_adapter.generate_response()
                 SUCCESS -> return
                 FAIL -> raise exception
```

**Trivial prompt detection** (keyword heuristic):
- Cheap keywords → local: `classify`, `categorize`, `label`, `sentiment`, `yes/no`, `is this`, `true or false`
- Complex keywords → cloud only: `reason`, `explain`, `plan`, `analyze`, `code`, `implement`, `refactor`, `design`
- Prompts > 800 characters → always cloud
- Tool-calling requests → always cloud (local models have inconsistent function-calling support)

---

### 3.4 Swarm Orchestration

For multi-agent workloads, `SwarmOrchestrator` provides intent classification and typed worker dispatch:

```
User Prompt
     |
     v
SwarmTask(request_id, user_input, context)
     |
     v
ClassifierAgent.classify(task)
  |-- PromptRegistry.get_prompt("classifier.system")
  |-- StructuredGenerator → LLM → ClassificationDecision(intent, confidence, reasoning)
     |
     +-- confidence below threshold? → use default_intent or first registered worker
     |
     v
get_worker(intent) → worker.run(task, decision)
  |-- SearchAgent  → ElasticsearchDiscoveryTool → SearchAgentOutput
  |-- SummaryAgent → PromptRegistry + StructuredGenerator → SummaryAgentOutput
  |-- <custom>    → implement BaseWorker.run() → any BaseModel
     |
     v
SwarmResult(request_id, intent, worker, output, classification)
```

Custom workers can be added via `orchestrator.register_worker(worker)`. All inter-agent payloads are **Pydantic models** — never raw dicts.

---

## 4. Project Structure

```
llm-universal-adapter/
├── src/
│   ├── main.py                  # FastAPI app entry point, lifespan, API routes
│   ├── config.py                # Pydantic Settings (.env loader)
│   ├── adapter/
│   │   ├── base.py              # BaseLLMAdapter ABC (interface contract)
│   │   ├── factory.py           # LLMAdapterFactory + middleware stacking
│   │   ├── openai_adapter.py    # OpenAI GPT-4o (tiktoken, function calling)
│   │   ├── gemini_adapter.py    # Google Gemini 2.5 Flash (google-genai SDK)
│   │   └── local_adapter.py     # OpenAI-compatible local HTTP (vLLM / Ollama)
│   ├── orchestration/
│   │   ├── router.py            # RouterManager (3-tier failover)
│   │   └── swarm.py             # SwarmOrchestrator, ClassifierAgent, workers
│   ├── prompts/
│   │   └── registry.py          # Versioned prompt registry (A/B, LRU cache)
│   ├── security/
│   │   └── guardrail.py         # InputGuardrailMiddleware, PIIMasker, InjectionDetector
│   ├── telemetry/
│   │   └── tracer.py            # ObservabilityMiddleware (JSON telemetry)
│   ├── cache/
│   │   └── semantic.py          # SemanticCache (Redis + pgvector), decorator
│   ├── memory/
│   │   └── manager.py           # ConversationManager (sliding-window context)
│   ├── utils/
│   │   └── structured.py        # StructuredGenerator (Pydantic schema enforcement)
│   ├── validator/
│   │   └── llm_judge.py         # StrictValidator (LLM-as-judge scoring)
│   ├── scraper/
│   │   └── async_crawler.py     # AgenticScraper (BFS + LLM relevance filter)
│   └── tools/
│       └── es_discovery.py      # ElasticsearchDiscoveryTool (diverse search)
├── tests/                       # pytest test suite
├── graphify-out/                # Knowledge graph (graph.json, GRAPH_REPORT.md)
├── .github/workflows/ci.yml     # CI: ruff + mypy + pytest on Python 3.11
├── docker-compose.yml           # api + db (pgvector) + redis
├── Dockerfile
├── requirements.txt
├── pyproject.toml               # ruff + mypy configuration
├── pytest.ini                   # pythonpath=src, asyncio_mode=auto
├── .env.example
├── CLAUDE.md                    # Developer guidance for AI coding assistants
└── antigravity.md               # Full architecture & module reference
```

---

## 5. Prerequisites

- **Python** 3.11 or higher
- **Docker** + **Docker Compose** (for containerized deployment)
- At least one LLM API key (`OPENAI_API_KEY` or `GEMINI_API_KEY`)

---

## 6. Environment Variables

Copy `.env.example` to `.env` and populate the values you need.

### LLM Keys (managed by `src/config.py` via Pydantic Settings)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key (required if primary/fallback is `openai`) |
| `GEMINI_API_KEY` | — | Google Gemini API key (required if primary/fallback is `gemini`) |
| `DEFAULT_TEMPERATURE` | `0.7` | Default sampling temperature |

### Infrastructure & Routing (read directly by `src/main.py` via `os.getenv`)

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | — | PostgreSQL DSN — e.g. `postgresql://user:pass@db:5432/rag_db` |
| `REDIS_URL` | — | Redis DSN — e.g. `redis://redis:6379/0` |
| `PRIMARY_PROVIDER` | `openai` | Primary cloud adapter (`openai` or `gemini`) |
| `FALLBACK_PROVIDER` | `gemini` | Fallback cloud adapter (`openai` or `gemini`) |
| `LOCAL_PROVIDER_BASE_URL` | — | Base URL for local model server (opt-in 3rd tier) |
| `LOCAL_MODEL` | `llama3.1` | Local model identifier (e.g. `qwen2.5:7b`) |
| `LOCAL_PROVIDER_API_KEY` | — | Optional bearer token for vLLM servers |
| `PROMPT_CACHE_SIZE` | `256` | LRU cache capacity for prompt registry |
| `PROMPT_CACHE_TTL` | `10.0` | Prompt cache TTL in seconds |

> **Note:** If `DATABASE_URL` or `REDIS_URL` are not set, the app starts successfully — the semantic cache and prompt persistence are simply skipped. The in-memory prompt defaults are used instead.

---

## 7. Installation & Setup

### a) Bare-metal (Virtual Environment)

```bash
# Clone the repository
git clone https://github.com/aminzayer/llm-universal-adapter.git
cd llm-universal-adapter

# A .venv/ is already included — activate it instead of creating a new one
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set at least one LLM API key
```

### b) Docker Compose (Recommended)

```bash
# Build and start api + db (pgvector) + redis
docker-compose up --build -d
```

---

## 8. Running the Application

### Bare-metal

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

> **Important:** Use `uvicorn` (ASGI), not `uwsgi` (WSGI). The app is an async FastAPI application.

### Docker Compose

```bash
docker-compose up        # foreground
docker-compose up -d     # background
docker-compose down      # stop and remove containers
```

---

## 9. API Reference

The service exposes an **OpenAI-compatible** REST API, making it a drop-in replacement for the OpenAI client in any application.

### `GET /v1/health`

Liveness check.

```json
{"status": "ok"}
```

### `POST /v1/chat/completions`

Standard OpenAI chat completions format. Supports both synchronous and streaming responses.

**Request body:**

```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize the latest AI research."}
  ],
  "temperature": 0.7,
  "stream": false,
  "max_tokens": 1024
}
```

**Synchronous response** (`stream: false`):

```json
{
  "id": "chatcmpl-1719230428",
  "object": "chat.completion",
  "created": 1719230428,
  "model": "gpt-4o",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

**Streaming response** (`stream: true`): Returns `text/event-stream` SSE chunks in OpenAI format, terminated by `data: [DONE]`.

---

## 10. Testing & Code Quality

Tests and quality tools are all pre-installed via `requirements.txt`.

```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_router.py

# Run a specific test function
pytest tests/test_router.py::test_failover_to_fallback

# Static type checking
mypy src/

# Lint and style check
ruff check .
```

**Test configuration** (`pytest.ini`):
- `pythonpath = src` — imports are top-level (e.g. `from adapter.base import BaseLLMAdapter`)
- `asyncio_mode = auto` — async test functions are discovered automatically; no `@pytest.mark.asyncio` needed

---

## 11. Deployment

### CI/CD (GitHub Actions)

Every push and pull request to `main` triggers `.github/workflows/ci.yml`:

1. Install Python 3.11
2. Install `requirements.txt`
3. `ruff check .` — lint
4. `mypy src/` — type check
5. `pytest tests/` — unit & integration tests

### Docker Compose

Three services:

| Service | Image | Port |
|---|---|---|
| `api` | Built from `Dockerfile` | `8000` |
| `db` | `ankane/pgvector` (PostgreSQL + pgvector) | `5432` |
| `redis` | `redis` | `6379` |

The `docker-compose.yml` is a ready-to-deploy baseline compatible with any cloud provider that supports Docker Compose, or as a starting point for Kubernetes (via Kompose or manual conversion).

---

> For a full module-by-module reference, middleware deep-dives, and data flow diagrams, see [`antigravity.md`](./antigravity.md).
