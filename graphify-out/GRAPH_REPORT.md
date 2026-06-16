# Graph Report - .  (2026-06-16)

## Corpus Check
- Corpus is ~9,353 words - fits in a single context window. You may not need a graph.

## Summary
- 360 nodes · 512 edges · 43 communities (28 shown, 15 thin omitted)
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 84 edges (avg confidence: 0.66)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_BaseLLMAdapter Interface|BaseLLMAdapter Interface]]
- [[_COMMUNITY_Memory & Token Counting|Memory & Token Counting]]
- [[_COMMUNITY_Adapter Factory Wrapping|Adapter Factory Wrapping]]
- [[_COMMUNITY_FastAPI App & Lifespan|FastAPI App & Lifespan]]
- [[_COMMUNITY_Scraper & Tooling|Scraper & Tooling]]
- [[_COMMUNITY_Validator Tests & Fixtures|Validator Tests & Fixtures]]
- [[_COMMUNITY_CI & Project Docs|CI & Project Docs]]
- [[_COMMUNITY_OpenAI Adapter|OpenAI Adapter]]
- [[_COMMUNITY_Structured Generation & Generation Flow|Structured Generation & Generation Flow]]
- [[_COMMUNITY_Semantic Cache|Semantic Cache]]
- [[_COMMUNITY_Adapter Tests|Adapter Tests]]
- [[_COMMUNITY_Elasticsearch Discovery|Elasticsearch Discovery]]
- [[_COMMUNITY_Gemini Adapter|Gemini Adapter]]
- [[_COMMUNITY_Gemini Streaming & Tools|Gemini Streaming & Tools]]
- [[_COMMUNITY_Gemini Adapter Tests|Gemini Adapter Tests]]
- [[_COMMUNITY_Streaming Pipeline|Streaming Pipeline]]
- [[_COMMUNITY_Config Settings|Config Settings]]
- [[_COMMUNITY_Docker & Deployment|Docker & Deployment]]
- [[_COMMUNITY_Cross-Provider Tool Tests|Cross-Provider Tool Tests]]
- [[_COMMUNITY_Tool Generation Interface|Tool Generation Interface]]
- [[_COMMUNITY_Misc|Misc]]
- [[_COMMUNITY_Misc|Misc]]
- [[_COMMUNITY_Misc|Misc]]
- [[_COMMUNITY_Misc|Misc]]
- [[_COMMUNITY_Misc|Misc]]
- [[_COMMUNITY_Misc|Misc]]
- [[_COMMUNITY_Misc|Misc]]
- [[_COMMUNITY_Misc|Misc]]
- [[_COMMUNITY_Misc|Misc]]
- [[_COMMUNITY_Misc|Misc]]
- [[_COMMUNITY_Misc|Misc]]
- [[_COMMUNITY_Misc|Misc]]
- [[_COMMUNITY_Misc|Misc]]
- [[_COMMUNITY_Misc|Misc]]
- [[_COMMUNITY_Misc|Misc]]

## God Nodes (most connected - your core abstractions)
1. `BaseLLMAdapter` - 43 edges
2. `ObservabilityMiddleware` - 30 edges
3. `OpenAIAdapter` - 21 edges
4. `RouterManager` - 21 edges
5. `GeminiAdapter` - 20 edges
6. `ConversationManager` - 17 edges
7. `MockStructuredAdapter` - 15 edges
8. `LLMAdapterFactory` - 14 edges
9. `MockTokenAdapter` - 14 edges
10. `DummyInnerAdapter` - 14 edges

## Surprising Connections (you probably didn't know these)
- `Factory wraps adapters in ObservabilityMiddleware` --conceptually_related_to--> `test_register_and_create_adapter()`  [INFERRED]
  CLAUDE.md → tests/test_factory.py
- `Use uvicorn not uwsgi (ASGI)` --semantically_similar_to--> `docker-compose api uses uwsgi command (stale)`  [INFERRED] [semantically similar]
  CLAUDE.md → docker-compose.yml
- `Use uvicorn not uwsgi (ASGI)` --semantically_similar_to--> `README mentions uWSGI (stale)`  [INFERRED] [semantically similar]
  CLAUDE.md → README.md
- `DummyAdapter` --uses--> `BaseLLMAdapter`  [INFERRED]
  tests/test_factory.py → src/adapter/base.py
- `Any` --uses--> `BaseLLMAdapter`  [INFERRED]
  tests/test_factory.py → src/adapter/base.py

## Import Cycles
- 1-file cycle: `src/main.py -> src/main.py`
- 1-file cycle: `src/tools/es_discovery.py -> src/tools/es_discovery.py`

## Hyperedges (group relationships)
- **All BaseLLMAdapter implementations** — adapter_openai_adapter_openaiadapter, adapter_gemini_adapter_geminiadapter, telemetry_tracer_observabilitymiddleware, orchestration_router_routermanager [EXTRACTED 1.00]
- **Tenacity-retry decorated generation methods** — adapter_openai_adapter_generate_response, adapter_gemini_adapter_generate_response, utils_structured_generate, validator_llm_judge_evaluate [EXTRACTED 1.00]
- **OpenAI-compatible HTTP API surface** — main_chat_completions, main__stream_generator, main_chatcompletionrequest, main_chatmessage [EXTRACTED 1.00]
- **Mock adapter pattern across test files** — tests_test_telemetry_dummyinneradapter, tests_test_factory_dummyadapter, tests_test_manager_mocktokenadapter, tests_test_structured_mockstructuredadapter, tests_test_adapters_dummyinneradapter [INFERRED 0.85]
- **fast_retries fixture used across retry tests** — tests_test_adapters_fast_retries, tests_test_validator_fast_retries [EXTRACTED 1.00]
- **uWSGI references that are stale relative to uvicorn directive** — readme_uwsgi_stale, docker_compose_uwsgi_command, claudemd_uvicorn_not_uwsgi [INFERRED 0.85]

## Communities (43 total, 15 thin omitted)

### Community 0 - "BaseLLMAdapter Interface"
Cohesion: 0.07
Nodes (28): ABC, BaseLLMAdapter, Registers a local Python function to be exposed to the LLM via MCP., Forces the specific provider implementation to handle function calling         u, Generates a text response from the underlying LLM., Asynchronously generates a streamed text response from the LLM., Calculates the number of tokens for the given text., Abstract base class for all LLM providers.     Ensures a consistent interface ac (+20 more)

### Community 1 - "Memory & Token Counting"
Cohesion: 0.07
Nodes (26): BaseLLMAdapter.get_token_count (abstract), GeminiAdapter.get_token_count, OpenAIAdapter.get_token_count, ConversationManager._calculate_total_tokens, ConversationManager, Manages multi-turn conversation states for Large Language Models.     Implements, Initializes the ConversationManager.          Args:             adapter (BaseLLM, Appends a new message to the conversation and enforces the sliding window limit. (+18 more)

### Community 2 - "Adapter Factory Wrapping"
Cohesion: 0.10
Nodes (20): Registers a new LLM provider adapter., Creates and returns an instance of the requested LLM adapter.         Raises Val, BaseLLMAdapter, Any, BaseLLMAdapter, Any, ObservabilityMiddleware, Middleware that intercepts all outgoing LLM requests and incoming responses. (+12 more)

### Community 3 - "FastAPI App & Lifespan"
Cohesion: 0.09
Nodes (25): LLMAdapterFactory.create_adapter, BaseModel, FastAPI, lifespan (async context manager), A router manager that wraps the LLMAdapterFactory to provide a strict failover m, Initializes the RouterManager with a primary and fallback LLM provider., Registers a tool on the router and explicitly propagates it to both the, Attempts to generate a response using the primary adapter.         Silently fall (+17 more)

### Community 4 - "Scraper & Tooling"
Cohesion: 0.09
Nodes (16): LLMAdapterFactory, Factory class to instantiate the appropriate LLM adapter     based on provider n, Factory wraps adapters in ObservabilityMiddleware, ClientSession, AgenticScraper, Fetches the HTML content of a given URL., Uses the LLM to determine if the scraped content is highly relevant., Executes the breadth-first search crawling loop. (+8 more)

### Community 5 - "Validator Tests & Fixtures"
Cohesion: 0.13
Nodes (16): Any, BaseLLMAdapter, fast_retries(), Fixture to mock `time.sleep` used by `tenacity` during backoff.     This ensures, Test successful JSON parsing on the first attempt., Test JSON parsing when the LLM wraps the response in markdown blocks., Test retry mechanism triggers when required JSON keys are missing., Test retry mechanism triggers when the LLM returns invalid JSON. (+8 more)

### Community 6 - "CI & Project Docs"
Cohesion: 0.11
Nodes (18): CI Pipeline Test Job, Mypy type-check step, Pytest test step, Python 3.11 runtime, Ruff lint step, Memory sliding-window preserves system prompt, CLAUDE.md project instructions, pytest.ini pythonpath=src and asyncio_mode=auto (+10 more)

### Community 7 - "OpenAI Adapter"
Cohesion: 0.15
Nodes (9): OpenAIAdapter, Executes a prompt allowing the LLM to utilize registered tools., Adapter for OpenAI's language models.     Handles interactions with the OpenAI A, Initializes the OpenAI adapter.          Args:             api_key (Optional[str, Generates a text response from the OpenAI model.         Uses exponential backof, Calculates the number of tokens for the given text using tiktoken.          Args, Any, Test that a RateLimitError triggers a retry and eventually succeeds. (+1 more)

### Community 8 - "Structured Generation & Generation Flow"
Cohesion: 0.15
Nodes (15): BaseLLMAdapter.generate_response (abstract), GeminiAdapter.generate_response, GeminiAdapter.generate_with_tools, OpenAIAdapter._build_openai_tool_schema, OpenAIAdapter.generate_response, OpenAIAdapter.generate_with_tools, Settings (pydantic-settings), POST /v1/chat/completions (+7 more)

### Community 9 - "Semantic Cache"
Cohesion: 0.14
Nodes (11): A two-tier semantic cache layer that stores prompts and their corresponding resp, A decorator to inject semantic caching into an adapter's generation method., Initializes the SemanticCache.          Args:             embedding_func (Callab, Retrieves a cached response checking Layer 1 (Redis) then Layer 2 (PostgreSQL)., Stores a prompt and its response in both the Redis and PostgreSQL caches., SemanticCache, with_semantic_cache(), F (+3 more)

### Community 10 - "Adapter Tests"
Cohesion: 0.14
Nodes (12): fast_retries(), Fixture to mock `time.sleep` used by `tenacity` during backoff.     This ensures, Test successful text generation using the new isolated Client architecture     a, Ensure OpenAIAdapter raises ValueError if API key is completely missing., Test successful response generation without retries., Test that repeated failures eventually raise the underlying exception., Test token counting logic using tiktoken., test_gemini_adapter_generate_response_retry_success() (+4 more)

### Community 11 - "Elasticsearch Discovery"
Cohesion: 0.20
Nodes (8): BaseLLMAdapter.register_tool, Elasticsearch, RouterManager.register_tool, ElasticsearchDiscoveryTool, A tool for querying an Elasticsearch backend and enforcing source diversity., Initializes the Elasticsearch discovery tool.          Args:             es_clie, Extracts the domain from a given URL.          Args:             url (str): The, Searches Elasticsearch and returns a diverse set of results based on domain.

### Community 12 - "Gemini Adapter"
Cohesion: 0.18
Nodes (10): GeminiAdapter, Adapter for Google's Gemini language models.     Handles interactions using the, Initializes the Gemini adapter.          Args:             api_key (Optional[str, Asynchronously calculates the number of tokens for the given text using the upda, MonkeyPatch, Test that initializing the adapter without an API key raises a ValueError., Test token counting functionality using the updated genai SDK., test_gemini_adapter_get_token_count() (+2 more)

### Community 13 - "Gemini Streaming & Tools"
Cohesion: 0.29
Nodes (4): Asynchronously executes a prompt allowing the LLM to utilize registered tools., Asynchronously generates a text response from the Gemini model.         Uses exp, Asynchronously generates a streamed text response from the Gemini model., Any

### Community 14 - "Gemini Adapter Tests"
Cohesion: 0.40
Nodes (3): adapter(), mock_genai_client(), test_agenerate_stream_success()

### Community 15 - "Streaming Pipeline"
Cohesion: 0.50
Nodes (4): BaseLLMAdapter.agenerate_stream (abstract), _stream_generator (SSE), RouterManager.agenerate_stream, ObservabilityMiddleware.agenerate_stream

### Community 16 - "Config Settings"
Cohesion: 0.50
Nodes (3): BaseSettings, Configuration settings for the LLM Universal Adapter.     Loads values securely, Settings

### Community 17 - "Docker & Deployment"
Cohesion: 0.50
Nodes (4): docker-compose api service, docker-compose db service (pgvector), docker-compose redis service, README env vars table

### Community 18 - "Cross-Provider Tool Tests"
Cohesion: 0.50
Nodes (4): Test tool execution pipeline for OpenAI., Test tool execution pipeline for Gemini., test_gemini_adapter_generate_with_tools(), test_openai_adapter_generate_with_tools()

### Community 19 - "Tool Generation Interface"
Cohesion: 0.67
Nodes (3): BaseLLMAdapter.generate_with_tools (abstract), RouterManager.generate_with_tools, ObservabilityMiddleware.generate_with_tools

## Knowledge Gaps
- **40 isolated node(s):** `Redis`, `Pool`, `F`, `LLMAdapterFactory.register_adapter`, `OpenAIAdapter.agenerate_stream` (+35 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BaseLLMAdapter` connect `BaseLLMAdapter Interface` to `Memory & Token Counting`, `Adapter Factory Wrapping`, `FastAPI App & Lifespan`, `Scraper & Tooling`, `Validator Tests & Fixtures`, `OpenAI Adapter`, `Gemini Adapter`, `Gemini Streaming & Tools`?**
  _High betweenness centrality (0.364) - this node is a cross-community bridge._
- **Why does `ObservabilityMiddleware` connect `Adapter Factory Wrapping` to `BaseLLMAdapter Interface`, `FastAPI App & Lifespan`, `Scraper & Tooling`, `OpenAI Adapter`, `Semantic Cache`, `Gemini Adapter`?**
  _High betweenness centrality (0.175) - this node is a cross-community bridge._
- **Why does `RouterManager` connect `FastAPI App & Lifespan` to `BaseLLMAdapter Interface`, `Semantic Cache`, `Adapter Factory Wrapping`, `Scraper & Tooling`?**
  _High betweenness centrality (0.109) - this node is a cross-community bridge._
- **Are the 29 inferred relationships involving `BaseLLMAdapter` (e.g. with `LLMAdapterFactory` and `GeminiAdapter`) actually correct?**
  _`BaseLLMAdapter` has 29 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `ObservabilityMiddleware` (e.g. with `LLMAdapterFactory` and `.create_adapter()`) actually correct?**
  _`ObservabilityMiddleware` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `OpenAIAdapter` (e.g. with `BaseLLMAdapter` and `GeminiAdapter`) actually correct?**
  _`OpenAIAdapter` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `RouterManager` (e.g. with `LLMAdapterFactory.create_adapter` and `FastAPI`) actually correct?**
  _`RouterManager` has 8 INFERRED edges - model-reasoned connections that need verification._