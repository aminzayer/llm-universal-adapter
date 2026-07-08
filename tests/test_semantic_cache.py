import json
from typing import Any, Dict, List, Optional
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from adapter.openai_adapter import OpenAIAdapter
from telemetry.tracer import ObservabilityMiddleware


class MockRedis:
    """In-memory mock for Redis cache."""

    def __init__(self) -> None:
        self.store: Dict[str, str] = {}

    async def get(self, key: str) -> Optional[str]:
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> None:
        self.store[key] = value


class MockDbConnection:
    """Mock connection that records queries and simulates pgvector responses."""

    def __init__(self, pg_rows: List[Dict[str, Any]]) -> None:
        self.pg_rows = pg_rows
        self.executed_queries: List[tuple] = []

    async def execute(self, query: str, *args: Any) -> None:
        self.executed_queries.append((query, args))

    async def fetchrow(self, query: str, *args: Any) -> Optional[Dict[str, Any]]:
        self.executed_queries.append((query, args))
        if self.pg_rows:
            return self.pg_rows.pop(0)
        return None


class MockDbPool:
    """Mock database pool for asyncpg."""

    def __init__(self, conn: MockDbConnection) -> None:
        self.conn = conn

    def acquire(self) -> "_AcquireCtx":
        return _AcquireCtx(self.conn)


class _AcquireCtx:
    """Mock context manager for asyncpg connection acquisition."""

    def __init__(self, conn: MockDbConnection) -> None:
        self.conn = conn

    async def __aenter__(self) -> MockDbConnection:
        return self.conn

    async def __aexit__(self, *args: Any) -> None:
        pass


@pytest.mark.asyncio
@patch("src.adapter.openai_adapter.openai.AsyncClient")
async def test_semantic_cache_flow(mock_openai_client: MagicMock) -> None:
    # Set up OpenAI client mocks
    mock_instance = mock_openai_client.return_value

    # 1. Embeddings response
    mock_emb_res = MagicMock()
    mock_emb_res.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_instance.embeddings.create = AsyncMock(return_value=mock_emb_res)

    # 2. Completions response
    mock_chat_res = MagicMock()
    mock_chat_res.choices = [MagicMock(message=MagicMock(content="llm response"))]
    mock_instance.chat.completions.create = AsyncMock(return_value=mock_chat_res)

    # Set up mock DB and Redis
    redis_client = MockRedis()
    db_conn = MockDbConnection(pg_rows=[])
    db_pool = MockDbPool(db_conn)

    # Patch main app_state
    from main import app_state
    app_state.redis_pool = redis_client  # type: ignore
    app_state.db_pool = db_pool  # type: ignore

    try:
        adapter = OpenAIAdapter(api_key="fake-key")
        middleware = ObservabilityMiddleware(adapter, "openai")

        # --- TEST 1: Cache Miss ---
        with patch("telemetry.tracer.logger") as mock_logger:
            response = await middleware.generate_response("hello")
            assert response == "llm response"
            # Verify LLM was called
            mock_instance.chat.completions.create.assert_awaited_once()

            # Verify middleware logged MISS
            mock_logger.info.assert_called_once()
            log_entry = json.loads(mock_logger.info.call_args[0][0])
            assert log_entry["cache_status"] == "MISS"

        # Verify it was cached in Redis
        assert redis_client.store["cache:exact:hello"] == "llm response"

        # --- TEST 2: Exact Match (Redis Hit) ---
        # Reset OpenAI completions call count to prove short-circuiting
        mock_instance.chat.completions.create.reset_mock()

        with patch("telemetry.tracer.logger") as mock_logger:
            response = await middleware.generate_response("hello")
            assert response == "llm response"
            # Verify LLM was NOT called
            mock_instance.chat.completions.create.assert_not_awaited()

            # Verify middleware logged HIT
            mock_logger.info.assert_called_once()
            log_entry = json.loads(mock_logger.info.call_args[0][0])
            assert log_entry["cache_status"] == "HIT"

        # --- TEST 3: Semantic Match (PGVector Hit) ---
        # Clear redis cache to force PGVector check
        redis_client.store.clear()

        # Populate the database mock response
        db_conn.pg_rows = [{"response": "semantic pgvector response"}]
        mock_instance.chat.completions.create.reset_mock()

        with patch("telemetry.tracer.logger") as mock_logger:
            response = await middleware.generate_response("hi there")
            assert response == "semantic pgvector response"
            # Verify LLM was NOT called
            mock_instance.chat.completions.create.assert_not_awaited()

            # Verify middleware logged HIT
            mock_logger.info.assert_called_once()
            log_entry = json.loads(mock_logger.info.call_args[0][0])
            assert log_entry["cache_status"] == "HIT"

        # Verify it backfilled Redis
        assert redis_client.store["cache:exact:hi there"] == "semantic pgvector response"

    finally:
        # Clean up global state
        app_state.redis_pool = None
        app_state.db_pool = None
