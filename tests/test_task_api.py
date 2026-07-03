"""
Integration tests for the FastAPI event-driven task endpoints.

Tests:
    POST /v1/tasks  — returns 202 + task_id, enqueues task to Redis
    GET  /v1/tasks/{task_id}  — returns last-known status
    WS   /v1/ws/{task_id}    — streams Pub/Sub events via WebSocket

All tests use httpx.AsyncClient + ASGI transport so no real server is
started. Redis interactions are mocked at the app_state level.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

try:
    from httpx import AsyncClient, ASGITransport
except ImportError:
    pytest.skip("httpx not installed", allow_module_level=True)

# pyrefly: ignore [missing-import]
from src.main import app, app_state
# pyrefly: ignore [missing-import]
from src.worker.task_models import TaskEvent, TaskStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_redis():
    """Provides a pre-configured AsyncMock that stands in for app_state.redis_pool."""
    mock = AsyncMock()
    mock.set = AsyncMock(return_value=True)
    mock.lpush = AsyncMock(return_value=1)
    mock.get = AsyncMock(return_value=None)
    mock.publish = AsyncMock(return_value=1)
    return mock


@pytest_asyncio.fixture()
async def client(mock_redis):
    """
    Returns an httpx AsyncClient wired to the FastAPI ASGI app.
    The Redis pool is replaced with the mock so no real Redis is needed.
    """
    app_state.redis_pool = mock_redis
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# POST /v1/tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_task_returns_202(client, mock_redis):
    """Submitting a valid task should return HTTP 202 with a task_id."""
    payload = {
        "task_id": "unit-test-001",
        "task_type": "crawl",
        "payload": {"url": "https://example.com", "max_depth": 1},
    }
    response = await client.post("/v1/tasks", json=payload)

    assert response.status_code == 202
    body = response.json()
    assert body["task_id"] == "unit-test-001"
    assert body["status"] == "queued"


@pytest.mark.asyncio
async def test_submit_task_enqueues_to_redis(client, mock_redis):
    """POST /v1/tasks must call redis LPUSH exactly once with a serialized TaskRequest."""
    payload = {
        "task_id": "unit-test-002",
        "task_type": "swarm",
        "payload": {"user_input": "hello world"},
    }
    await client.post("/v1/tasks", json=payload)

    mock_redis.lpush.assert_called_once()
    queue_key, raw_task = mock_redis.lpush.call_args[0]
    assert queue_key == "task:queue"
    task_data = json.loads(raw_task)
    assert task_data["task_type"] == "swarm"
    assert task_data["task_id"] == "unit-test-002"


@pytest.mark.asyncio
async def test_submit_task_writes_initial_status(client, mock_redis):
    """POST /v1/tasks must store an initial QUEUED status in Redis."""
    payload = {
        "task_id": "unit-test-003",
        "task_type": "chat",
        "payload": {"model": "gpt-4o-mini", "messages": []},
    }
    await client.post("/v1/tasks", json=payload)

    mock_redis.set.assert_called()
    set_key = mock_redis.set.call_args[0][0]
    assert set_key == "task:status:unit-test-003"
    raw_event = mock_redis.set.call_args[0][1]
    event = TaskEvent.model_validate_json(raw_event)
    assert event.status == TaskStatus.QUEUED


@pytest.mark.asyncio
async def test_submit_task_without_redis_returns_503(mock_redis):
    """When Redis is unavailable the endpoint must return 503."""
    app_state.redis_pool = None
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        payload = {"task_id": "t", "task_type": "crawl", "payload": {}}
        response = await ac.post("/v1/tasks", json=payload)
    assert response.status_code == 503
    app_state.redis_pool = mock_redis  # restore


# ---------------------------------------------------------------------------
# GET /v1/tasks/{task_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_task_status_returns_200(client, mock_redis):
    """GET /v1/tasks/{id} should return 200 with the stored event."""
    event = TaskEvent(task_id="abc", status=TaskStatus.RUNNING, message="In progress")
    mock_redis.get = AsyncMock(return_value=event.model_dump_json())

    response = await client.get("/v1/tasks/abc")

    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == "abc"
    assert body["status"] == "running"


@pytest.mark.asyncio
async def test_get_task_status_not_found(client, mock_redis):
    """GET /v1/tasks/{id} should return 404 when no status key exists."""
    mock_redis.get = AsyncMock(return_value=None)

    response = await client.get("/v1/tasks/nonexistent-id")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /v1/health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check(client):
    """Health check endpoint should always return 200 with status 'ok'."""
    response = await client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
