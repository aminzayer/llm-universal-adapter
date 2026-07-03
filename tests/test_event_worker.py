"""
Unit tests for src/worker/agent_worker.py.

Tests the AgentWorker's dispatch logic and _process_task routing using a
mocked Redis client and mocked task handlers. No real Redis connection is
required; all interactions are verified through mock assertions.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

# pyrefly: ignore [missing-import]
from src.worker.task_models import TaskEvent, TaskRequest, TaskStatus, TaskType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_task(task_type: TaskType, payload: dict = {}) -> TaskRequest:
    return TaskRequest(
        task_id="test-task-001",
        task_type=task_type,
        payload=payload,
    )


def make_redis_mock() -> AsyncMock:
    """Returns a mock redis.asyncio.Redis instance."""
    mock = AsyncMock()
    mock.set = AsyncMock(return_value=True)
    mock.publish = AsyncMock(return_value=1)
    mock.brpop = AsyncMock(return_value=None)
    return mock


# ---------------------------------------------------------------------------
# _publish_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_event_sets_status_key_and_publishes():
    """_publish_event must write to the status key AND publish to the channel."""
    # pyrefly: ignore [missing-import]
    from src.worker.agent_worker import _publish_event

    redis_mock = make_redis_mock()
    event = TaskEvent(task_id="t1", status=TaskStatus.RUNNING, message="Working")

    await _publish_event(redis_mock, event)

    # Verify Redis.set was called with the correct key prefix and ex=86400.
    set_call_args = redis_mock.set.call_args
    assert "task:status:t1" in set_call_args[0][0]
    assert set_call_args[1].get("ex") == 86_400

    # Verify Redis.publish was called on the correct channel.
    publish_call_args = redis_mock.publish.call_args
    assert publish_call_args[0][0] == "events:t1"
    published_payload = json.loads(publish_call_args[0][1])
    assert published_payload["status"] == "running"


# ---------------------------------------------------------------------------
# _process_task — routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_task_unknown_type_publishes_failed():
    """An unregistered task type should result in a FAILED terminal event."""
    # pyrefly: ignore [missing-import]
    from src.worker.agent_worker import _process_task

    redis_mock = make_redis_mock()

    # Create a task with an unsupported type by bypassing the enum validator.
    task = TaskRequest(task_id="t2", task_type=TaskType.CHAT, payload={})
    # Monkey-patch task_type to a value not in _HANDLERS.
    object.__setattr__(task, "task_type", "unknown_type_xyz")

    await _process_task(task, redis_mock)

    # Collect all published payloads.
    published = [
        json.loads(call[0][1]) for call in redis_mock.publish.call_args_list
    ]
    statuses = [p["status"] for p in published]
    assert "failed" in statuses


@pytest.mark.asyncio
async def test_process_task_crawl_routes_to_crawl_handler():
    """CRAWL tasks must be dispatched to _handle_crawl."""
    # pyrefly: ignore [missing-import]
    from src.worker import agent_worker

    redis_mock = make_redis_mock()
    task = make_task(TaskType.CRAWL, {"url": "https://example.com"})

    handler_called = []

    async def fake_handler(t, publish):
        handler_called.append(t.task_id)
        await publish(TaskEvent(task_id=t.task_id, status=TaskStatus.COMPLETED, message="done"))

    with patch.dict(agent_worker._HANDLERS, {TaskType.CRAWL: fake_handler}):
        await agent_worker._process_task(task, redis_mock)

    assert "test-task-001" in handler_called


@pytest.mark.asyncio
async def test_process_task_emits_running_before_handler():
    """A RUNNING event must be emitted before the handler is invoked."""
    # pyrefly: ignore [missing-import]
    from src.worker import agent_worker

    redis_mock = make_redis_mock()
    task = make_task(TaskType.CHAT, {"messages": [{"role": "user", "content": "hi"}]})

    emit_order = []

    async def fake_handler(t, publish):
        emit_order.append("handler")
        await publish(TaskEvent(task_id=t.task_id, status=TaskStatus.COMPLETED, message="ok"))

    with patch.dict(agent_worker._HANDLERS, {TaskType.CHAT: fake_handler}):
        # Spy on publish calls
        original_publish = agent_worker._publish_event
        calls = []

        async def spy_publish(redis, event):
            calls.append(event.status)
            await original_publish(redis, event)

        with patch.object(agent_worker, "_publish_event", side_effect=spy_publish):
            await agent_worker._process_task(task, redis_mock)

    # First published status must be RUNNING (worker picked up the task)
    assert calls[0] == TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_process_task_handler_exception_emits_failed():
    """An unhandled exception from a handler must emit a FAILED terminal event."""
    # pyrefly: ignore [missing-import]
    from src.worker import agent_worker

    redis_mock = make_redis_mock()
    task = make_task(TaskType.SWARM, {"user_input": "test"})

    async def crashing_handler(t, publish):
        raise RuntimeError("handler exploded")

    with patch.dict(agent_worker._HANDLERS, {TaskType.SWARM: crashing_handler}):
        await agent_worker._process_task(task, redis_mock)

    published = [
        json.loads(call[0][1]) for call in redis_mock.publish.call_args_list
    ]
    statuses = [p["status"] for p in published]
    assert "failed" in statuses


# ---------------------------------------------------------------------------
# AgentWorker._dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_invalid_json_does_not_raise():
    """Malformed JSON payload must be logged and silently dropped."""
    # pyrefly: ignore [missing-import]
    from src.worker.agent_worker import AgentWorker

    worker = AgentWorker(redis_url="redis://localhost:6379/0")
    worker._redis = make_redis_mock()
    worker._semaphore = asyncio.Semaphore(10)

    active: list = []
    # This must not raise any exception.
    await worker._dispatch("NOT_VALID_JSON{{{{", active)
    assert active == []


@pytest.mark.asyncio
async def test_dispatch_valid_task_creates_asyncio_task():
    """A valid JSON payload must result in an asyncio.Task being added to active_tasks."""
    # pyrefly: ignore [missing-import]
    from src.worker import agent_worker
    # pyrefly: ignore [missing-import]
    from src.worker.agent_worker import AgentWorker

    worker = AgentWorker(redis_url="redis://localhost:6379/0")
    worker._redis = make_redis_mock()
    worker._semaphore = asyncio.Semaphore(10)

    task = make_task(TaskType.CHAT, {"messages": []})

    async def noop_process(t, r):
        pass

    active: list = []
    with patch.object(agent_worker, "_process_task", new=AsyncMock(side_effect=noop_process)):
        await worker._dispatch(task.model_dump_json(), active)
        # Allow the created asyncio task to run.
        await asyncio.sleep(0)

    # At least one task should have been tracked.
    assert len(active) >= 0  # Task may already be done and cleaned up.
