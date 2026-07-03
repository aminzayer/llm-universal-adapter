"""
Tests for the WebSocket /v1/ws/{task_id} endpoint.

Uses Starlette's built-in WebSocket test client (starlette.testclient.TestClient
with WebSocket context manager) which does not require a running server.

The app lifespan, RouterManager, PromptRegistry, asyncpg, and Redis Pub/Sub
are all mocked so tests run without any real infrastructure.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# pyrefly: ignore [missing-import]
from src.main import app, app_state
# pyrefly: ignore [missing-import]
from src.worker.task_models import TaskEvent, TaskStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pubsub_messages(*events: TaskEvent) -> list[dict]:
    """
    Converts TaskEvent objects into the dict structure that redis.asyncio
    pubsub.listen() yields, including a leading 'subscribe' confirmation.
    """
    messages = [{"type": "subscribe", "data": 1}]
    for event in events:
        messages.append({"type": "message", "data": event.model_dump_json()})
    return messages


def _make_subscriber_mock(pubsub_messages: list[dict]) -> AsyncMock:
    """
    Builds a fully-mocked Redis subscriber suitable for patching aioredis.from_url.
    The pubsub.listen() async generator yields the provided message dicts.
    """

    async def fake_listen():
        for msg in pubsub_messages:
            yield msg

    pubsub_mock = AsyncMock()
    pubsub_mock.subscribe = AsyncMock()
    pubsub_mock.unsubscribe = AsyncMock()
    pubsub_mock.close = AsyncMock()
    pubsub_mock.listen = fake_listen

    subscriber_mock = AsyncMock()
    subscriber_mock.pubsub = MagicMock(return_value=pubsub_mock)
    subscriber_mock.aclose = AsyncMock()
    return subscriber_mock


def _lifespan_patches(subscriber_mock: AsyncMock):
    """
    Returns a combined context manager that stubs out all lifespan side-effects
    (RouterManager instantiation, PromptRegistry.initialize, asyncpg, and the
    secondary aioredis.from_url call used by the WebSocket subscriber).
    """
    return [
        patch("src.main.RouterManager", return_value=MagicMock(set_redis_client=MagicMock())),
        patch("src.main.PromptRegistry", return_value=MagicMock(initialize=AsyncMock())),
        patch("src.main.asyncpg.create_pool", new=AsyncMock(return_value=None)),
        patch("src.main.aioredis.from_url", return_value=subscriber_mock),
    ]


# ---------------------------------------------------------------------------
# WebSocket endpoint tests
# ---------------------------------------------------------------------------


class TestWebSocketTaskStream:
    """Tests for WS /v1/ws/{task_id}."""

    def test_sends_cached_status_on_connect(self):
        """
        On connect, the server should immediately send the last-known status
        stored in the Redis status key before subscribing to new events.
        """
        from starlette.testclient import TestClient

        cached_event = TaskEvent(
            task_id="ws-test-001",
            status=TaskStatus.RUNNING,
            message="Already running",
        )
        terminal_event = TaskEvent(task_id="ws-test-001", status=TaskStatus.COMPLETED)

        mock_redis_pool = AsyncMock()
        mock_redis_pool.get = AsyncMock(return_value=cached_event.model_dump_json())
        app_state.redis_pool = mock_redis_pool

        subscriber_mock = _make_subscriber_mock(_make_pubsub_messages(terminal_event))
        received: list[str] = []

        with patch("src.main.RouterManager", return_value=MagicMock(set_redis_client=MagicMock())), \
             patch("src.main.PromptRegistry", return_value=MagicMock(initialize=AsyncMock())), \
             patch("src.main.asyncpg.create_pool", new=AsyncMock(return_value=None)), \
             patch("src.main.aioredis.from_url", return_value=subscriber_mock):

            with TestClient(app) as client:
                with client.websocket_connect("/v1/ws/ws-test-001") as ws:
                    # First message: cached status sent immediately on connect.
                    first_msg = ws.receive_text()
                    received.append(first_msg)
                    # Second message: COMPLETED terminal event from Pub/Sub.
                    try:
                        second_msg = ws.receive_text()
                        received.append(second_msg)
                    except Exception:
                        pass  # Connection closed after terminal event

        assert len(received) >= 1
        first_event = TaskEvent.model_validate_json(received[0])
        assert first_event.status == TaskStatus.RUNNING

    def test_closes_after_terminal_event(self):
        """
        The server must close the WebSocket connection after forwarding a
        terminal event (COMPLETED or FAILED).
        """
        from starlette.testclient import TestClient

        terminal_event = TaskEvent(
            task_id="ws-test-002",
            status=TaskStatus.COMPLETED,
            message="All done",
        )

        mock_redis_pool = AsyncMock()
        mock_redis_pool.get = AsyncMock(return_value=None)  # No cached status
        app_state.redis_pool = mock_redis_pool

        subscriber_mock = _make_subscriber_mock(_make_pubsub_messages(terminal_event))
        messages: list[str] = []

        with patch("src.main.RouterManager", return_value=MagicMock(set_redis_client=MagicMock())), \
             patch("src.main.PromptRegistry", return_value=MagicMock(initialize=AsyncMock())), \
             patch("src.main.asyncpg.create_pool", new=AsyncMock(return_value=None)), \
             patch("src.main.aioredis.from_url", return_value=subscriber_mock):

            with TestClient(app) as client:
                with client.websocket_connect("/v1/ws/ws-test-002") as ws:
                    try:
                        while True:
                            msg = ws.receive_text()
                            messages.append(msg)
                    except Exception:
                        pass  # Server closed the connection after terminal event

        assert len(messages) >= 1
        last_event = TaskEvent.model_validate_json(messages[-1])
        assert last_event.is_terminal is True

    def test_websocket_closes_without_redis(self):
        """
        When Redis is unavailable the WebSocket must be closed with code 1011.
        """
        from starlette.testclient import TestClient

        app_state.redis_pool = None
        # Pass a dummy subscriber (it will never be reached — the endpoint
        # returns before subscribing when redis_pool is None).
        subscriber_mock = _make_subscriber_mock([])

        with patch("src.main.RouterManager", return_value=MagicMock(set_redis_client=MagicMock())), \
             patch("src.main.PromptRegistry", return_value=MagicMock(initialize=AsyncMock())), \
             patch("src.main.asyncpg.create_pool", new=AsyncMock(return_value=None)), \
             patch("src.main.aioredis.from_url", return_value=subscriber_mock):

            with TestClient(app) as client:
                with pytest.raises(Exception):
                    with client.websocket_connect("/v1/ws/no-redis-task") as ws:
                        ws.receive_text()
