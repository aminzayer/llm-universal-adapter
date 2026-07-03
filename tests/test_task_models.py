"""
Unit tests for src/worker/task_models.py.

Validates Pydantic model instantiation, field defaults, enum behavior,
is_terminal property, and JSON round-trip serialization.
"""

from __future__ import annotations

import time

import pytest

# pyrefly: ignore [missing-import]
from src.worker.task_models import (
    TaskAcceptedResponse,
    TaskEvent,
    TaskRequest,
    TaskStatus,
    TaskStatusResponse,
    TaskType,
)


# ---------------------------------------------------------------------------
# TaskType enum
# ---------------------------------------------------------------------------


class TestTaskType:
    def test_values(self):
        assert TaskType.CRAWL == "crawl"
        assert TaskType.SWARM == "swarm"
        assert TaskType.CHAT == "chat"

    def test_membership(self):
        assert "crawl" in TaskType._value2member_map_


# ---------------------------------------------------------------------------
# TaskStatus enum
# ---------------------------------------------------------------------------


class TestTaskStatus:
    def test_terminal_statuses(self):
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"

    def test_non_terminal_statuses(self):
        assert TaskStatus.QUEUED == "queued"
        assert TaskStatus.RUNNING == "running"


# ---------------------------------------------------------------------------
# TaskRequest model
# ---------------------------------------------------------------------------


class TestTaskRequest:
    def test_minimal_construction(self):
        req = TaskRequest(
            task_id="abc-123",
            task_type=TaskType.CRAWL,
        )
        assert req.task_id == "abc-123"
        assert req.task_type == TaskType.CRAWL
        assert req.payload == {}
        assert req.priority == 0

    def test_with_payload(self):
        req = TaskRequest(
            task_id="def-456",
            task_type=TaskType.SWARM,
            payload={"user_input": "hello"},
        )
        assert req.payload["user_input"] == "hello"

    def test_priority_non_negative(self):
        with pytest.raises(Exception):
            TaskRequest(task_id="x", task_type=TaskType.CHAT, priority=-1)

    def test_created_at_auto_set(self):
        before = time.time()
        req = TaskRequest(task_id="t", task_type=TaskType.CHAT)
        after = time.time()
        assert before <= req.created_at <= after

    def test_json_round_trip(self):
        req = TaskRequest(
            task_id="round-trip",
            task_type=TaskType.CRAWL,
            payload={"url": "https://example.com", "max_depth": 3},
        )
        serialized = req.model_dump_json()
        restored = TaskRequest.model_validate_json(serialized)
        assert restored.task_id == req.task_id
        assert restored.task_type == req.task_type
        assert restored.payload == req.payload


# ---------------------------------------------------------------------------
# TaskEvent model
# ---------------------------------------------------------------------------


class TestTaskEvent:
    def test_minimal_construction(self):
        event = TaskEvent(task_id="t1", status=TaskStatus.QUEUED)
        assert event.task_id == "t1"
        assert event.status == TaskStatus.QUEUED
        assert event.message == ""
        assert event.data == {}

    def test_is_terminal_completed(self):
        event = TaskEvent(task_id="t", status=TaskStatus.COMPLETED)
        assert event.is_terminal is True

    def test_is_terminal_failed(self):
        event = TaskEvent(task_id="t", status=TaskStatus.FAILED)
        assert event.is_terminal is True

    def test_is_not_terminal_running(self):
        event = TaskEvent(task_id="t", status=TaskStatus.RUNNING)
        assert event.is_terminal is False

    def test_is_not_terminal_queued(self):
        event = TaskEvent(task_id="t", status=TaskStatus.QUEUED)
        assert event.is_terminal is False

    def test_timestamp_auto_set(self):
        before = time.time()
        event = TaskEvent(task_id="t", status=TaskStatus.RUNNING)
        after = time.time()
        assert before <= event.timestamp <= after

    def test_json_round_trip(self):
        event = TaskEvent(
            task_id="rt",
            status=TaskStatus.COMPLETED,
            message="Done",
            data={"key": "value"},
        )
        restored = TaskEvent.model_validate_json(event.model_dump_json())
        assert restored.task_id == event.task_id
        assert restored.status == event.status
        assert restored.data == event.data


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class TestTaskAcceptedResponse:
    def test_defaults(self):
        resp = TaskAcceptedResponse(task_id="abc")
        assert resp.status == TaskStatus.QUEUED
        assert "queued" in resp.message.lower()


class TestTaskStatusResponse:
    def test_with_event(self):
        event = TaskEvent(task_id="abc", status=TaskStatus.RUNNING, message="Working…")
        resp = TaskStatusResponse(
            task_id="abc",
            status=TaskStatus.RUNNING,
            last_event=event,
            message="Working…",
        )
        assert resp.last_event is not None
        assert resp.last_event.status == TaskStatus.RUNNING
