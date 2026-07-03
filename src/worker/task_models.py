"""
Shared Pydantic models for the Event-Driven task queue.

These models define the contract between the FastAPI API layer and the
AgentWorker process. They are serialized to/from JSON when tasks are
pushed onto the Redis list and when events are published to Pub/Sub channels.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    """Supported task types that can be submitted to the task queue."""

    CRAWL = "crawl"
    SWARM = "swarm"
    CHAT = "chat"


class TaskStatus(str, Enum):
    """
    Lifecycle states of a queued task.

    Transitions:  QUEUED -> RUNNING -> COMPLETED
                                    -> FAILED
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Task Request — pushed onto Redis list by the API
# ---------------------------------------------------------------------------


class TaskRequest(BaseModel):
    """
    Payload submitted by the client via POST /v1/tasks.

    The ``payload`` field is task-type specific:
      - CRAWL:  {"url": str, "max_depth": int, "provider": str}
      - SWARM:  {"user_input": str, "context": dict}
      - CHAT:   {"model": str, "messages": list, "temperature": float}
    """

    task_id: str = Field(..., description="UUID4 assigned by the API before enqueue.")
    task_type: TaskType = Field(..., description="The type of work to perform.")
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task-type-specific parameters.",
    )
    priority: int = Field(
        default=0,
        ge=0,
        description="Higher values indicate higher priority (reserved for future use).",
    )
    created_at: float = Field(
        default_factory=time.time,
        description="Unix timestamp when the task was created.",
    )


# ---------------------------------------------------------------------------
# Task Event — published to Redis Pub/Sub channel by the worker
# ---------------------------------------------------------------------------


class TaskEvent(BaseModel):
    """
    A structured state-change notification published by the AgentWorker.

    Events are published to the Redis channel ``events:{task_id}`` so that
    connected WebSocket clients receive real-time progress updates.
    """

    task_id: str = Field(..., description="The task this event belongs to.")
    status: TaskStatus = Field(..., description="Current lifecycle status.")
    message: str = Field(default="", description="Human-readable progress message.")
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured payload (e.g. crawl results, LLM output).",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp when this event was emitted.",
    )

    @property
    def is_terminal(self) -> bool:
        """Returns True if this event marks the end of the task lifecycle."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)


# ---------------------------------------------------------------------------
# API Response Models
# ---------------------------------------------------------------------------


class TaskAcceptedResponse(BaseModel):
    """HTTP 202 response body returned by POST /v1/tasks."""

    task_id: str
    status: TaskStatus = TaskStatus.QUEUED
    message: str = "Task queued successfully."


class TaskStatusResponse(BaseModel):
    """HTTP 200 response body returned by GET /v1/tasks/{task_id}."""

    task_id: str
    status: Optional[TaskStatus] = None
    last_event: Optional[TaskEvent] = None
    message: str = ""


__all__ = [
    "TaskAcceptedResponse",
    "TaskEvent",
    "TaskRequest",
    "TaskStatus",
    "TaskStatusResponse",
    "TaskType",
]
