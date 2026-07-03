"""
FastAPI application — LLM Universal Adapter.

This module provides the HTTP + WebSocket API layer. Long-running tasks
(crawl, swarm dispatch) are enqueued to Redis and executed by the standalone
``AgentWorker`` process, so this server never blocks its event loop.

New endpoints (Event-Driven Architecture):
    POST /v1/tasks            — Enqueue a task; returns HTTP 202 + task_id.
    GET  /v1/tasks/{task_id}  — Poll last-known task status.
    WS   /v1/ws/{task_id}     — Stream live task events via WebSocket.

Existing endpoints (unchanged):
    GET  /v1/health
    POST /v1/chat/completions  — Synchronous / streaming OpenAI-compatible chat.
    POST /v1/approval          — Resume or abort a HITL-suspended execution.
"""

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

import asyncpg  # type: ignore
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from orchestration.hitl import ApprovalRequiredError
from orchestration.router import RouterManager
from prompts import PromptRegistry
from worker.task_models import (
    TaskAcceptedResponse,
    TaskEvent,
    TaskRequest,
    TaskStatus,
    TaskStatusResponse,
)

# ---------------------------------------------------------------------------
# Pydantic Models for OpenAI-compatible API Contract
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=None)
    stream: Optional[bool] = False
    max_tokens: Optional[int] = Field(default=None)


class ApprovalRequest(BaseModel):
    state_id: str
    action: str = Field(..., description="Action to perform: 'approve' or 'abort'")


# ---------------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------------


class AppState:
    db_pool: Optional[asyncpg.Pool] = None
    redis_pool: Optional[aioredis.Redis] = None
    router_manager: Optional[RouterManager] = None
    prompt_registry: Optional[PromptRegistry] = None


app_state = AppState()

# ---------------------------------------------------------------------------
# Redis Key Helpers
# ---------------------------------------------------------------------------

# The name of the Redis list used as the task queue.
TASK_QUEUE_KEY = "task:queue"


def _status_key(task_id: str) -> str:
    """Redis key storing the last-known serialized TaskEvent for a task."""
    return f"task:status:{task_id}"


def _events_channel(task_id: str) -> str:
    """Redis Pub/Sub channel name for live WebSocket delivery."""
    return f"events:{task_id}"


# ---------------------------------------------------------------------------
# Application Lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for the FastAPI application.
    Handles dynamic initialization of database and cache connection pools,
    as well as core managers on startup, and graceful teardown on shutdown.
    """
    # Initialize PostgreSQL connection pool
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        app_state.db_pool = await asyncpg.create_pool(dsn=database_url)

    # Initialize Redis connection pool (shared by queue enqueue + Pub/Sub relay)
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        app_state.redis_pool = aioredis.from_url(redis_url, decode_responses=True)

    # Initialize the LLM Router Manager. Optionally wire a local provider
    # (Ollama/vLLM/LM Studio) to absorb trivial, high-volume traffic and save
    # cloud API spend. The local adapter is opt-in via LOCAL_PROVIDER_BASE_URL.
    local_base_url = os.getenv("LOCAL_PROVIDER_BASE_URL")
    local_provider_name: str | None = None
    local_kwargs: dict | None = None
    if local_base_url:
        local_provider_name = "local"
        local_kwargs = {
            "base_url": local_base_url,
            "model": os.getenv("LOCAL_MODEL", "llama3.1"),
        }
        if local_api_key := os.getenv("LOCAL_PROVIDER_API_KEY"):
            local_kwargs["api_key"] = local_api_key

    app_state.router_manager = RouterManager(
        primary_provider=os.getenv("PRIMARY_PROVIDER", "openai"),
        fallback_provider=os.getenv("FALLBACK_PROVIDER", "gemini"),
        local_provider=local_provider_name,
        local_kwargs=local_kwargs,
    )
    if app_state.redis_pool:
        app_state.router_manager.set_redis_client(app_state.redis_pool)

    # Initialize the PromptRegistry for dynamic, versioned prompts with A/B
    # allocation. The registry gracefully degrades to module-level defaults
    # when the database is unavailable.
    app_state.prompt_registry = PromptRegistry(
        db_pool=app_state.db_pool,
        cache_size=int(os.getenv("PROMPT_CACHE_SIZE", "256")),
        cache_ttl_seconds=float(os.getenv("PROMPT_CACHE_TTL", "10.0")),
    )
    await app_state.prompt_registry.initialize()

    yield

    # Cleanup resources on shutdown
    if app_state.db_pool:
        await app_state.db_pool.close()
    if app_state.redis_pool:
        await app_state.redis_pool.aclose()


# ---------------------------------------------------------------------------
# FastAPI Application Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM Universal Adapter",
    description=(
        "Resilient backend service standardizing interactions with multiple LLMs. "
        "Supports synchronous chat completions, event-driven long-running tasks, "
        "and real-time WebSocket streaming."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------


@app.exception_handler(ApprovalRequiredError)
async def approval_required_exception_handler(request: Any, exc: ApprovalRequiredError) -> JSONResponse:
    """
    Exception handler to catch ApprovalRequiredError and yield a 202 status
    code to the client so it can submit an approval or abort decision.
    """
    return JSONResponse(
        status_code=202,
        content={
            "status": "requires_approval",
            "state_id": exc.state_id,
            "tool_name": exc.tool_name,
            "tool_args": exc.tool_args,
            "message": str(exc),
        },
    )


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


@app.get("/v1/health")
async def health_check() -> Dict[str, str]:
    """Basic health check endpoint to verify that the API is up and running."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Event-Driven Task Endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/tasks", status_code=202, response_model=TaskAcceptedResponse)
async def submit_task(request: TaskRequest) -> TaskAcceptedResponse:
    """
    Enqueue a long-running task (crawl, swarm, or chat) for asynchronous
    execution by the AgentWorker process.

    The endpoint assigns a UUID4 ``task_id``, pushes the serialized payload
    to the ``task:queue`` Redis list, stores an initial QUEUED status, and
    returns HTTP 202 Accepted immediately — the client does not wait for the
    task to complete.

    Use ``GET /v1/tasks/{task_id}`` to poll or ``WS /v1/ws/{task_id}`` to
    stream live progress events.
    """
    if not app_state.redis_pool:
        raise HTTPException(status_code=503, detail="Redis not available — task queue is offline.")

    # Assign a correlation ID if the client did not provide one.
    task_id = request.task_id or str(uuid.uuid4())
    task = TaskRequest(
        task_id=task_id,
        task_type=request.task_type,
        payload=request.payload,
        priority=request.priority,
    )

    # Persist initial QUEUED status so polling clients see a result immediately.
    initial_event = TaskEvent(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        message="Task accepted and queued for processing.",
    )
    await app_state.redis_pool.set(
        _status_key(task_id),
        initial_event.model_dump_json(),
        ex=86_400,  # 24-hour TTL
    )

    # Push the serialized task to the left of the list; AgentWorker pops from
    # the right (BRPOP), which gives FIFO ordering.
    await app_state.redis_pool.lpush(TASK_QUEUE_KEY, task.model_dump_json())

    return TaskAcceptedResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        message="Task queued successfully.",
    )


@app.get("/v1/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    Returns the last-known status of a queued task.

    This endpoint is a polling fallback for clients that cannot maintain a
    WebSocket connection. For real-time updates prefer ``WS /v1/ws/{task_id}``.
    """
    if not app_state.redis_pool:
        raise HTTPException(status_code=503, detail="Redis not available.")

    raw = await app_state.redis_pool.get(_status_key(task_id))
    if raw is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")

    try:
        event = TaskEvent.model_validate_json(raw)
    except Exception:
        raise HTTPException(status_code=500, detail="Corrupted task status record.")

    return TaskStatusResponse(
        task_id=task_id,
        status=event.status,
        last_event=event,
        message=event.message,
    )


@app.websocket("/v1/ws/{task_id}")
async def websocket_task_stream(websocket: WebSocket, task_id: str) -> None:
    """
    WebSocket endpoint that streams live TaskEvent updates to the client.

    Lifecycle:
      1. Accept the WebSocket connection.
      2. Send the last-known status event immediately (so the client gets
         context even if it connects after RUNNING has been published).
      3. Subscribe to the ``events:{task_id}`` Redis Pub/Sub channel.
      4. Forward each published JSON event as a WebSocket text frame.
      5. Close the connection automatically on a terminal event
         (COMPLETED or FAILED), or on client disconnect.

    Args:
        websocket: The WebSocket connection managed by FastAPI/Starlette.
        task_id: The task correlation ID to subscribe to.
    """
    if not app_state.redis_pool:
        await websocket.close(code=1011, reason="Redis not available.")
        return

    await websocket.accept()

    # Send the last-known status immediately on connect so the client
    # has context regardless of when it connects in the task lifecycle.
    raw_status = await app_state.redis_pool.get(_status_key(task_id))
    if raw_status:
        await websocket.send_text(str(raw_status))

    # Create a dedicated subscriber connection (Pub/Sub requires its own
    # connection object separate from the command connection).
    subscriber: aioredis.Redis = aioredis.from_url(
        os.getenv("REDIS_URL", "redis://redis:6379/0"),
        decode_responses=True,
    )
    pubsub = subscriber.pubsub()
    channel = _events_channel(task_id)
    await pubsub.subscribe(channel)

    try:
        async for raw_message in pubsub.listen():
            # The first message from pubsub.listen() is a subscription
            # confirmation of type "subscribe" — skip it.
            if raw_message["type"] != "message":
                continue

            payload: str = raw_message["data"]
            await websocket.send_text(payload)

            # Close the connection cleanly once a terminal event is received.
            try:
                event = TaskEvent.model_validate_json(payload)
                if event.is_terminal:
                    break
            except Exception:
                # Malformed event — continue streaming rather than disconnecting.
                pass

    except WebSocketDisconnect:
        # Client disconnected; cancel the subscription and clean up.
        pass
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.close()
        await subscriber.aclose()
        # Best-effort graceful close; ignore if the connection is already gone.
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Synchronous / Streaming Chat Completions (unchanged)
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> Any:
    """
    OpenAI-compatible chat completions endpoint.
    Supports both synchronous and streaming generation via the RouterManager.
    """
    if not app_state.router_manager:
        raise HTTPException(status_code=500, detail="RouterManager not initialized.")

    # Adapt standard request to internal format
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    temperature = (
        request.temperature
        if request.temperature is not None
        else getattr(settings, "default_temperature", 0.7)
    )
    # Convert messages to a string prompt for the adapter interface
    prompt_str = json.dumps(messages)

    try:
        if request.stream:
            return StreamingResponse(
                _stream_generator(request.model, prompt_str, messages, temperature),
                media_type="text/event-stream",
            )
        else:
            # Synchronous response generation.
            # If tools are registered, invoke generate_with_tools for MCP support.
            if app_state.router_manager.tools:
                response_content = await app_state.router_manager.generate_with_tools(
                    prompt=prompt_str
                )
            else:
                response_content = await app_state.router_manager.generate_response(
                    prompt=prompt_str,
                    messages=messages,
                    model=request.model,
                    temperature=temperature,
                )

            # Construct OpenAI-compatible synchronous response
            return JSONResponse(
                content={
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": response_content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        # Optional token counting logic can be plugged here
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                }
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_generator(
    model: str,
    prompt: str,
    messages: List[Dict[str, str]],
    temperature: float,
) -> AsyncGenerator[str, None]:
    """
    Asynchronous generator yielding SSE-formatted chunks for streaming responses.
    """
    if not app_state.router_manager:
        yield 'data: {"error": "RouterManager not initialized"}\n\n'
        return

    try:
        async for chunk in app_state.router_manager.agenerate_stream(
            prompt=prompt, messages=messages, model=model, temperature=temperature
        ):
            # Serialize the chunk in the OpenAI Server-Sent Events format.
            chunk_data = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"

        # Send final stop sequence
        final_data = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_data)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        # In a streaming response, the HTTP status cannot be changed once
        # streaming has begun. Emit the error inside an SSE event instead.
        error_data = {"error": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"


# ---------------------------------------------------------------------------
# HITL Approval Endpoint (unchanged)
# ---------------------------------------------------------------------------


@app.post("/v1/approval")
async def handle_approval(request: ApprovalRequest) -> Any:
    """
    Manually resume or abort a suspended execution state.
    """
    if not app_state.redis_pool:
        raise HTTPException(status_code=500, detail="Redis connection pool not initialized.")
    if not app_state.router_manager:
        raise HTTPException(status_code=500, detail="RouterManager not initialized.")

    from orchestration.hitl import HITLStateManager

    manager = HITLStateManager(app_state.redis_pool)
    state = await manager.get_state(request.state_id)
    if not state:
        raise HTTPException(status_code=404, detail="Suspended state not found.")

    if state.status != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"State is not in pending status (current: {state.status}).",
        )

    if request.action == "abort":
        await manager.update_status(request.state_id, "aborted")
        return {"status": "aborted", "message": "Tool execution was aborted."}

    elif request.action == "approve":
        await manager.update_status(request.state_id, "approved")
        try:
            # Resume execution on the correct underlying adapter.
            response_content = await app_state.router_manager.resume_with_tools(state)
            # Delete state after successful completion.
            await manager.delete_state(request.state_id)

            return JSONResponse(
                content={
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": state.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": response_content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                }
            )
        except Exception as e:
            if isinstance(e, ApprovalRequiredError):
                raise
            raise HTTPException(status_code=500, detail=str(e))

    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid action. Must be 'approve' or 'abort'.",
        )
