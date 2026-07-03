"""
AgentWorker — standalone event-driven task consumer process.

This module is the entry point for the ``worker`` Docker service. It runs
an infinite asyncio loop that:

  1. Connects to Redis and blocks on ``BRPOP task:queue`` waiting for work.
  2. Deserializes the ``TaskRequest`` payload from JSON.
  3. Routes the task to the correct handler (crawl / swarm / chat).
  4. Publishes structured ``TaskEvent`` objects to ``events:{task_id}`` so
     WebSocket clients connected to the FastAPI layer receive live updates.
  5. Writes the last-known status to ``task:status:{task_id}`` for polling.

Run this module directly:
    python -m src.worker.agent_worker

Or via Docker:
    command: python -m src.worker.agent_worker
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from typing import Any, Callable, Coroutine, Dict, Optional

import redis.asyncio as aioredis

# ---------------------------------------------------------------------------
# Internal imports — adjust sys.path so the module is importable both as
# ``python -m src.worker.agent_worker`` and as a plain script.
# ---------------------------------------------------------------------------
# Insert the project root (parent of ``src``) so that ``from src.xxx``
# imports and bare ``from worker.task_models`` imports both resolve.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from worker.task_models import TaskEvent, TaskRequest, TaskStatus, TaskType  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("agent_worker")

# ---------------------------------------------------------------------------
# Redis key / channel helpers
# ---------------------------------------------------------------------------

TASK_QUEUE_KEY = "task:queue"


def _status_key(task_id: str) -> str:
    """Redis key used to store the last-known status for polling fallback."""
    return f"task:status:{task_id}"


def _events_channel(task_id: str) -> str:
    """Redis Pub/Sub channel name for live WebSocket event delivery."""
    return f"events:{task_id}"


# ---------------------------------------------------------------------------
# Event publishing helpers
# ---------------------------------------------------------------------------

# Type alias for the async publish callback injected into task handlers.
PublishFn = Callable[[TaskEvent], Coroutine[Any, Any, None]]

# TTL (seconds) for the status key stored in Redis (24 hours).
STATUS_TTL_SECONDS = 86_400


async def _publish_event(
    redis_client: aioredis.Redis,
    event: TaskEvent,
) -> None:
    """
    Publishes a ``TaskEvent`` to the appropriate Redis Pub/Sub channel and
    updates the persistent status key so polling clients stay in sync.

    Args:
        redis_client: Active async Redis connection.
        event: The event to publish.
    """
    payload = event.model_dump_json()
    channel = _events_channel(event.task_id)

    # Persist last-known status for GET /v1/tasks/{task_id} polling.
    await redis_client.set(_status_key(event.task_id), payload, ex=STATUS_TTL_SECONDS)

    # Broadcast to all subscribed WebSocket relay coroutines.
    await redis_client.publish(channel, payload)
    logger.debug("Published event status=%s task_id=%s", event.status, event.task_id)


# ---------------------------------------------------------------------------
# Task Handlers
# ---------------------------------------------------------------------------


async def _handle_crawl(
    task: TaskRequest,
    publish: PublishFn,
) -> None:
    """
    Crawl handler — runs ``AgenticScraper.crawl()`` and emits progress events.

    Expected payload keys:
        url (str): The seed URL to start crawling from.
        max_depth (int, optional): BFS depth limit. Defaults to 2.
        provider (str, optional): LLM provider name. Defaults to 'openai'.
    """
    from scraper.async_crawler import AgenticScraper  # type: ignore[import]

    url: str = task.payload.get("url", "")
    max_depth: int = int(task.payload.get("max_depth", 2))
    provider: str = task.payload.get("provider", os.getenv("PRIMARY_PROVIDER", "openai"))

    if not url:
        await publish(
            TaskEvent(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                message="Missing required payload field: 'url'.",
            )
        )
        return

    scraper = AgenticScraper(provider=provider, max_depth=max_depth)

    async def _progress(event: TaskEvent) -> None:
        """Forwards scraper progress events to the publish callback."""
        await publish(event)

    await scraper.crawl(start_url=url, task_id=task.task_id, publish_fn=_progress)


async def _handle_swarm(
    task: TaskRequest,
    publish: PublishFn,
) -> None:
    """
    Swarm handler — dispatches a ``SwarmTask`` through ``SwarmOrchestrator``.

    Expected payload keys:
        user_input (str): The raw user query or prompt.
        context (dict, optional): Free-form metadata passed to workers.
    """
    import uuid

    from adapter.factory import LLMAdapterFactory  # type: ignore[import]
    from orchestration.swarm import (  # type: ignore[import]
        ClassifierAgent,
        SearchAgent,
        SummaryAgent,
        SwarmOrchestrator,
        SwarmTask,
        UnknownIntentError,
    )
    from tools.es_discovery import ElasticsearchDiscoveryTool  # type: ignore[import]

    user_input: str = task.payload.get("user_input", "")
    context: Dict[str, Any] = task.payload.get("context", {})
    provider: str = task.payload.get("provider", os.getenv("PRIMARY_PROVIDER", "openai"))

    if not user_input:
        await publish(
            TaskEvent(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                message="Missing required payload field: 'user_input'.",
            )
        )
        return

    await publish(
        TaskEvent(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            message="Classifying intent and dispatching to worker agent.",
        )
    )

    try:
        adapter = LLMAdapterFactory.create_adapter(provider)
        classifier = ClassifierAgent(
            adapter=adapter,
            candidate_intents=["search", "summary"],
        )
        orchestrator = SwarmOrchestrator(classifier=classifier, default_intent="summary")

        # Register built-in workers
        from elasticsearch import Elasticsearch  # noqa: PLC0415

        es_host = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        es_client = Elasticsearch(hosts=[es_host])
        es_tool = ElasticsearchDiscoveryTool(es_client=es_client)
        orchestrator.register_worker(SearchAgent(es_tool=es_tool))
        orchestrator.register_worker(SummaryAgent(adapter=adapter))

        swarm_task = SwarmTask(
            request_id=str(uuid.uuid4()),
            user_input=user_input,
            context=context,
        )
        result = await orchestrator.dispatch(swarm_task)

        await publish(
            TaskEvent(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                message=f"Swarm task completed via worker '{result.worker}'.",
                data={
                    "intent": result.intent,
                    "worker": result.worker,
                    "output": result.output.model_dump(),
                    "confidence": result.classification.confidence,
                },
            )
        )
    except UnknownIntentError as exc:
        await publish(
            TaskEvent(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                message=f"Unknown intent from classifier: {exc}",
            )
        )
    except Exception as exc:
        logger.exception("Swarm task %s failed: %s", task.task_id, exc)
        await publish(
            TaskEvent(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                message=f"Swarm task execution error: {exc}",
            )
        )


async def _handle_chat(
    task: TaskRequest,
    publish: PublishFn,
) -> None:
    """
    Chat handler — runs a non-streaming LLM completion via RouterManager.

    Expected payload keys:
        model (str): LLM model name.
        messages (list): List of {role, content} message dicts.
        temperature (float, optional): Sampling temperature. Defaults to 0.7.
    """
    from orchestration.router import RouterManager  # type: ignore[import]

    model: str = task.payload.get("model", "gpt-4o-mini")
    messages: list = task.payload.get("messages", [])
    temperature: float = float(task.payload.get("temperature", 0.7))

    if not messages:
        await publish(
            TaskEvent(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                message="Missing required payload field: 'messages'.",
            )
        )
        return

    await publish(
        TaskEvent(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            message="Generating LLM response.",
        )
    )

    try:
        router = RouterManager(
            primary_provider=os.getenv("PRIMARY_PROVIDER", "openai"),
            fallback_provider=os.getenv("FALLBACK_PROVIDER", "gemini"),
        )
        import json as _json

        prompt = _json.dumps(messages)
        content = await router.generate_response(
            prompt=prompt,
            messages=messages,
            model=model,
            temperature=temperature,
        )
        await publish(
            TaskEvent(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                message="LLM response generated successfully.",
                data={"model": model, "content": content},
            )
        )
    except Exception as exc:
        logger.exception("Chat task %s failed: %s", task.task_id, exc)
        await publish(
            TaskEvent(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                message=f"LLM generation error: {exc}",
            )
        )


# ---------------------------------------------------------------------------
# Task Dispatcher
# ---------------------------------------------------------------------------

# Maps each TaskType to its handler coroutine.
_HANDLERS: Dict[TaskType, Any] = {
    TaskType.CRAWL: _handle_crawl,
    TaskType.SWARM: _handle_swarm,
    TaskType.CHAT: _handle_chat,
}


async def _process_task(task: TaskRequest, redis_client: aioredis.Redis) -> None:
    """
    Processes a single task end-to-end: emits a RUNNING event, delegates to
    the appropriate handler, and guarantees a terminal event is always published
    even if the handler raises an unexpected exception.

    Args:
        task: The deserialized TaskRequest from the queue.
        redis_client: Active async Redis connection for event publishing.
    """

    async def publish(event: TaskEvent) -> None:
        """Local wrapper that binds the shared Redis client."""
        await _publish_event(redis_client, event)

    # Announce that we are starting work on this task.
    await publish(
        TaskEvent(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            message=f"Worker picked up task of type '{task.task_type}'.",
            timestamp=time.time(),
        )
    )

    handler = _HANDLERS.get(task.task_type)
    if handler is None:
        await publish(
            TaskEvent(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                message=f"No handler registered for task type '{task.task_type}'.",
            )
        )
        return

    try:
        await handler(task, publish)
    except Exception as exc:
        # Catch-all safety net — individual handlers should not let exceptions
        # escape, but this guarantees clients always receive a terminal event.
        logger.exception("Unhandled exception in handler for task %s: %s", task.task_id, exc)
        await publish(
            TaskEvent(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                message=f"Internal worker error: {exc}",
            )
        )


# ---------------------------------------------------------------------------
# Main Event Loop
# ---------------------------------------------------------------------------


class AgentWorker:
    """
    The top-level worker process that consumes tasks from the Redis queue.

    Lifecycle:
        1. Connect to Redis.
        2. Enter the BRPOP loop, waiting up to ``poll_timeout`` seconds.
        3. On receiving a task, call ``_process_task`` concurrently via
           ``asyncio.create_task`` so the queue remains responsive.
        4. On SIGTERM / SIGINT, set the shutdown event and wait for all
           in-flight tasks to complete before exiting.
    """

    def __init__(
        self,
        redis_url: str,
        poll_timeout: float = 5.0,
        max_concurrent_tasks: int = 10,
    ) -> None:
        """
        Initializes the AgentWorker.

        Args:
            redis_url: Redis connection string (e.g. redis://redis:6379/0).
            poll_timeout: BRPOP block timeout in seconds. Controls how quickly
                          the worker reacts to shutdown signals when the queue
                          is empty.
            max_concurrent_tasks: Maximum number of tasks processed in parallel.
        """
        self.redis_url = redis_url
        self.poll_timeout = poll_timeout
        self.max_concurrent_tasks = max_concurrent_tasks
        self._shutdown_event = asyncio.Event()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._redis: Optional[aioredis.Redis] = None

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _register_signal_handlers(self) -> None:
        """
        Registers SIGTERM and SIGINT handlers for graceful shutdown.

        On signal receipt the shutdown event is set, causing the BRPOP loop
        to exit after its current blocking call. Any in-flight tasks are
        allowed to complete naturally.
        """
        loop = asyncio.get_running_loop()

        def _handle(sig: signal.Signals) -> None:
            logger.info("Received signal %s — initiating graceful shutdown.", sig.name)
            self._shutdown_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _handle, sig)

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Entry point for the worker process. Connects to Redis, registers
        signal handlers, then enters the task consumption loop.
        """
        self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
        self._semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        self._register_signal_handlers()
        logger.info(
            "AgentWorker started. Listening on queue '%s' (max_concurrent=%d).",
            TASK_QUEUE_KEY,
            self.max_concurrent_tasks,
        )

        active_tasks: list[asyncio.Task] = []

        try:
            while not self._shutdown_event.is_set():
                # BRPOP blocks for up to poll_timeout seconds, then returns None.
                # This allows the shutdown check to run periodically even when
                # the queue is empty.
                result = await self._redis.brpop(TASK_QUEUE_KEY, timeout=int(self.poll_timeout))
                if result is None:
                    # Queue empty — loop back to check shutdown_event.
                    continue

                _key, raw_payload = result
                await self._dispatch(str(raw_payload), active_tasks)

        finally:
            # Wait for all in-flight tasks to complete before exiting.
            if active_tasks:
                logger.info(
                    "Shutdown requested — waiting for %d in-flight task(s) to finish.",
                    len(active_tasks),
                )
                await asyncio.gather(*active_tasks, return_exceptions=True)

            if self._redis is not None:
                await self._redis.aclose()
            logger.info("AgentWorker shut down cleanly.")

    async def _dispatch(self, raw_payload: str, active_tasks: list) -> None:
        """
        Deserializes the raw JSON payload and schedules task execution.

        Args:
            raw_payload: JSON string from the Redis queue.
            active_tasks: Shared list of in-flight asyncio Tasks.
        """
        try:
            task_data = json.loads(raw_payload)
            task = TaskRequest.model_validate(task_data)
        except Exception as exc:
            logger.error("Failed to deserialize task payload: %s — payload: %s", exc, raw_payload)
            return

        logger.info(
            "Dequeued task task_id=%s type=%s",
            task.task_id,
            task.task_type,
        )

        # The semaphore limits the number of tasks executing concurrently.
        async def _run_with_semaphore() -> None:
            async with self._semaphore:  # type: ignore[union-attr]
                await _process_task(task, self._redis)  # type: ignore[arg-type]

        t = asyncio.create_task(_run_with_semaphore())

        # Clean up completed tasks from the tracking list to avoid memory growth.
        active_tasks[:] = [at for at in active_tasks if not at.done()]
        active_tasks.append(t)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


async def _main() -> None:
    """Async entry point. Reads configuration from environment variables."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    poll_timeout = float(os.getenv("WORKER_POLL_TIMEOUT", "5.0"))
    max_concurrent = int(os.getenv("WORKER_MAX_CONCURRENT_TASKS", "10"))

    worker = AgentWorker(
        redis_url=redis_url,
        poll_timeout=poll_timeout,
        max_concurrent_tasks=max_concurrent,
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(_main())
