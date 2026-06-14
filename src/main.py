import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

import asyncpg  # type: ignore
import redis.asyncio as redis  # type: ignore
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from orchestration.router import RouterManager

# -------------------------------------------------------------------------
# Pydantic Models for OpenAI-compatible API Contract
# -------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=None)
    stream: Optional[bool] = False
    max_tokens: Optional[int] = Field(default=None)


class AppState:
    db_pool: Optional[asyncpg.Pool] = None
    redis_pool: Optional[redis.Redis] = None
    router_manager: Optional[RouterManager] = None


app_state = AppState()

# -------------------------------------------------------------------------
# Application Lifecycle
# -------------------------------------------------------------------------


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

    # Initialize Redis connection pool
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        app_state.redis_pool = redis.from_url(redis_url, decode_responses=True)

    # Initialize the LLM Router Manager
    app_state.router_manager = RouterManager(primary_provider=os.getenv("PRIMARY_PROVIDER", "openai"), fallback_provider=os.getenv("FALLBACK_PROVIDER", "gemini"))

    yield

    # Cleanup resources on shutdown
    if app_state.db_pool:
        await app_state.db_pool.close()
    if app_state.redis_pool:
        await app_state.redis_pool.aclose()


# -------------------------------------------------------------------------
# FastAPI Application Initialization
# -------------------------------------------------------------------------

app = FastAPI(
    title="LLM Universal Adapter",
    description="Resilient backend service standardizing interactions with multiple LLMs.",
    version="1.0.0",
    lifespan=lifespan,
)

# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------


@app.get("/v1/health")
async def health_check() -> Dict[str, str]:
    """
    Basic health check endpoint to verify that the API is up and running.
    """
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> Any:
    """
    OpenAI-compatible chat completions endpoint.
    Supports both synchronous and streaming generation via the RouterManager.
    """
    if not app_state.router_manager:
        raise HTTPException(status_code=500, detail="RouterManager not initialized")

    # Adapt standard request to internal format
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    temperature = request.temperature if request.temperature is not None else getattr(settings, "default_temperature", 0.7)
    # Convert messages to a string prompt for the adapter interface
    prompt_str = json.dumps(messages)

    try:
        if request.stream:
            return StreamingResponse(_stream_generator(request.model, prompt_str, messages, temperature), media_type="text/event-stream")
        else:
            # Synchronous response generation
            response_content = await app_state.router_manager.generate_response(prompt=prompt_str, messages=messages, model=request.model, temperature=temperature)

            # Construct OpenAI-compatible synchronous response
            return JSONResponse(content={
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content,
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    # Optional token counting logic can be plugged here
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_generator(model: str, prompt: str, messages: List[Dict[str, str]], temperature: float) -> AsyncGenerator[str, None]:
    """
    Asynchronous generator yielding SSE formatted chunks for streaming responses.
    """
    if not app_state.router_manager:
        yield 'data: {"error": "RouterManager not initialized"}\n\n'
        return

    try:
        async for chunk in app_state.router_manager.agenerate_stream(prompt=prompt, messages=messages, model=model, temperature=temperature):
            # Serialize the chunk in the OpenAI Server-Sent Events format
            chunk_data = {"id": f"chatcmpl-{int(time.time())}", "object": "chat.completion.chunk", "created": int(time.time()), "model": model, "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk_data)}\n\n"

        # Send final stop sequence
        final_data = {"id": f"chatcmpl-{int(time.time())}", "object": "chat.completion.chunk", "created": int(time.time()), "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(final_data)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        # In a streaming response, HTTP status cannot be easily changed once streaming begins.
        # Yield the exception gracefully inside an event.
        error_data = {"error": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"
