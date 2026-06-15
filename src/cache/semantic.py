from functools import wraps
from typing import Any, Awaitable, Callable, Optional, Tuple, TypeVar, cast

import asyncpg  # type: ignore
import redis.asyncio as redis

F = TypeVar('F', bound=Callable[..., Awaitable[str]])


class SemanticCache:
    """
    A two-tier semantic cache layer that stores prompts and their corresponding responses.
    Layer 1 uses Redis for sub-millisecond exact string matching.
    Layer 2 queries PostgreSQL using the pgvector extension for semantic cosine similarity using embeddings.
    """

    def __init__(self, embedding_func: Callable[[str], Awaitable[list[float]]], redis_client: redis.Redis, db_pool: asyncpg.Pool, threshold: float = 0.95) -> None:
        """
        Initializes the SemanticCache.

        Args:
            embedding_func (Callable[[str], Awaitable[List[float]]]): An async function that
                takes a string (prompt) and returns a list of floats (its embedding).
            redis_client (redis.Redis): An active redis.asyncio.Redis instance.
            db_pool (asyncpg.Pool): An active asyncpg connection pool.
            threshold (float): The cosine similarity threshold to consider a match.
        """
        self.embedding_func = embedding_func
        self.redis_client = redis_client
        self.db_pool = db_pool
        self.threshold = threshold

    async def get(self, prompt: str) -> Tuple[Optional[str], str]:
        """
        Retrieves a cached response checking Layer 1 (Redis) then Layer 2 (PostgreSQL).

        Args:
            prompt (str): The incoming prompt to check against the cache.

        Returns:
            Tuple[Optional[str], str]: A tuple containing the cached response (if any)
            and the cache tier hit status ('REDIS', 'PGVECTOR', or 'MISS').
        """
        # Layer 1: Exact match via Redis
        redis_key = f"cache:exact:{prompt}"
        cached_exact = await self.redis_client.get(redis_key)
        if cached_exact:
            # cast is used because mypy infers bytes | str, but decode_responses=True guarantees str
            return cast(str, cached_exact), "REDIS"

        # Layer 2: Semantic match via PostgreSQL pgvector
        prompt_embedding = await self.embedding_func(prompt)
        # Format the vector as a string to pass it to the PostgreSQL query
        embedding_str = f"[{','.join(map(str, prompt_embedding))}]"

        # In pgvector, the `<=>` operator computes cosine distance.
        # Cosine similarity is 1 - distance.
        query = """
            SELECT response, 1 - (embedding <=> $1::vector) AS similarity
            FROM semantic_cache
            WHERE 1 - (embedding <=> $1::vector) >= $2
            ORDER BY similarity DESC
            LIMIT 1
        """
        async with self.db_pool.acquire() as conn:
            # Ensure the table and extension exist (ideally handled in a DB migration)
            await conn.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id SERIAL PRIMARY KEY,
                    prompt TEXT,
                    embedding vector,
                    response TEXT
                );
            """)
            row = await conn.fetchrow(query, embedding_str, self.threshold)

        if row:
            response = row["response"]
            # Backfill Layer 1 with the exact prompt for faster subsequent access
            await self.redis_client.set(redis_key, response)
            return response, "PGVECTOR"

        return None, "MISS"

    async def set(self, prompt: str, response: str) -> None:
        """
        Stores a prompt and its response in both the Redis and PostgreSQL caches.

        Args:
            prompt (str): The prompt to be cached.
            response (str): The response to be cached.
        """
        # Layer 1: Insert into Redis
        redis_key = f"cache:exact:{prompt}"
        await self.redis_client.set(redis_key, response)

        # Layer 2: Insert into PostgreSQL
        prompt_embedding = await self.embedding_func(prompt)
        embedding_str = f"[{','.join(map(str, prompt_embedding))}]"

        query = """
            INSERT INTO semantic_cache (prompt, embedding, response)
            VALUES ($1, $2::vector, $3)
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute(query, prompt, embedding_str, response)


def with_semantic_cache(func: F) -> F:
    """
    A decorator to inject semantic caching into an adapter's generation method.
    It expects the instance (self) to optionally have a `semantic_cache` attribute
    of type `SemanticCache`. If present, it utilizes the two-tier cache.

    Args:
        func: The asynchronous generation method to be decorated.

    Returns:
        The wrapped asynchronous method.
    """

    @wraps(func)
    async def wrapper(self: Any, prompt: str, *args: Any, **kwargs: Any) -> str:
        cache: Optional[SemanticCache] = getattr(self, "semantic_cache", None)

        if cache is not None:
            cached_response, tier = await cache.get(prompt)
            if cached_response is not None:
                # Surface the cache tier hit to the ObservabilityMiddleware
                self._last_cache_tier = tier
                return cached_response

        # Cache miss or cache not configured
        if cache is not None:
            self._last_cache_tier = "MISS"

        # Call the original LLM method
        response = await func(self, prompt, *args, **kwargs)

        if cache is not None:
            await cache.set(prompt, response)

        return response

    return cast(F, wrapper)
