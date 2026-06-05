import math
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, cast

F = TypeVar('F', bound=Callable[..., Awaitable[str]])


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Computes the cosine similarity between two vectors.

    Args:
        vec1 (List[float]): The first vector.
        vec2 (List[float]): The second vector.

    Returns:
        float: The cosine similarity score between -1.0 and 1.0.
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


class SemanticCache:
    """
    A semantic cache layer that stores prompts and their corresponding responses.
    It uses embeddings to find semantically similar prompts to save LLM calls.
    """

    def __init__(self, embedding_func: Callable[[str], Awaitable[List[float]]], threshold: float = 0.95) -> None:
        """
        Initializes the SemanticCache.

        Args:
            embedding_func (Callable[[str], Awaitable[List[float]]]): An async function that
                takes a string (prompt) and returns a list of floats (its embedding).
            threshold (float): The cosine similarity threshold to consider a match.
        """
        self.embedding_func = embedding_func
        self.threshold = threshold
        self.cache: List[Dict[str, Any]] = []

    async def get(self, prompt: str) -> Optional[str]:
        """
        Retrieves a cached response if a semantically similar prompt exists.

        Args:
            prompt (str): The incoming prompt to check against the cache.

        Returns:
            Optional[str]: The cached response if a match is found, otherwise None.
        """
        if not self.cache:
            return None

        prompt_embedding = await self.embedding_func(prompt)
        best_score = -1.0
        best_response: Optional[str] = None

        for entry in self.cache:
            score = cosine_similarity(prompt_embedding, entry["embedding"])
            if score > best_score:
                best_score = score
                best_response = entry["response"]

        if best_score >= self.threshold:
            return best_response

        return None

    async def set(self, prompt: str, response: str) -> None:
        """
        Stores a prompt and its response in the cache.

        Args:
            prompt (str): The prompt to be cached.
            response (str): The response to be cached.
        """
        prompt_embedding = await self.embedding_func(prompt)
        self.cache.append({"prompt": prompt, "embedding": prompt_embedding, "response": response})


def with_semantic_cache(func: F) -> F:
    """
    A decorator to inject semantic caching into an adapter's generation method.
    It expects the instance (self) to optionally have a `semantic_cache` attribute
    of type `SemanticCache`. If present, it checks the cache before calling the LLM.

    Args:
        func: The asynchronous generation method to be decorated.

    Returns:
        The wrapped asynchronous method.
    """

    @wraps(func)
    async def wrapper(self: Any, prompt: str, *args: Any, **kwargs: Any) -> str:
        cache: Optional[SemanticCache] = getattr(self, "semantic_cache", None)

        if cache is not None:
            cached_response = await cache.get(prompt)
            if cached_response is not None:
                return cached_response

        # Cache miss or cache not configured; call the original LLM method
        response = await func(self, prompt, *args, **kwargs)

        if cache is not None:
            await cache.set(prompt, response)

        return response

    return cast(F, wrapper)
