"""
Versioned prompt registry backed by PostgreSQL.

The :class:`PromptRegistry` stores prompt versions per category and supports
weighted A/B sampling on the read path. Versions are append-only: every save
inserts a new row with a monotonically increasing ``version_number`` and never
mutates an existing row. A/B allocation is driven by the ``weight`` column,
which is read at cache-fill time and sampled per dispatch via
``random.choices``.

When no database pool is supplied (or the category is empty in the database),
the registry falls back to a ``defaults`` mapping supplied at construction time
— or, if none is supplied, to the module-level :data:`DEFAULT_PROMPTS`. This
keeps the registry fully functional in local development and unit tests.

A bounded LRU cache with a single fixed TTL shields the database from the
hot read path. Cache entries store the entire :class:`PromptCategory` plus a
monotonic timestamp; entries are dropped on read if they exceed the TTL.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Optional

import asyncpg  # type: ignore
from cachetools import LRUCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default prompt templates
# ---------------------------------------------------------------------------


# Module-level fallback prompts. These mirror the literal text that was
# previously hardcoded in ``SwarmOrchestrator`` workers so behavior is
# byte-identical when no database is configured. The strings are templates
# using ``str.format`` placeholders — workers call ``.format(...)`` after
# fetching the template from the registry.
DEFAULT_PROMPTS: dict[str, str] = {
    "classifier.system": (
        "You are a routing classifier for a multi-agent system. "
        "Classify the user's request into exactly one of the following intents: "
        "[{intent_list}].\n\n"
        "User request:\n{user_input}"
    ),
    "summary.system": (
        "Summarize the following text. Return a concise 'summary' string and "
        "up to {max_points} bullet-style 'key_points' that capture "
        "the most important facts.\n\n"
        "Text:\n{text}"
    ),
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PromptVersion:
    """A single immutable version of a prompt."""

    version_id: int
    category: str
    version_number: int
    text: str
    weight: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class PromptCategory:
    """A category and all of its known versions."""

    category: str
    versions: tuple[PromptVersion, ...]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PromptRegistryError(Exception):
    """Base class for PromptRegistry errors."""


class PromptRegistryUnavailable(PromptRegistryError):
    """Raised when a write is attempted on a registry with no database pool."""


class PromptCategoryNotFound(PromptRegistryError, KeyError):
    """
    Raised when a category cannot be resolved from either the database or the
    ``defaults`` mapping. Subclassing ``KeyError`` keeps ``except KeyError:``
    patterns functional.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_version(row: asyncpg.Record) -> PromptVersion:
    """Convert an asyncpg row from ``prompt_versions`` into a ``PromptVersion``."""
    return PromptVersion(
        version_id=int(row["id"]),
        category=str(row["category"]),
        version_number=int(row["version_number"]),
        text=str(row["text"]),
        weight=int(row["weight"]),
        created_at=row["created_at"],
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PromptRegistry:
    """
    Versioned prompt registry with weighted A/B selection and an LRU cache.

    Reads go through the LRU cache first. On a miss (or expiry), the registry
    either queries the database or, if no pool is configured, consults the
    ``defaults`` mapping. Selection among versions uses ``random.choices`` with
    the cached weights.
    """

    def __init__(
        self,
        db_pool: Optional[asyncpg.Pool],
        *,
        cache_size: int = 256,
        cache_ttl_seconds: float = 10.0,
        defaults: Optional[Mapping[str, str]] = None,
        rng: Optional[random.Random] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if cache_size <= 0:
            raise ValueError("cache_size must be positive.")
        if cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be non-negative.")

        # Effective defaults: caller-supplied mapping wins over the module
        # constant. We copy to insulate from later mutation.
        if defaults is not None:
            merged: dict[str, str] = dict(DEFAULT_PROMPTS)
            merged.update(defaults)
            self._defaults: dict[str, str] = merged
        else:
            self._defaults = dict(DEFAULT_PROMPTS)

        self._db_pool: Optional[asyncpg.Pool] = db_pool
        self._cache: LRUCache[str, tuple[PromptCategory, float]] = LRUCache(maxsize=cache_size)
        self._cache_ttl: float = cache_ttl_seconds
        self._rng: random.Random = rng if rng is not None else random.Random()
        self._clock: Callable[[], float] = clock
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Idempotently create the ``prompt_versions`` table and its index. Mirrors
        the inline-migration pattern used by :class:`SemanticCache`. Safe to
        call multiple times.
        """
        if self._db_pool is None:
            logger.info("PromptRegistry.initialize skipped: no database pool configured.")
            return

        async with self._db_pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_versions (
                    id              BIGSERIAL PRIMARY KEY,
                    category        TEXT        NOT NULL,
                    version_number  INTEGER     NOT NULL,
                    text            TEXT        NOT NULL,
                    weight          INTEGER     NOT NULL DEFAULT 0 CHECK (weight >= 0),
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (category, version_number)
                );
                CREATE INDEX IF NOT EXISTS idx_prompt_versions_category
                    ON prompt_versions (category);
                """
            )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def create_version(self, category: str, text: str, weight: int = 100) -> PromptVersion:
        """
        Insert a new immutable version of ``category``. ``version_number`` is
        assigned as ``MAX(version_number) + 1`` within the category. Returns the
        newly inserted :class:`PromptVersion`.
        """
        if weight < 0:
            raise ValueError("weight must be non-negative.")
        if not category:
            raise ValueError("category must be a non-empty string.")
        if not text:
            raise ValueError("text must be a non-empty string.")

        if self._db_pool is None:
            raise PromptRegistryUnavailable(
                "Cannot create_version: PromptRegistry has no database pool."
            )

        query = """
            INSERT INTO prompt_versions (category, version_number, text, weight)
            VALUES (
                $1,
                COALESCE(
                    (SELECT MAX(version_number) + 1 FROM prompt_versions WHERE category = $1),
                    1
                ),
                $2,
                $3
            )
            RETURNING id, category, version_number, text, weight, created_at
        """
        async with self._db_pool.acquire() as conn:
            row = await conn.fetchrow(query, category, text, weight)

        assert row is not None  # RETURNING is guaranteed by INSERT semantics
        version = _row_to_version(row)

        # Invalidate the cache entry for this category so the next read
        # observes the new version immediately.
        try:
            del self._cache[category]
        except KeyError:
            pass

        return version

    async def list_versions(self, category: str) -> list[PromptVersion]:
        """Return all versions for ``category`` ordered by version_number ASC."""
        if self._db_pool is None:
            # Surface default behavior: no DB rows. Tests that exercise the
            # CRUD path provide a real pool; this branch is the read-only mode.
            return []

        query = """
            SELECT id, category, version_number, text, weight, created_at
            FROM prompt_versions
            WHERE category = $1
            ORDER BY version_number ASC
        """
        async with self._db_pool.acquire() as conn:
            rows = await conn.fetch(query, category)

        return [_row_to_version(row) for row in rows]

    # ------------------------------------------------------------------
    # Read path (cache-aware)
    # ------------------------------------------------------------------

    async def _fetch_category(self, category: str) -> PromptCategory:
        """
        Bypass the cache and resolve ``category`` from the database or the
        ``defaults`` mapping. Raises :class:`PromptCategoryNotFound` if neither
        source has the category.
        """
        versions: list[PromptVersion]
        if self._db_pool is not None:
            versions = await self.list_versions(category)
        else:
            versions = []

        if not versions and category in self._defaults:
            # No DB rows but a fallback template is configured. Wrap it as a
            # single-version, weight-100 category so A/B sampling still works
            # uniformly downstream.
            fallback = PromptVersion(
                version_id=0,
                category=category,
                version_number=0,
                text=self._defaults[category],
                weight=1,
                created_at=datetime.fromtimestamp(0),
            )
            return PromptCategory(category=category, versions=(fallback,))

        if not versions:
            raise PromptCategoryNotFound(category)

        return PromptCategory(category=category, versions=tuple(versions))

    async def get_category(self, category: str) -> PromptCategory:
        """
        Return a :class:`PromptCategory` for ``category``, consulting the LRU
        cache first. Cache entries older than ``cache_ttl_seconds`` are
        discarded and the category is re-fetched.
        """
        now = self._clock()
        cached = self._cache.get(category)
        if cached is not None:
            value, fetched_at = cached
            if now - fetched_at < self._cache_ttl:
                self._hits += 1
                return value
            # Expired — drop and refetch.
            try:
                del self._cache[category]
            except KeyError:
                pass

        self._misses += 1
        value = await self._fetch_category(category)
        self._cache[category] = (value, now)
        return value

    async def get_prompt(self, category: str) -> str:
        """
        Return a single prompt string for ``category``, sampled from the
        version set via weighted random selection.

        Edge cases:
            * Single version: always returned.
            * All weights zero: raises :class:`PromptCategoryNotFound`.
            * Missing category (no DB, no defaults): raises
              :class:`PromptCategoryNotFound`.
        """
        category_obj = await self.get_category(category)
        versions = category_obj.versions

        if len(versions) == 1:
            return versions[0].text

        weights = [v.weight for v in versions]
        if sum(weights) <= 0:
            raise PromptCategoryNotFound(
                f"Category '{category}' has no active versions (all weights are 0)."
            )

        texts = [v.text for v in versions]
        # ``random.choices`` is stdlib-typed via typeshed and returns a list.
        # We pick exactly one draw per call.
        chosen = self._rng.choices(texts, weights=weights, k=1)
        return chosen[0]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def cache_stats(self) -> dict[str, int]:
        """Return a snapshot of cache counters (useful for tests and metrics)."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
        }

    def invalidate_cache(self) -> None:
        """Drop all cached categories. Primarily for tests and admin tooling."""
        self._cache.clear()


__all__ = [
    "DEFAULT_PROMPTS",
    "PromptCategory",
    "PromptCategoryNotFound",
    "PromptRegistry",
    "PromptRegistryError",
    "PromptRegistryUnavailable",
    "PromptVersion",
]


# Silence "unused import" warnings for symbols that exist for type-checking
# only in some environments. Keeps the import block clean for mypy.
_ = Any  # pragma: no cover
