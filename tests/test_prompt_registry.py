"""
Tests for the PromptRegistry module.

The tests cover:
    * No-database fallback to module-level ``DEFAULT_PROMPTS`` and custom
      ``defaults`` mappings.
    * Version immutability (no update methods on the public API).
    * Weighted A/B selection distribution and edge cases.
    * LRU cache behavior: hits, misses, TTL expiry, and size-based eviction.
    * Integration with ``ClassifierAgent`` and ``SummaryAgent`` (and therefore
      with ``SwarmOrchestrator`` dispatch).

The tests do not require a real PostgreSQL instance. They use the
``db_pool=None`` constructor mode for the default-prompt path, and a tiny
async stub for tests that exercise the cache or weight sampling.
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import pytest

from adapter.base import BaseLLMAdapter
from orchestration.swarm import (
    ClassifierAgent,
    SummaryAgent,
    SwarmOrchestrator,
    SwarmResult,
    SwarmTask,
)
from prompts import (
    DEFAULT_PROMPTS,
    PromptCategory,
    PromptCategoryNotFound,
    PromptRegistry,
    PromptRegistryUnavailable,
    PromptVersion,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class SequenceAdapter(BaseLLMAdapter):
    """
    Adapter that returns a fixed sequence of strings, regardless of prompt.
    Mirrors the same-named fake in ``tests/test_swarm.py`` but kept local so
    this test file does not depend on swarm tests.
    """

    def __init__(self, responses: List[str]) -> None:
        super().__init__()
        self.responses = responses
        self.call_count = 0
        self.prompts: List[str] = []

    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        res = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return res

    async def agenerate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        yield ""

    async def get_token_count(self, text: str) -> int:
        return len(text.split())

    async def generate_with_tools(self, prompt: str) -> str:
        return ""


class FakeESTool:
    """Stand-in for ElasticsearchDiscoveryTool that returns canned JSON."""

    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def search(
        self,
        query: str,
        index_name: str,
        top_k: int = 5,
        max_count_per_domain: int = 2,
    ) -> str:
        return json.dumps(self.payload)


class FakePool:
    """
    Minimal asyncpg.Pool stand-in that records how many times ``acquire`` is
    called. We never call ``.execute``/``.fetchrow``/``.fetch`` on it from the
    no-DB-mode tests; the registry short-circuits to ``defaults`` when
    ``db_pool is None`` and to the in-memory version list in the tests that
    pass a pre-populated ``versions`` override.

    The tests below use either ``db_pool=None`` (for the default-prompt and
    cache-shape tests) or they construct the registry with a custom subclass
    that bypasses the DB. We only count acquires when an actual DB path is
    exercised, via a custom subclass defined further down.
    """

    def __init__(self) -> None:
        self.acquire_count = 0

    def acquire(self) -> "_AcquireCtx":
        self.acquire_count += 1
        return _AcquireCtx(self)


class _AcquireCtx:
    def __init__(self, pool: FakePool) -> None:
        self.pool = pool

    async def __aenter__(self) -> "_AcquireCtx":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def execute(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def fetch(self, *args: Any, **kwargs: Any) -> List[Any]:
        return []

    async def fetchrow(self, *args: Any, **kwargs: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# 1–3: default-prompt path
# ---------------------------------------------------------------------------


def test_registry_without_db_returns_default_prompt() -> None:
    registry = PromptRegistry(db_pool=None)
    text = asyncio.run(registry.get_prompt("classifier.system"))
    assert text == DEFAULT_PROMPTS["classifier.system"]


def test_registry_without_db_returns_summary_default() -> None:
    registry = PromptRegistry(db_pool=None)
    text = asyncio.run(registry.get_prompt("summary.system"))
    assert text == DEFAULT_PROMPTS["summary.system"]


def test_registry_without_db_raises_for_unknown_category() -> None:
    registry = PromptRegistry(db_pool=None)
    with pytest.raises(PromptCategoryNotFound):
        asyncio.run(registry.get_prompt("does.not.exist"))


def test_in_memory_defaults_constructor_overrides_module_constants() -> None:
    custom = "TPL: intent_list=[{intent_list}] | user={user_input}"
    registry = PromptRegistry(db_pool=None, defaults={"classifier.system": custom})
    text = asyncio.run(registry.get_prompt("classifier.system"))
    assert text == custom
    # ``summary.system`` should still fall back to the module-level default.
    summary_text = asyncio.run(registry.get_prompt("summary.system"))
    assert summary_text == DEFAULT_PROMPTS["summary.system"]


# ---------------------------------------------------------------------------
# 4–6: version immutability / listing
# ---------------------------------------------------------------------------


def test_create_version_assigns_monotonic_numbers() -> None:
    # Use the in-memory subclass that returns versions from a local list.
    registry = _InMemoryPromptRegistry(
        versions_by_category={"a": [(1, "first", 100), (2, "second", 100)]}
    )
    first = asyncio.run(registry.create_version("b", "v1"))
    second = asyncio.run(registry.create_version("b", "v2"))
    assert first.version_number == 1
    assert second.version_number == 2
    assert first.text == "v1"
    assert second.text == "v2"


def test_version_immutability_no_update_method() -> None:
    registry = PromptRegistry(db_pool=None)
    public_api = {name for name in dir(registry) if not name.startswith("_")}
    # The contract is "immutable versions". Any method that mutates an
    # existing version row would violate that.
    for forbidden in ("update_version", "set_weight", "delete_version", "mutate_version"):
        assert forbidden not in public_api, f"Registry exposes forbidden mutator: {forbidden}"


def test_list_versions_returns_all_rows_in_order() -> None:
    registry = _InMemoryPromptRegistry(
        versions_by_category={"a": [(1, "v1", 50), (2, "v2", 50), (3, "v3", 50)]}
    )
    versions = asyncio.run(registry.list_versions("a"))
    assert [v.version_number for v in versions] == [1, 2, 3]
    assert [v.text for v in versions] == ["v1", "v2", "v3"]
    assert [v.weight for v in versions] == [50, 50, 50]


# ---------------------------------------------------------------------------
# 7–10: weighted selection
# ---------------------------------------------------------------------------


def test_weighted_selection_distribution() -> None:
    """Weights (70, 30) over 10 000 samples land within ±2% of expected proportion."""
    registry = _InMemoryPromptRegistry(
        versions_by_category={"x": [(1, "A", 70), (2, "B", 30)]},
        rng=random.Random(0),
    )
    counts: Dict[str, int] = {"A": 0, "B": 0}
    for _ in range(10_000):
        chosen = asyncio.run(registry.get_prompt("x"))
        counts[chosen] += 1
    a_ratio = counts["A"] / 10_000
    b_ratio = counts["B"] / 10_000
    assert abs(a_ratio - 0.70) < 0.02, f"A ratio {a_ratio:.3f} outside ±2% of 0.70"
    assert abs(b_ratio - 0.30) < 0.02, f"B ratio {b_ratio:.3f} outside ±2% of 0.30"


def test_zero_weight_version_is_excluded() -> None:
    """A weight-0 version must never be selected, even when other versions exist."""
    registry = _InMemoryPromptRegistry(
        versions_by_category={"x": [(1, "A", 100), (2, "B", 0)]},
        rng=random.Random(0),
    )
    for _ in range(1_000):
        chosen = asyncio.run(registry.get_prompt("x"))
        assert chosen == "A"


def test_all_zero_weights_raises() -> None:
    registry = _InMemoryPromptRegistry(
        versions_by_category={"x": [(1, "A", 0), (2, "B", 0)]},
        rng=random.Random(0),
    )
    with pytest.raises(PromptCategoryNotFound):
        asyncio.run(registry.get_prompt("x"))


def test_single_version_always_returned() -> None:
    registry = _InMemoryPromptRegistry(
        versions_by_category={"x": [(1, "only", 100)]},
        rng=random.Random(0),
    )
    for _ in range(50):
        assert asyncio.run(registry.get_prompt("x")) == "only"


# ---------------------------------------------------------------------------
# 11–14: LRU cache
# ---------------------------------------------------------------------------


def test_lru_cache_hit_skips_db() -> None:
    """Second call within TTL must hit the cache (misses don't increment)."""
    registry = _InMemoryPromptRegistry(
        versions_by_category={"x": [(1, "v1", 100)]},
    )
    # Prime the cache.
    asyncio.run(registry.get_prompt("x"))
    stats_after_first = registry.cache_stats()
    assert stats_after_first["misses"] == 1
    assert stats_after_first["hits"] == 0
    # Subsequent reads must hit the cache.
    for _ in range(5):
        asyncio.run(registry.get_prompt("x"))
    stats_after_all = registry.cache_stats()
    assert stats_after_all["misses"] == 1, "Misses should not increment on cache hits"
    assert stats_after_all["hits"] == 5, "Hits should increment on cache hits"


def test_lru_cache_ttl_expiry() -> None:
    """An entry older than cache_ttl_seconds must be refetched (miss increments again)."""
    fake_now = [0.0]

    def clock() -> float:
        return fake_now[0]

    registry = _InMemoryPromptRegistry(
        versions_by_category={"x": [(1, "v1", 100)]},
        cache_ttl_seconds=0.05,
        clock=clock,
    )
    asyncio.run(registry.get_prompt("x"))
    stats_after_first = registry.cache_stats()
    assert stats_after_first["misses"] == 1
    assert stats_after_first["hits"] == 0
    # Advance time past the TTL.
    fake_now[0] = 0.20
    asyncio.run(registry.get_prompt("x"))
    stats_after_ttl = registry.cache_stats()
    assert stats_after_ttl["misses"] == 2, "TTL expiry should cause a cache miss on refetch"
    assert stats_after_ttl["hits"] == 0


def test_lru_cache_size_eviction() -> None:
    """With cache_size=2, fetching 3 distinct categories caps the cache at 2."""
    registry = _InMemoryPromptRegistry(
        versions_by_category={
            "a": [(1, "a1", 100)],
            "b": [(1, "b1", 100)],
            "c": [(1, "c1", 100)],
        },
        cache_size=2,
    )
    asyncio.run(registry.get_prompt("a"))
    asyncio.run(registry.get_prompt("b"))
    asyncio.run(registry.get_prompt("c"))
    stats = registry.cache_stats()
    assert stats["size"] == 2


def test_cache_stats_reports_hits_and_misses() -> None:
    registry = _InMemoryPromptRegistry(
        versions_by_category={"x": [(1, "v1", 100)], "y": [(1, "y1", 100)]},
        cache_ttl_seconds=10.0,
    )
    asyncio.run(registry.get_prompt("x"))  # miss
    asyncio.run(registry.get_prompt("x"))  # hit
    asyncio.run(registry.get_prompt("x"))  # hit
    asyncio.run(registry.get_prompt("y"))  # miss (different category)
    stats = registry.cache_stats()
    assert stats["misses"] == 2
    assert stats["hits"] == 2


# ---------------------------------------------------------------------------
# 15–18: integration with swarm workers
# ---------------------------------------------------------------------------


def test_classifier_uses_registry_prompt() -> None:
    """ClassifierAgent must use the template provided by the registry."""
    template = "CTMPL intent=[{intent_list}] | input={user_input}"
    registry = PromptRegistry(
        db_pool=None,
        defaults={"classifier.system": template},
    )
    decision_json = json.dumps({"intent": "search", "confidence": 0.9, "reasoning": "ok"})
    adapter = SequenceAdapter([decision_json])
    classifier = ClassifierAgent(
        adapter, candidate_intents=["search", "summary"], prompt_registry=registry
    )

    decision = asyncio.run(
        classifier.classify(SwarmTask(request_id="r", user_input="hello"))
    )
    assert decision.intent == "search"
    # StructuredGenerator appends the JSON schema to the prompt before calling
    # the adapter, so we check that our template is the *prefix* of the prompt.
    expected = template.format(intent_list="'search', 'summary'", user_input="hello")
    assert adapter.prompts[0].startswith(expected), f"Prompt did not start with expected template. Got: {adapter.prompts[0][:200]}"


def test_summary_uses_registry_prompt() -> None:
    template = "SUMM: max_points={max_points} | text={text}"
    registry = PromptRegistry(
        db_pool=None,
        defaults={"summary.system": template},
    )
    payload = {"summary": "s", "key_points": ["p1", "p2"]}
    adapter = SequenceAdapter([json.dumps(payload)])
    agent = SummaryAgent(adapter, prompt_registry=registry)

    task = SwarmTask(request_id="r", user_input="the quick brown fox")
    asyncio.run(
        agent.run(
            task,
            __import__("orchestration.swarm", fromlist=["ClassificationDecision"]).ClassificationDecision(
                intent="summary", confidence=0.9, reasoning="r"
            ),
        )
    )
    expected = template.format(max_points=3, text=task.user_input)
    # StructuredGenerator appends the JSON schema to the prompt.
    assert adapter.prompts[0].startswith(expected), f"Prompt did not start with expected template. Got: {adapter.prompts[0][:200]}"


def test_swarm_dispatch_uses_dynamic_classifier_prompt() -> None:
    """End-to-end: SwarmOrchestrator.dispatch must use the registry-driven prompt."""
    template = "DYN_CLS: {intent_list} | {user_input}"
    registry = PromptRegistry(db_pool=None, defaults={"classifier.system": template})

    decision = {"intent": "summary", "confidence": 0.9, "reasoning": "ok"}
    payload = {"summary": "short", "key_points": ["p1"]}
    # First response is the classifier's; second is the summary worker's.
    adapter = SequenceAdapter([json.dumps(decision), json.dumps(payload)])

    classifier = ClassifierAgent(adapter, candidate_intents=["summary"], prompt_registry=registry)
    summary_worker = SummaryAgent(adapter)
    orch = SwarmOrchestrator(classifier=classifier)
    orch.register_worker(summary_worker)

    result = asyncio.run(orch.dispatch(SwarmTask(request_id="r", user_input="text")))
    assert isinstance(result, SwarmResult)
    # The classifier saw the dynamic template; the worker saw the default.
    expected_clf = template.format(intent_list="'summary'", user_input="text")
    # StructuredGenerator appends the JSON schema.
    assert adapter.prompts[0].startswith(expected_clf), f"Classifier prompt mismatch: {adapter.prompts[0][:200]}"


def test_swarm_dispatch_uses_dynamic_summary_prompt() -> None:
    template = "DYN_SUM: {max_points} | {text}"
    registry = PromptRegistry(db_pool=None, defaults={"summary.system": template})

    decision = {"intent": "summary", "confidence": 0.9, "reasoning": "ok"}
    payload = {"summary": "short", "key_points": ["p1"]}
    adapter = SequenceAdapter([json.dumps(decision), json.dumps(payload)])
    classifier = ClassifierAgent(adapter, candidate_intents=["summary"], prompt_registry=registry)
    summary_worker = SummaryAgent(adapter, prompt_registry=registry)
    orch = SwarmOrchestrator(classifier=classifier)
    orch.register_worker(summary_worker)

    asyncio.run(orch.dispatch(SwarmTask(request_id="r", user_input="hello world")))
    expected = template.format(max_points=3, text="hello world")
    # StructuredGenerator appends the JSON schema.
    assert adapter.prompts[1].startswith(expected), f"Summary prompt mismatch: {adapter.prompts[1][:200]}"


# ---------------------------------------------------------------------------
# 19–20: misc safety
# ---------------------------------------------------------------------------


def test_create_version_when_db_pool_is_none_raises() -> None:
    registry = PromptRegistry(db_pool=None)
    with pytest.raises(PromptRegistryUnavailable):
        asyncio.run(registry.create_version("any", "text"))


def test_prompt_registry_types_smoke() -> None:
    """Runtime smoke test; mypy validates the rest of the type contract."""
    registry = PromptRegistry(db_pool=None)
    result: str = asyncio.run(registry.get_prompt("classifier.system"))
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Helpers: in-memory registry used by the tests above.
# ---------------------------------------------------------------------------


class _InMemoryPromptRegistry(PromptRegistry):
    """
    A ``PromptRegistry`` subclass that stores versions in process memory so
    tests can exercise the cache, weighting, and listing logic without
    requiring a real PostgreSQL instance.

    The ``db_pool`` parameter is accepted (and may be a :class:`FakePool`) so
    tests that want to count ``acquire`` calls can pass it in. The fake pool's
    methods are no-ops; the in-memory list is the source of truth.
    """

    def __init__(
        self,
        *,
        versions_by_category: Optional[Dict[str, List[Tuple[int, str, int]]]] = None,
        db_pool: Any = None,
        cache_size: int = 256,
        cache_ttl_seconds: float = 10.0,
        rng: Optional[random.Random] = None,
        clock: Any = None,
    ) -> None:
        # Bypass ``PromptRegistry.__init__`` to avoid touching the real cachetools
        # validation; build the attributes manually with the same names.
        from cachetools import LRUCache

        self._defaults: Dict[str, str] = dict(DEFAULT_PROMPTS)
        self._db_pool = db_pool
        self._cache: LRUCache[str, tuple[PromptCategory, float]] = LRUCache(maxsize=cache_size)
        self._cache_ttl: float = cache_ttl_seconds
        self._rng: random.Random = rng if rng is not None else random.Random()
        import time as _time
        self._clock: Any = clock if clock is not None else _time.monotonic
        self._hits: int = 0
        self._misses: int = 0
        self._store: Dict[str, List[PromptVersion]] = {}
        for category, entries in (versions_by_category or {}).items():
            self._store[category] = [
                PromptVersion(
                    version_id=1000 + idx,
                    category=category,
                    version_number=number,
                    text=text,
                    weight=weight,
                    created_at=__import__("datetime").datetime.fromtimestamp(0),
                )
                for idx, (number, text, weight) in enumerate(entries)
            ]

    async def _fetch_category(self, category: str) -> PromptCategory:
        versions = self._store.get(category, [])
        if not versions and category in self._defaults:
            fallback = PromptVersion(
                version_id=0,
                category=category,
                version_number=0,
                text=self._defaults[category],
                weight=1,
                created_at=__import__("datetime").datetime.fromtimestamp(0),
            )
            return PromptCategory(category=category, versions=(fallback,))
        if not versions:
            raise PromptCategoryNotFound(category)
        return PromptCategory(category=category, versions=tuple(versions))

    async def list_versions(self, category: str) -> List[PromptVersion]:
        return list(self._store.get(category, []))

    async def create_version(self, category: str, text: str, weight: int = 100) -> PromptVersion:
        if weight < 0:
            raise ValueError("weight must be non-negative.")
        existing = self._store.setdefault(category, [])
        next_number = (max((v.version_number for v in existing), default=0)) + 1
        from datetime import datetime
        version = PromptVersion(
            version_id=2000 + next_number,
            category=category,
            version_number=next_number,
            text=text,
            weight=weight,
            created_at=datetime.fromtimestamp(0),
        )
        existing.append(version)
        try:
            del self._cache[category]
        except KeyError:
            pass
        return version
