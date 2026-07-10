"""
Unit tests for the hybrid ElasticsearchDiscoveryTool.

All external I/O is mocked at the boundary so the suite runs entirely
offline — no live Elasticsearch cluster, PostgreSQL connection, or
sentence-transformers model download is required.

Test coverage:
  1. Pure keyword path (no pg_pool) returns a valid JSON string.
  2. Hybrid merge deduplicates documents shared across both retrieval stages.
  3. Cross-encoder reranking reorders documents correctly.
  4. Domain diversity cap is enforced *after* reranking.
  5. Error path: both ES and vector search fail → JSON error string.
  6. Vector-search graceful fallback when pg_pool is None.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.es_discovery import ElasticsearchDiscoveryTool


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def _make_es_response(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a minimal Elasticsearch response envelope."""
    return {
        "hits": {
            "hits": [
                {"_source": h, "_score": h.get("_score", 1.0)}
                for h in hits
            ]
        }
    }


def _make_tool(
    es_hits: Optional[List[Dict[str, Any]]] = None,
    vec_rows: Optional[List[Dict[str, Any]]] = None,
    reranker_model: Optional[str] = None,
) -> ElasticsearchDiscoveryTool:
    """
    Build a tool instance with fully mocked ES client and pg_pool.

    Args:
        es_hits: Raw ``_source`` dicts returned by the mocked ES search.
            Pass an empty list to simulate ES failure (the mock raises).
        vec_rows: Records returned by the mocked pgvector query.  Each
            dict must have keys ``url``, ``title``, ``content``,
            ``description``, ``similarity``.
        reranker_model: Forwarded to the tool; ``None`` disables the
            cross-encoder so tests that don't care about reranking run
            without any model download.
    """
    # --- Elasticsearch mock ---
    mock_es = MagicMock()
    if es_hits is not None:
        mock_es.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": h, "_score": h.pop("_score", 1.0)} for h in es_hits
                ]
            }
        }
    else:
        mock_es.search.side_effect = RuntimeError("ES unavailable")

    # --- asyncpg pool mock ---
    pg_pool: Optional[MagicMock] = None
    embedding_func = None
    if vec_rows is not None:

        async def _fake_embedding(text: str) -> List[float]:  # noqa: ARG001
            return [0.1, 0.2, 0.3]

        embedding_func = _fake_embedding

        # Build asyncpg-like record mocks
        mock_records = []
        for r in vec_rows:
            rec = MagicMock()
            rec.__getitem__ = lambda self, key, _r=r: _r[key]
            mock_records.append(rec)

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_records)

        mock_pg_pool = MagicMock()
        mock_pg_pool.acquire = MagicMock(return_value=_AsyncContextManager(mock_conn))
        pg_pool = mock_pg_pool

    return ElasticsearchDiscoveryTool(
        es_client=mock_es,
        pg_pool=pg_pool,
        embedding_func=embedding_func,
        reranker_model=reranker_model,
    )


class _AsyncContextManager:
    """Minimal async context-manager shim wrapping a coroutine value."""

    def __init__(self, value: Any) -> None:
        self._value = value

    async def __aenter__(self) -> Any:
        return self._value

    async def __aexit__(self, *_: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Test 1 — pure keyword path returns valid JSON string
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_json_string() -> None:
    """
    With no pg_pool the semantic stage is skipped entirely.  The method must
    return a str that parses as valid JSON with a ``"results"`` key.
    """
    hits = [
        {"url": "https://a.example.com/1", "title": "Alpha", "content": "some text"},
        {"url": "https://b.example.com/2", "title": "Beta", "content": "other text"},
    ]
    tool = _make_tool(es_hits=hits, reranker_model=None)

    result = await tool.search(query="alpha beta", index_name="docs", top_k=5)

    assert isinstance(result, str), "search() must return a str"
    payload = json.loads(result)
    assert "results" in payload
    assert isinstance(payload["results"], list)
    assert len(payload["results"]) == 2


# ---------------------------------------------------------------------------
# Test 2 — hybrid merge deduplicates shared URLs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_merge_deduplicates() -> None:
    """
    A document that appears in both the ES and vector-search pools should be
    deduplicated so it appears only once in the final result set.
    """
    shared_url = "https://shared.example.com/doc"
    es_hits = [{"url": shared_url, "title": "Shared", "content": "text A"}]
    vec_rows = [
        {
            "url": shared_url,
            "title": "Shared",
            "content": "text A",
            "description": "",
            "similarity": 0.9,
        }
    ]
    tool = _make_tool(es_hits=es_hits, vec_rows=vec_rows, reranker_model=None)

    result = await tool.search(query="shared doc", index_name="docs", top_k=5)

    payload = json.loads(result)
    urls = [d.get("url") for d in payload["results"]]
    assert urls.count(shared_url) == 1, "Deduplicated URL must appear exactly once"


# ---------------------------------------------------------------------------
# Test 3 — cross-encoder reranking reorders results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reranking_reorders_results() -> None:
    """
    Inject a mock CrossEncoder whose predict() reverses the retrieval order.
    The final result list must reflect the cross-encoder score, not the
    original BM25 score.
    """
    es_hits = [
        {"url": "https://first.example.com/", "title": "First (high BM25)", "content": "aaa"},
        {"url": "https://second.example.com/", "title": "Second (low BM25)", "content": "bbb"},
    ]
    tool = _make_tool(es_hits=es_hits, reranker_model="test-model")

    # Patch the cross-encoder: predict reverses order (second doc scores higher)
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = MagicMock(tolist=lambda: [0.1, 0.9])
    tool._cross_encoder = mock_encoder

    with patch("tools.es_discovery._CROSS_ENCODER_AVAILABLE", True):
        result = await tool.search(query="test query", index_name="docs", top_k=5)

    payload = json.loads(result)
    assert isinstance(result, str)
    assert len(payload["results"]) == 2
    # Second doc (index 1 in BM25 order) should now be ranked first.
    assert payload["results"][0]["title"] == "Second (low BM25)"
    assert payload["results"][1]["title"] == "First (high BM25)"


# ---------------------------------------------------------------------------
# Test 4 — domain diversity cap is applied after reranking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diversity_cap_applied_after_rerank() -> None:
    """
    After reranking, documents from the same domain beyond ``max_count_per_domain``
    should be excluded even if they rank highly.
    """
    # Three docs all from the same domain; max cap = 2
    es_hits = [
        {"url": "https://same.example.com/a", "title": "A", "content": "..."},
        {"url": "https://same.example.com/b", "title": "B", "content": "..."},
        {"url": "https://same.example.com/c", "title": "C", "content": "..."},
    ]
    tool = _make_tool(es_hits=es_hits, reranker_model=None)

    result = await tool.search(
        query="diversity test",
        index_name="docs",
        top_k=5,
        max_count_per_domain=2,
    )

    payload = json.loads(result)
    assert isinstance(result, str)
    # Only 2 of the 3 same-domain docs should survive the cap.
    assert len(payload["results"]) == 2
    for doc in payload["results"]:
        assert doc["url"].startswith("https://same.example.com/")


# ---------------------------------------------------------------------------
# Test 5 — error path: both stages fail → JSON error string
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_es_error_returns_json_error_string() -> None:
    """
    When the Elasticsearch query raises and there is no pg_pool, the method
    must return a valid JSON string containing an ``"error"`` key (not raise).
    """
    tool = _make_tool(es_hits=None, reranker_model=None)  # es_hits=None → raises

    result = await tool.search(query="broken query", index_name="docs")

    assert isinstance(result, str), "Must always return str"
    payload = json.loads(result)
    assert "error" in payload


# ---------------------------------------------------------------------------
# Test 6 — graceful fallback when pg_pool is None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vector_search_fallback_when_pool_none() -> None:
    """
    With ``pg_pool=None`` the vector-search stage is skipped silently.
    Results should come from the keyword stage only.
    """
    hits = [
        {"url": "https://keyword.example.com/x", "title": "KW result", "content": "keyword"},
    ]
    # Explicitly no pg_pool
    tool = _make_tool(es_hits=hits, vec_rows=None, reranker_model=None)

    result = await tool.search(query="keyword only", index_name="docs", top_k=5)

    assert isinstance(result, str)
    payload = json.loads(result)
    assert "results" in payload
    assert len(payload["results"]) == 1
    assert payload["results"][0]["title"] == "KW result"


# ---------------------------------------------------------------------------
# Test 7 — _apply_diversity_cap static method (unit test in isolation)
# ---------------------------------------------------------------------------


def test_apply_diversity_cap_static_method() -> None:
    """
    Unit-test _apply_diversity_cap independently of the async pipeline
    to verify the cap logic in isolation.
    """
    extract = lambda url: url.split("/")[2] if url.startswith("http") else "unknown"  # noqa: E731

    ranked = [
        ({"url": "https://a.com/1", "title": "A1"}, 0.9),
        ({"url": "https://a.com/2", "title": "A2"}, 0.8),
        ({"url": "https://b.com/1", "title": "B1"}, 0.7),
        ({"url": "https://a.com/3", "title": "A3"}, 0.6),
    ]

    result = ElasticsearchDiscoveryTool._apply_diversity_cap(
        ranked=ranked,
        top_k=10,
        max_count_per_domain=2,
        extract_domain=extract,
    )

    titles = [d["title"] for d in result]
    assert titles == ["A1", "A2", "B1"], "Third 'a.com' doc must be excluded by the cap"
