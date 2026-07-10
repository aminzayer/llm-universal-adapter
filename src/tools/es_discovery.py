"""
Hybrid Search / Reranking pipeline for document discovery.

Pipeline stages:
  1. Keyword recall   — Elasticsearch ``multi_match`` (BM25).
  2. Semantic recall  — PostgreSQL ``pgvector`` cosine similarity.
  3. Cross-encoder re-rank — ``sentence-transformers`` CrossEncoder
     (soft dependency; gracefully skipped when not installed).
  4. Diversity cap    — ``max_count_per_domain`` applied on the
     final ranked list so the cap reflects relevance order, not
     insertion order.

The public :meth:`ElasticsearchDiscoveryTool.search` method is **async**
and still returns a strict JSON string so callers remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional soft-import for sentence-transformers
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import CrossEncoder as _CrossEncoder  # type: ignore[import]

    _CROSS_ENCODER_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    _CrossEncoder = None  # type: ignore[assignment]
    _CROSS_ENCODER_AVAILABLE = False


class ElasticsearchDiscoveryTool:
    """
    A tool for hybrid document retrieval with cross-encoder reranking.

    The retrieval pipeline has four stages:

    1. **Keyword recall** — BM25 ``multi_match`` query against Elasticsearch.
    2. **Semantic recall** — pgvector cosine similarity query against
       PostgreSQL (skipped when *pg_pool* is ``None``).
    3. **Cross-encoder rerank** — local ``sentence-transformers`` model
       re-scores the merged candidate pool (skipped when the library is
       not installed or *reranker_model* is ``None``).
    4. **Diversity cap** — ``max_count_per_domain`` applied *after*
       reranking so the cap reflects the relevance-ordered list.

    Designed to be registered with ``BaseLLMAdapter.register_tool``.
    """

    def __init__(
        self,
        es_client: Elasticsearch,
        *,
        pg_pool: Optional[Any] = None,
        embedding_func: Optional[Callable[[str], Any]] = None,
        reranker_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        """
        Initialises the hybrid retrieval tool.

        Args:
            es_client: An initialised :class:`elasticsearch.Elasticsearch` client.
            pg_pool: An ``asyncpg.Pool`` connected to a PostgreSQL database that
                has the ``pgvector`` extension and a ``documents`` table with at
                least ``url``, ``title``, ``content``, and ``embedding`` columns.
                When ``None`` the semantic-recall stage is skipped.
            embedding_func: An async callable ``(text: str) -> list[float]`` used
                to embed the query before the vector search.  Required when
                *pg_pool* is provided; ignored otherwise.
            reranker_model: HuggingFace model identifier for the
                ``sentence-transformers`` CrossEncoder.  Pass ``None`` to disable
                cross-encoder reranking (the merged list will be returned in
                score-descending order instead).
        """
        self.es_client = es_client
        self._pg_pool = pg_pool
        self._embedding_func = embedding_func

        # Lazily loaded; None until first use.
        self._cross_encoder: Optional[Any] = None
        self._reranker_model = reranker_model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_domain(self, url: str) -> str:
        """
        Extracts the netloc from *url*.

        Returns ``'unknown'`` when the URL is empty or cannot be parsed.
        """
        if not url:
            return "unknown"
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower() or "unknown"
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to parse URL '%s': %s", url, exc)
            return "unknown"

    @staticmethod
    def _doc_fingerprint(doc: Dict[str, Any]) -> str:
        """
        Returns a stable fingerprint for *doc* used during deduplication.

        Priority: ``url`` field → ``content`` field (first 200 chars) → full repr.
        """
        key = doc.get("url") or doc.get("content", "")[:200] or repr(doc)
        return hashlib.sha1(key.encode("utf-8", errors="replace")).hexdigest()

    def _get_cross_encoder(self) -> Optional[Any]:
        """
        Lazily initialises and caches the CrossEncoder model.

        Returns ``None`` when the library is unavailable or *reranker_model*
        is ``None``.
        """
        if self._reranker_model is None:
            return None
        if not _CROSS_ENCODER_AVAILABLE:
            logger.warning(
                "sentence-transformers not installed; cross-encoder reranking disabled. "
                "Run: pip install sentence-transformers"
            )
            return None
        if self._cross_encoder is None:
            logger.debug("Loading CrossEncoder model '%s'.", self._reranker_model)
            # pyrefly: ignore [not-callable]
            self._cross_encoder = _CrossEncoder(self._reranker_model)
        return self._cross_encoder

    # ------------------------------------------------------------------
    # Stage 1 — Elasticsearch keyword recall
    # ------------------------------------------------------------------

    def _keyword_search(self, query: str, fetch_size: int) -> List[Tuple[Dict[str, Any], float]]:
        """
        Runs a BM25 ``multi_match`` query and returns ``(doc, score)`` pairs.

        Returns an empty list on error (the error is logged at ERROR level).
        """
        body: Dict[str, Any] = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title", "content", "description"],
                }
            },
            "size": fetch_size,
        }
        try:
            response = self.es_client.search(index="_all", body=body)
        except Exception as exc:
            logger.error("Elasticsearch keyword search failed: %s", exc)
            return []

        hits = response.get("hits", {}).get("hits", [])
        results: List[Tuple[Dict[str, Any], float]] = []
        for hit in hits:
            source = hit.get("_source", {})
            score = float(hit.get("_score") or 0.0)
            results.append((source, score))
        return results

    # ------------------------------------------------------------------
    # Stage 2 — pgvector semantic recall
    # ------------------------------------------------------------------

    async def _vector_search(
        self, query: str, fetch_size: int
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Queries the ``documents`` table via pgvector cosine similarity.

        Skips gracefully when *pg_pool* or *embedding_func* is ``None``,
        or when the query itself fails.

        The expected table schema::

            CREATE TABLE documents (
                id      SERIAL PRIMARY KEY,
                url     TEXT,
                title   TEXT,
                content TEXT,
                description TEXT,
                embedding vector
            );

        Returns:
            List of ``(doc_dict, similarity_score)`` pairs ordered by
            descending similarity.
        """
        if self._pg_pool is None or self._embedding_func is None:
            return []

        try:
            query_vec = await self._embedding_func(query)
        except Exception as exc:
            logger.warning("Embedding function failed; skipping vector search: %s", exc)
            return []

        embedding_str = f"[{','.join(map(str, query_vec))}]"
        sql = """
            SELECT url, title, content, description,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM documents
            ORDER BY similarity DESC
            LIMIT $2
        """
        try:
            async with self._pg_pool.acquire() as conn:
                rows = await conn.fetch(sql, embedding_str, fetch_size)
        except Exception as exc:
            logger.warning("pgvector query failed; skipping vector search: %s", exc)
            return []

        results: List[Tuple[Dict[str, Any], float]] = []
        for row in rows:
            doc = {
                "url": row["url"],
                "title": row["title"],
                "content": row["content"],
                "description": row["description"],
            }
            results.append((doc, float(row["similarity"])))
        return results

    # ------------------------------------------------------------------
    # Stage 3 — cross-encoder reranking
    # ------------------------------------------------------------------

    def _rerank(
        self, query: str, candidates: List[Tuple[Dict[str, Any], float]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Re-scores *candidates* with the cross-encoder model.

        Falls back to sorting by the original retrieval score when the
        cross-encoder is unavailable.

        Args:
            query: The user's search query.
            candidates: ``(doc, retrieval_score)`` pairs.

        Returns:
            The same pairs, sorted by descending cross-encoder score
            (or descending retrieval score on fallback).
        """
        encoder = self._get_cross_encoder()
        if encoder is None or not candidates:
            # Fallback: sort by original retrieval score descending.
            return sorted(candidates, key=lambda x: x[1], reverse=True)

        texts = [
            doc.get("content") or doc.get("description") or doc.get("title") or ""
            for doc, _ in candidates
        ]
        pairs = [[query, t] for t in texts]

        try:
            scores: List[float] = encoder.predict(pairs).tolist()
        except Exception as exc:
            logger.warning("CrossEncoder.predict failed; falling back to retrieval scores: %s", exc)
            return sorted(candidates, key=lambda x: x[1], reverse=True)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(doc_score[0], ce_score) for doc_score, ce_score in ranked]

    # ------------------------------------------------------------------
    # Stage 4 — diversity cap
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_diversity_cap(
        ranked: List[Tuple[Dict[str, Any], float]],
        top_k: int,
        max_count_per_domain: int,
        extract_domain: Callable[[str], str],
    ) -> List[Dict[str, Any]]:
        """
        Iterates the reranked list in relevance order and enforces the
        per-domain cap, returning at most *top_k* documents.

        This is intentionally a static method so it can be unit-tested
        independently.
        """
        diverse: List[Dict[str, Any]] = []
        domain_counts: Dict[str, int] = defaultdict(int)

        for doc, _score in ranked:
            if len(diverse) >= top_k:
                break
            domain = extract_domain(doc.get("url", ""))
            if domain_counts[domain] < max_count_per_domain:
                diverse.append(doc)
                domain_counts[domain] += 1

        return diverse

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        index_name: str,
        top_k: int = 5,
        max_count_per_domain: int = 2,
    ) -> str:
        """
        Hybrid search: keyword + semantic recall → cross-encoder rerank →
        diversity-capped results as a JSON string.

        Args:
            query: The user's search query.
            index_name: Elasticsearch index to search (used for logging;
                the ES client's default index routing applies).
            top_k: Maximum number of diverse results to return.
            max_count_per_domain: Maximum documents from a single domain
                in the final result set.

        Returns:
            A JSON string ``{"results": [...]}`` on success, or
            ``{"error": "..."}`` when the keyword stage itself fails
            and no candidates are available from either source.
        """
        fetch_size = top_k * 5
        logger.debug(
            "Hybrid search | query=%r index=%s top_k=%d fetch_size=%d",
            query,
            index_name,
            top_k,
            fetch_size,
        )

        # --- Stage 1: Keyword recall ---
        kw_hits = self._keyword_search(query, fetch_size)

        # --- Stage 2: Semantic recall (async) ---
        vec_hits = await self._vector_search(query, fetch_size)

        # If both stages failed, surface a structured error.
        if not kw_hits and not vec_hits:
            logger.error("Both keyword and vector search returned no results for query: %r", query)
            return json.dumps(
                {"error": f"Search failed: no results from any retrieval stage for query '{query}'"}
            )

        # --- Merge and deduplicate (prefer higher retrieval score) ---
        seen: Dict[str, float] = {}
        all_docs: Dict[str, Dict[str, Any]] = {}

        for doc, score in kw_hits + vec_hits:
            fp = self._doc_fingerprint(doc)
            if fp not in seen or score > seen[fp]:
                seen[fp] = score
                all_docs[fp] = doc

        candidates: List[Tuple[Dict[str, Any], float]] = [
            (all_docs[fp], seen[fp]) for fp in all_docs
        ]

        # --- Stage 3: Cross-encoder rerank ---
        ranked = self._rerank(query, candidates)

        # --- Stage 4: Diversity cap (applied on ranked list) ---
        diverse = self._apply_diversity_cap(
            ranked, top_k, max_count_per_domain, self._extract_domain
        )

        return json.dumps({"results": diverse})
