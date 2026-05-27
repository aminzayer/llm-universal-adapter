import json
import logging
from collections import defaultdict
from typing import Any, Dict, List
from urllib.parse import urlparse

from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


class ElasticsearchDiscoveryTool:
    """
    A tool for querying an Elasticsearch backend and enforcing source diversity.
    Designed to be registered with BaseLLMAdapter.register_tool.
    """

    def __init__(self, es_client: Elasticsearch) -> None:
        """
        Initializes the Elasticsearch discovery tool.

        Args:
            es_client (Elasticsearch): An initialized Elasticsearch client.
        """
        self.es_client = es_client

    def _extract_domain(self, url: str) -> str:
        """
        Extracts the domain from a given URL.

        Args:
            url (str): The full URL string.

        Returns:
            str: The extracted domain or 'unknown' if parsing fails or URL is empty.
        """
        if not url:
            return "unknown"
        try:
            parsed_url = urlparse(url)
            return parsed_url.netloc.lower() or "unknown"
        except Exception as e:
            logger.warning(f"Failed to parse URL '{url}': {e}")
            return "unknown"

    def search(self, query: str, index_name: str, top_k: int = 5, max_count_per_domain: int = 2) -> str:
        """
        Searches Elasticsearch and returns a diverse set of results based on domain.

        Args:
            query (str): The user's search query.
            index_name (str): The Elasticsearch index to search.
            top_k (int): The maximum number of diverse results to return.
            max_count_per_domain (int): The maximum allowed documents from a single domain.

        Returns:
            str: A JSON-formatted string containing the diverse search results.
        """
        # Fetch a larger pool of results to account for potential filtering
        fetch_size = top_k * 5

        search_body: Dict[str, Any] = {"query": {"multi_match": {"query": query, "fields": ["title", "content", "description"]}}, "size": fetch_size}

        try:
            response = self.es_client.search(index=index_name, body=search_body)
        except Exception as e:
            logger.error(f"Elasticsearch query failed: {e}")
            return json.dumps({"error": f"Search failed: {str(e)}"})

        hits = response.get("hits", {}).get("hits", [])

        diverse_results: List[Dict[str, Any]] = []
        domain_counts: Dict[str, int] = defaultdict(int)

        for hit in hits:
            if len(diverse_results) >= top_k:
                break

            source = hit.get("_source", {})
            domain = self._extract_domain(source.get("url", ""))

            if domain_counts[domain] < max_count_per_domain:
                # Append document snippet to results
                diverse_results.append(source)
                domain_counts[domain] += 1

        return json.dumps({"results": diverse_results})
