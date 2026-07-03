"""
Asynchronous BFS web scraper with LLM-based content evaluation.

The ``AgenticScraper.crawl()`` method now accepts a ``publish_fn`` callback
so it can emit structured ``TaskEvent`` progress updates to the AgentWorker,
which forwards them to WebSocket clients via Redis Pub/Sub. This decouples
the scraper from any transport layer (the callback can be a Redis publisher,
an in-memory queue, or a no-op for tests).
"""

from __future__ import annotations

import logging
from typing import Callable, Coroutine, Optional, Set

import aiohttp
from bs4 import BeautifulSoup

from adapter.factory import LLMAdapterFactory

logger = logging.getLogger(__name__)

# Type alias for the async progress callback injected by the worker.
PublishFn = Optional[Callable[..., Coroutine]]


class AgenticScraper:
    """
    Asynchronous BFS web scraper that uses an LLM to evaluate content relevance.

    Progress is reported via an optional ``publish_fn`` callback instead of
    ``print()`` statements, enabling integration with the event-driven worker.
    """

    def __init__(self, provider: str, max_depth: int = 2) -> None:
        """
        Initializes the AgenticScraper.

        Args:
            provider: LLM provider name used for content evaluation.
            max_depth: Maximum BFS crawl depth (0 = seed URL only).
        """
        self.max_depth = max_depth
        self.visited_urls: Set[str] = set()
        # Initialize the LLM adapter dynamically via the factory.
        self.llm = LLMAdapterFactory.create_adapter(provider)

    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> str:
        """
        Fetches the HTML content of a given URL.

        Returns an empty string on any network or HTTP error so the caller
        can treat it as a non-result without raising.

        Args:
            session: Shared aiohttp session for connection pooling.
            url: The URL to fetch.
        """
        timeout = aiohttp.ClientTimeout(total=10)
        try:
            async with session.get(url, timeout=timeout) as response:
                return await response.text()
        except Exception as exc:
            logger.warning("Error fetching %s: %s", url, exc)
            return ""

    async def evaluate_content(self, text: str) -> bool:
        """
        Uses the LLM to determine if the scraped content is highly relevant.

        Args:
            text: Plain-text content extracted from the page.

        Returns:
            True if the LLM classifies the content as relevant, False otherwise.
        """
        prompt = (
            "Evaluate if this text contains technical AI/ML infrastructure details. "
            f"Return 'YES' or 'NO':\n{text[:1000]}"
        )
        response = await self.llm.generate_response(prompt)
        return "YES" in response.upper()

    async def crawl(
        self,
        start_url: str,
        task_id: str = "",
        publish_fn: PublishFn = None,
    ) -> None:
        """
        Executes the breadth-first search crawling loop.

        Emits progress events via ``publish_fn`` if provided. Events include
        per-URL status updates and a terminal COMPLETED or FAILED event at
        the end of the crawl.

        Args:
            start_url: The seed URL to begin crawling from.
            task_id: Correlation ID passed through to published events.
            publish_fn: Optional async callback for TaskEvent progress updates.
                        Signature: async (TaskEvent) -> None
        """
        # Import here to avoid circular imports when task_models is unavailable.
        from worker.task_models import TaskEvent, TaskStatus

        async def _emit(status: TaskStatus, message: str, data: dict = {}) -> None:
            """Helper that emits an event only if a publish callback is registered."""
            if publish_fn is not None and task_id:
                event = TaskEvent(
                    task_id=task_id,
                    status=status,
                    message=message,
                    data=data,
                )
                await publish_fn(event)

        queue = [(start_url, 0)]
        relevant_urls: list[str] = []

        try:
            async with aiohttp.ClientSession() as session:
                while queue:
                    current_url, depth = queue.pop(0)

                    if depth > self.max_depth or current_url in self.visited_urls:
                        continue

                    self.visited_urls.add(current_url)
                    logger.info("Crawling (depth=%d): %s", depth, current_url)

                    await _emit(
                        TaskStatus.RUNNING,
                        f"Crawling URL at depth {depth}: {current_url}",
                        {"url": current_url, "depth": depth},
                    )

                    html = await self.fetch_page(session, current_url)
                    if not html:
                        await _emit(
                            TaskStatus.RUNNING,
                            f"Skipping unreachable URL: {current_url}",
                            {"url": current_url, "skipped": True},
                        )
                        continue

                    soup = BeautifulSoup(html, "html.parser")
                    text_content = soup.get_text(strip=True)

                    # Evaluate relevance via the LLM adapter.
                    is_relevant = await self.evaluate_content(text_content)
                    if is_relevant:
                        logger.info("Relevant content found at: %s", current_url)
                        relevant_urls.append(current_url)
                        await _emit(
                            TaskStatus.RUNNING,
                            f"Relevant content found at: {current_url}",
                            {"url": current_url, "relevant": True},
                        )

                    # Enqueue child links if we have depth budget remaining.
                    if depth < self.max_depth:
                        for link in soup.find_all("a", href=True):
                            href: str = str(link["href"])
                            if href.startswith("http") and href not in self.visited_urls:
                                queue.append((href, depth + 1))

            # Emit terminal COMPLETED event with aggregated results.
            await _emit(
                TaskStatus.COMPLETED,
                f"Crawl complete. Visited {len(self.visited_urls)} URL(s), "
                f"found {len(relevant_urls)} relevant page(s).",
                {
                    "visited_count": len(self.visited_urls),
                    "relevant_urls": relevant_urls,
                },
            )

        except Exception as exc:
            logger.exception("Crawl failed for start_url=%s: %s", start_url, exc)
            await _emit(
                TaskStatus.FAILED,
                f"Crawl error: {exc}",
                {"start_url": start_url},
            )
