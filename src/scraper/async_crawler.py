import aiohttp
from typing import Set
from bs4 import BeautifulSoup
from src.adapter.factory import LLMAdapterFactory


class AgenticScraper:
    """
    Asynchronous BFS web scraper that uses an LLM to evaluate content relevance.
    """

    def __init__(self, provider: str, max_depth: int = 2):
        self.max_depth = max_depth
        self.visited_urls: Set[str] = set()
        # Initialize the LLM adapter dynamically
        self.llm = LLMAdapterFactory.create_adapter(provider)

    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> str:
        """
        Fetches the HTML content of a given URL.
        """
        try:
            async with session.get(url, timeout=10) as response:
                return await response.text()
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return ""

    async def evaluate_content(self, text: str) -> bool:
        """
        Uses the LLM to determine if the scraped content is highly relevant.
        """
        prompt = f"Evaluate if this text contains technical AI/ML infrastructure details. Return 'YES' or 'NO':\n{text[:1000]}"
        response = self.llm.generate_response(prompt)
        return "YES" in response.upper()

    async def crawl(self, start_url: str) -> None:
        """
        Executes the breadth-first search crawling loop.
        """
        queue = [(start_url, 0)]

        async with aiohttp.ClientSession() as session:
            while queue:
                current_url, depth = queue.pop(0)

                if depth > self.max_depth or current_url in self.visited_urls:
                    continue

                self.visited_urls.add(current_url)
                html = await self.fetch_page(session, current_url)

                if not html:
                    continue

                soup = BeautifulSoup(html, 'html.parser')
                text_content = soup.get_text(strip=True)

                # Context evaluation using LLM adapter
                is_relevant = await self.evaluate_content(text_content)
                if is_relevant:
                    print(f"Highly relevant content found at: {current_url}")
