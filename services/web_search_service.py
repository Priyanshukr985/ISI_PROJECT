from __future__ import annotations

import os

import requests


class WebSearchService:
    """Tavily-backed web search with safe fallbacks."""

    def __init__(self, api_key=None, timeout=10):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.timeout = timeout

    def _clean_snippets(self, results):
        snippets = []
        for item in results or []:
            content = ""
            if isinstance(item, dict):
                content = item.get("content") or item.get("snippet") or ""
            cleaned = " ".join(content.split()).strip()
            if cleaned:
                snippets.append(cleaned)
        return snippets[:3]

    def search_web(self, query: str):
        if not query or not self.api_key:
            return []

        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 3,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            return self._clean_snippets(payload.get("results", []))
        except Exception:
            return []
