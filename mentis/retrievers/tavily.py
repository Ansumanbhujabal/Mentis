"""Tavily web search retriever."""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime

from tavily import TavilyClient

from mentis.schemas import Snippet

logger = logging.getLogger(__name__)


class TavilyRetriever:
    name = "tavily"
    source_kind = "market"

    def __init__(self) -> None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY not set")
        self._client = TavilyClient(api_key=api_key)

    async def search(self, query: str, n: int = 5) -> list[Snippet]:
        def _do_search():
            return self._client.search(query=query, max_results=n, search_depth="advanced")

        try:
            resp = await asyncio.to_thread(_do_search)
        except Exception as e:
            logger.warning(f"tavily search failed for {query!r}: {e!r}")
            return []

        out: list[Snippet] = []
        for r in resp.get("results", [])[:n]:
            try:
                out.append(
                    Snippet(
                        text=r.get("content", "")[:400],
                        url=r["url"],
                        source_name="Tavily",
                        source_kind="market",
                        title=r.get("title"),
                        retrieved_at=datetime.now(),
                    )
                )
            except Exception as e:
                logger.warning(f"skipping tavily result: {e!r}")
        return out
