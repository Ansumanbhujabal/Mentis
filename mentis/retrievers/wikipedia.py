"""Wikipedia retriever via REST API."""
from __future__ import annotations

import logging
from datetime import datetime
from urllib.parse import quote

import httpx

from mentis.schemas import Snippet

logger = logging.getLogger(__name__)

SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


class WikipediaRetriever:
    name = "wikipedia"
    source_kind = "background"

    async def search(self, query: str, n: int = 1) -> list[Snippet]:
        # Wikipedia REST API requires a User-Agent or returns 403 Forbidden.
        # Per Wikimedia policy, identify the app + contact URL.
        headers = {
            "User-Agent": "Mentis/0.1 (https://github.com/Ansumanbhujabal/Mentis) procurement-intelligence-poc",
            "Accept": "application/json",
        }
        title = quote(query.replace(" ", "_"))
        async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
            try:
                r = await client.get(SUMMARY_URL.format(title=title))
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                logger.warning(f"wikipedia fetch failed for {query!r}: {e!r}")
                return []

        extract = data.get("extract", "")
        if not extract:
            return []

        page_url = data.get("content_urls", {}).get("desktop", {}).get("page")
        if not page_url:
            return []

        return [
            Snippet(
                text=extract[:400],
                url=page_url,
                source_name="Wikipedia",
                source_kind="background",
                title=data.get("title"),
                retrieved_at=datetime.now(),
            )
        ]
