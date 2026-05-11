from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mentis.retrievers.wikipedia import WikipediaRetriever


@pytest.mark.asyncio
async def test_wikipedia_returns_snippet() -> None:
    fake = {
        "title": "Ranitidine",
        "extract": "Ranitidine is an H2 receptor antagonist used to treat ulcers.",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Ranitidine"}},
    }

    async def fake_get(url, *args, **kwargs):
        r = MagicMock()
        r.json = MagicMock(return_value=fake)
        r.raise_for_status = MagicMock()
        return r

    retriever = WikipediaRetriever()
    with patch("mentis.retrievers.wikipedia.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(side_effect=fake_get)
        snippets = await retriever.search("ranitidine")
    assert len(snippets) == 1
    assert "ranitidine" in snippets[0].text.lower()
    assert snippets[0].source_name == "Wikipedia"
