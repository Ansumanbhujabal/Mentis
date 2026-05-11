from __future__ import annotations

from unittest.mock import patch

import pytest

from mentis.retrievers.tavily import TavilyRetriever


@pytest.mark.asyncio
async def test_tavily_returns_snippets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    fake_response = {
        "results": [
            {
                "title": "Ranitidine market overview 2025",
                "url": "https://example.org/ranitidine-market",
                "content": (
                    "The global ranitidine market reached $1.2B in 2024, "
                    "with 8% YoY growth driven by generic adoption."
                ),
            },
            {
                "title": "Top H2 antagonist manufacturers",
                "url": "https://example.org/h2-mfrs",
                "content": "Leading manufacturers include Pfizer, Glenmark, and Cipla.",
            },
        ]
    }
    with patch("mentis.retrievers.tavily.TavilyClient") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.search = lambda *args, **kwargs: fake_response
        retriever = TavilyRetriever()
        snippets = await retriever.search("ranitidine market", n=2)
    assert len(snippets) == 2
    assert snippets[0].source_name == "Tavily"
    assert "ranitidine" in snippets[0].text.lower()
    assert str(snippets[0].url) == "https://example.org/ranitidine-market"
