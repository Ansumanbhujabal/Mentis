from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mentis.retrievers.pubmed import PubmedRetriever

FIXTURES = Path(__file__).parent / "fixtures" / "pubmed"


@pytest.mark.asyncio
async def test_pubmed_returns_snippets() -> None:
    esearch_json = json.loads((FIXTURES / "esearch_response.json").read_text())
    efetch_xml = (FIXTURES / "efetch_response.xml").read_text()

    async def fake_get(url, *args, **kwargs):
        resp = MagicMock()
        if "esearch" in url:
            resp.json = MagicMock(return_value=esearch_json)
        else:
            resp.text = efetch_xml
        resp.raise_for_status = MagicMock()
        return resp

    retriever = PubmedRetriever()
    with patch("mentis.retrievers.pubmed.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(side_effect=fake_get)
        snippets = await retriever.search("ranitidine pharmacokinetics", n=2)
    assert len(snippets) == 2
    assert snippets[0].source_name == "PubMed"
    assert snippets[0].source_kind == "scientific"
    text_lower = snippets[0].text.lower()
    title_lower = (snippets[0].title or "").lower()
    assert "ranitidine" in text_lower or "ranitidine" in title_lower
    assert "pubmed.ncbi.nlm.nih.gov" in str(snippets[0].url)
