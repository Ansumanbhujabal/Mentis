from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mentis.retrievers.openfda import OpenFDARetriever

FIXTURES = Path(__file__).parent / "fixtures" / "openfda"


@pytest.mark.asyncio
async def test_openfda_returns_snippets() -> None:
    fake = json.loads((FIXTURES / "label_response.json").read_text())

    async def fake_get(url, *args, **kwargs):
        resp = MagicMock()
        resp.json = MagicMock(return_value=fake)
        resp.raise_for_status = MagicMock()
        return resp

    retriever = OpenFDARetriever()
    with patch("mentis.retrievers.openfda.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(side_effect=fake_get)
        snippets = await retriever.search("ranitidine", n=3)
    assert len(snippets) >= 1
    assert snippets[0].source_name == "OpenFDA"
    assert snippets[0].source_kind == "regulatory"
    assert "ranitidine" in snippets[0].text.lower() or "ndma" in snippets[0].text.lower()
