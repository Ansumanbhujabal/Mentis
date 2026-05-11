from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mentis.retrievers.rxnav import RxNavRetriever


@pytest.mark.asyncio
async def test_rxnav_normalize() -> None:
    fake = {
        "drugGroup": {
            "conceptGroup": [
                {
                    "tty": "IN",
                    "conceptProperties": [
                        {"name": "ranitidine hydrochloride", "rxcui": "12345"},
                        {"name": "Zantac"},
                    ],
                }
            ]
        }
    }

    async def fake_get(url, *args, **kwargs):
        r = MagicMock()
        r.json = MagicMock(return_value=fake)
        r.raise_for_status = MagicMock()
        return r

    retriever = RxNavRetriever()
    with patch("mentis.retrievers.rxnav.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(side_effect=fake_get)
        norm = await retriever.normalize("ranitidine")
    assert norm["normalized_term"] == "ranitidine hydrochloride"
    assert "Zantac" in norm["synonyms"]
    assert norm["rxcui"] == "12345"
