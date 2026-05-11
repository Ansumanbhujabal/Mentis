"""RxNav retriever — drug normalization, synonyms, RxCUI codes."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import httpx

from mentis.schemas import Snippet

logger = logging.getLogger(__name__)

DRUGS_URL = "https://rxnav.nlm.nih.gov/REST/drugs.json"
RXNAV_PAGE = "https://mor.nlm.nih.gov/RxNav/search?searchBy=String&searchTerm={query}"


class RxNavRetriever:
    name = "rxnav"
    source_kind = "scientific"

    async def normalize(self, query: str) -> dict:
        """Return {'normalized_term': str|None, 'synonyms': list[str], 'rxcui': str|None}.

        RxNav can be intermittently slow. Retry once with a longer timeout on
        timeout errors before giving up. Empty result is treated as "no match"
        and reported up so the planner can carry on without a normalized term.
        """
        data = None
        last_err: Exception | None = None
        for attempt, timeout in enumerate((20.0, 40.0), start=1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    r = await client.get(DRUGS_URL, params={"name": query})
                    r.raise_for_status()
                    data = r.json()
                    break
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_err = e
                logger.warning(
                    f"rxnav timeout for {query!r} (attempt {attempt}, timeout={timeout}s); "
                    f"{'retrying with longer timeout' if attempt == 1 else 'giving up'}"
                )
                if attempt == 1:
                    await asyncio.sleep(1.5)
                continue
            except Exception as e:
                last_err = e
                logger.warning(f"rxnav lookup failed for {query!r}: {e!r}")
                break

        if data is None:
            return {"normalized_term": None, "synonyms": [], "rxcui": None}

        groups = data.get("drugGroup", {}).get("conceptGroup", []) or []
        synonyms: list[str] = []
        normalized: str | None = None
        rxcui: str | None = None
        for g in groups:
            for c in g.get("conceptProperties", []) or []:
                if "name" in c:
                    synonyms.append(c["name"])
                if normalized is None and c.get("name"):
                    normalized = c["name"]
                if rxcui is None and c.get("rxcui"):
                    rxcui = c["rxcui"]
        return {"normalized_term": normalized, "synonyms": synonyms[:10], "rxcui": rxcui}

    async def search(self, query: str, n: int = 1) -> list[Snippet]:
        norm = await self.normalize(query)
        if not norm["normalized_term"]:
            return []
        text = (
            f"Normalized form: {norm['normalized_term']}. "
            f"Synonyms / brand forms: {', '.join(norm['synonyms'][:5])}. "
            f"RxCUI: {norm['rxcui']}."
        )
        return [
            Snippet(
                text=text,
                url=RXNAV_PAGE.format(query=query.replace(" ", "%20")),
                source_name="RxNav (NLM)",
                source_kind="scientific",
                title=norm["normalized_term"],
                retrieved_at=datetime.now(),
            )
        ]
