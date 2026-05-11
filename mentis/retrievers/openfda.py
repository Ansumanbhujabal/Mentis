"""OpenFDA retriever — drug labels."""
from __future__ import annotations

import logging
from datetime import datetime

import httpx

from mentis.schemas import Snippet

logger = logging.getLogger(__name__)

LABEL_URL = "https://api.fda.gov/drug/label.json"
ARTICLE_URL = "https://nctr-crs.fda.gov/fdalabel/services/spl/summaries/{id}"


class OpenFDARetriever:
    name = "openfda"
    source_kind = "regulatory"

    async def search(self, query: str, n: int = 5) -> list[Snippet]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                search_query = f'(openfda.generic_name:"{query}" OR openfda.brand_name:"{query}")'
                r = await client.get(
                    LABEL_URL,
                    params={
                        "search": search_query,
                        "limit": n,
                    },
                )
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                logger.warning(f"openfda label search failed for {query!r}: {e!r}")
                return []

        out: list[Snippet] = []
        for result in data.get("results", [])[:n]:
            openfda_meta = result.get("openfda", {})
            brand = openfda_meta.get("brand_name", ["?"])[0]
            generic = openfda_meta.get("generic_name", ["?"])[0]
            mfr = openfda_meta.get("manufacturer_name", ["?"])[0]
            indications = " ".join(result.get("indications_and_usage", []))[:400]
            warnings = " ".join(result.get("warnings", []))[:400]
            text_parts = [
                f"{brand} ({generic}) — Manufacturer: {mfr}.",
                f"Indications: {indications}",
                f"Warnings: {warnings}",
            ]
            text = " ".join(text_parts)[:800]
            label_id = result.get("id", "")
            try:
                out.append(
                    Snippet(
                        text=text,
                        url=ARTICLE_URL.format(id=label_id),
                        source_name="OpenFDA",
                        source_kind="regulatory",
                        title=f"{brand} ({generic})",
                        retrieved_at=datetime.now(),
                    )
                )
            except Exception as e:
                logger.warning(f"skipping openfda result: {e!r}")
        return out
