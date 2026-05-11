"""Per-section source routing + parallel retrieval orchestrator."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel

from mentis.cache import Cache
from mentis.schemas import QueryPlan, SectionName, Snippet

logger = logging.getLogger(__name__)

ORCH_VERSION = "v1"
CONCURRENCY = 10

SOURCE_ROUTING: dict[SectionName, list[str]] = {
    "product_profile":      ["rxnav", "pubmed", "wikipedia"],
    "clinical_use":         ["pubmed", "openfda"],
    "market_demand":        ["tavily"],
    "manufacturers":        ["tavily"],
    "regulatory":           ["openfda", "tavily"],
    "sourcing_pricing":     ["tavily"],
    "risks_alternatives":   ["openfda", "tavily"],
}


class _SnippetBundle(BaseModel):
    snippets: list[Snippet]


async def _retrieve_one(
    retriever, query: str, n: int, cache: Cache, sem: asyncio.Semaphore
) -> list[Snippet]:
    key = Cache.compute_hash(retriever.name, query, ORCH_VERSION)
    hit = cache.get("retrieve", key, _SnippetBundle)
    if hit is not None and hit.snippets:
        # Only trust a cache hit if it has actual snippets. Empty results may have
        # come from a transient API failure — let the retriever retry next time.
        return hit.snippets
    async with sem:
        try:
            snips = await retriever.search(query, n=n)
        except Exception as e:
            logger.warning(f"retriever {retriever.name} failed for {query!r}: {e!r}")
            return []
    # Only cache non-empty results
    if snips:
        cache.set("retrieve", key, _SnippetBundle(snippets=snips))
    return snips


async def retrieve_for_plan(
    plan: QueryPlan,
    *,
    retrievers: dict[str, Any],
    cache: Cache,
    n_per_query: int = 3,
) -> dict[str, list[Snippet]]:
    """For each SectionPlan, fire all (source, query) combos in parallel.

    Returns {section_name: deduped list of snippets}.
    """
    sem = asyncio.Semaphore(CONCURRENCY)

    user_query = plan.user_query
    normalized = plan.normalized_term

    async def _per_section(sp) -> tuple[str, list[Snippet]]:
        coros = []
        sources_to_use = sp.sources or SOURCE_ROUTING.get(sp.section_name, [])
        for src in sources_to_use:
            r = retrievers.get(src)
            if r is None:
                continue
            for q in sp.search_queries:
                coros.append(_retrieve_one(r, q, n_per_query, cache, sem))
        results = await asyncio.gather(*coros)
        seen: set[str] = set()
        merged: list[Snippet] = []
        for batch in results:
            for s in batch:
                key = str(s.url)
                if key not in seen:
                    seen.add(key)
                    merged.append(s)

        # Fallback: if the planner's queries returned nothing for this section,
        # retry each retriever with the bare user_query (and normalized term if available).
        # Common case: planner generates "minoxidil pharmacological profile" which Wikipedia /
        # RxNav can't match, but "minoxidil" alone returns the canonical article + entry.
        if not merged:
            fallback_queries = [q for q in (user_query, normalized) if q]
            for src in sources_to_use:
                r = retrievers.get(src)
                if r is None:
                    continue
                for q in fallback_queries:
                    snips = await _retrieve_one(r, q, n_per_query, cache, sem)
                    for s in snips:
                        key = str(s.url)
                        if key not in seen:
                            seen.add(key)
                            merged.append(s)

        return sp.section_name, merged

    pairs = await asyncio.gather(*[_per_section(sp) for sp in plan.section_plans])
    return dict(pairs)
