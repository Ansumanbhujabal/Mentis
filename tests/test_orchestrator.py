from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from mentis.cache import Cache
from mentis.orchestrator import retrieve_for_plan
from mentis.schemas import QueryPlan, SectionPlan, Snippet


def _snip(name: str, kind: str = "market") -> Snippet:
    return Snippet(
        text=f"fact about {name}",
        url=f"https://example.org/{name}",
        source_name=name,
        source_kind=kind,
        retrieved_at=datetime.now(),
    )


@pytest.mark.asyncio
async def test_orchestrator_routes_per_section(tmp_path: Path) -> None:
    cache = Cache(root=tmp_path / "cache")
    qp = QueryPlan(
        user_query="ranitidine",
        section_plans=[
            SectionPlan(
                section_name="product_profile",
                search_queries=["q1"],
                sources=["rxnav", "pubmed", "wikipedia"],
            ),
            SectionPlan(section_name="market_demand", search_queries=["q2"], sources=["tavily"]),
        ],
    )
    fake = {
        "rxnav": AsyncMock(return_value=[_snip("rxnav", "scientific")]),
        "pubmed": AsyncMock(return_value=[_snip("pubmed", "scientific")]),
        "wikipedia": AsyncMock(return_value=[_snip("wiki", "background")]),
        "tavily": AsyncMock(return_value=[_snip("tav", "market")]),
        "openfda": AsyncMock(return_value=[]),
    }
    retrievers = {name: type("R", (), {"search": fn, "name": name})() for name, fn in fake.items()}
    result = await retrieve_for_plan(qp, retrievers=retrievers, cache=cache)
    assert set(result.keys()) == {"product_profile", "market_demand"}
    assert any(s.source_name == "rxnav" for s in result["product_profile"])
    assert any(s.source_name == "tav" for s in result["market_demand"])
