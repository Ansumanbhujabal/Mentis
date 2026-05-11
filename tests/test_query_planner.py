from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from mentis.llm import LLMConfig, SafetyTrace
from mentis.query_planner import plan_query
from mentis.schemas import QueryPlan


class _FakeLLMOut(BaseModel):
    section_plans: list[dict]


@pytest.mark.asyncio
async def test_query_planner_returns_seven_sections(tmp_path: Path) -> None:
    # Set up local prompt file
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "query_planner.v1.j2").write_text(
        "plan for {{ user_query }} - normalized {{ normalized_term }}"
    )

    fake_plan = _FakeLLMOut(
        section_plans=[
            {"section_name": s, "search_queries": [f"{s} q"], "sources": ["tavily"]}
            for s in [
                "product_profile",
                "clinical_use",
                "market_demand",
                "manufacturers",
                "regulatory",
                "sourcing_pricing",
                "risks_alternatives",
            ]
        ]
    )

    fake_rxnav_normalize = AsyncMock(
        return_value={"normalized_term": "ranitidine HCl", "synonyms": ["Zantac"], "rxcui": "12345"}
    )

    fake_llm = AsyncMock(return_value=(fake_plan, "gemini/gemini-2.0-flash", SafetyTrace()))

    plan = await plan_query(
        user_query="ranitidine",
        llm_config=LLMConfig(),
        prompts_dir=prompts_dir,
        rxnav_normalize=fake_rxnav_normalize,
        llm_complete=fake_llm,
    )
    assert isinstance(plan, QueryPlan)
    assert plan.normalized_term == "ranitidine HCl"
    assert len(plan.section_plans) == 7
    assert plan.section_plans[0].section_name == "product_profile"
