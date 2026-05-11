"""Query planner: user_query → QueryPlan with 7 section plans + RxNav normalization."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import BaseModel, Field

from mentis.llm import LLMClient, LLMConfig
from mentis.prompts import PromptRegistry
from mentis.schemas import QueryPlan, SectionPlan

PLAN_VERSION = "v1"
SYSTEM_PROMPT = (
    "You output strictly valid JSON. Follow the user's schema exactly. "
    "No commentary outside JSON."
)


class _PlanLLMOut(BaseModel):
    section_plans: list[dict] = Field(default_factory=list)


async def _default_llm_complete(
    *, llm_config: LLMConfig, system: str, user: str, schema: type, prompt_version: str
):
    client = LLMClient(llm_config)
    return await client.complete_with_safety(
        system=system, user=user, schema=schema, prompt_version=prompt_version
    )


async def plan_query(
    *,
    user_query: str,
    llm_config: LLMConfig,
    prompts_dir: Path,
    rxnav_normalize: Callable[[str], Awaitable[dict]] | None = None,
    llm_complete: Callable | None = None,
) -> QueryPlan:
    """Plan a 7-content-section research query.

    rxnav_normalize: callable that normalizes drug name. Defaults to real RxNav.
    llm_complete: injectable for tests. Defaults to real LLM call.
    """
    # 1. Normalize via RxNav
    if rxnav_normalize is None:
        from mentis.retrievers.rxnav import RxNavRetriever

        rxnav_normalize = RxNavRetriever().normalize

    norm = await rxnav_normalize(user_query)

    # 2. Render planner prompt
    registry = PromptRegistry(in_repo_dir=prompts_dir)
    prompt = registry.get("query_planner", version=PLAN_VERSION)
    user_prompt = prompt.render(
        user_query=user_query,
        normalized_term=norm.get("normalized_term") or "(unknown)",
        rxnav_synonyms=norm.get("synonyms", []),
    )

    # 3. LLM call (with safety chain)
    if llm_complete is None:
        plan_out, _used, _trace = await _default_llm_complete(
            llm_config=llm_config,
            system=SYSTEM_PROMPT,
            user=user_prompt,
            schema=_PlanLLMOut,
            prompt_version=prompt.version,
        )
    else:
        plan_out, _used, _trace = await llm_complete(
            llm_config=llm_config,
            system=SYSTEM_PROMPT,
            user=user_prompt,
            schema=_PlanLLMOut,
            prompt_version=prompt.version,
        )

    # 4. Build QueryPlan
    section_plans = [SectionPlan(**sp) for sp in plan_out.section_plans]
    return QueryPlan(
        user_query=user_query,
        normalized_term=norm.get("normalized_term"),
        rxnav_synonyms=norm.get("synonyms", []),
        section_plans=section_plans,
        plan_version=PLAN_VERSION,
    )
