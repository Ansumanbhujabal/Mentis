"""Report assembler: 7 SectionDrafts → Executive Summary → References → final Report."""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from mentis.llm import LLMClient, LLMConfig, consume_pipeline_cost
from mentis.prompts import PromptRegistry
from mentis.schemas import Reference, Report, ReportMetadata, SectionDraft, SectionNames

logger = logging.getLogger(__name__)

MENTIS_VERSION = "0.1.0"
REPORT_VERSION = "v1"


class _ESLLMOut(BaseModel):
    executive_summary: str


SOURCE_AUTHORITY_ORDER = ["PubMed", "OpenFDA", "RxNav (NLM)", "Wikipedia", "Tavily"]


def _ref_sort_key(r: Reference) -> tuple[int, str]:
    try:
        idx = SOURCE_AUTHORITY_ORDER.index(r.source_name)
    except ValueError:
        idx = len(SOURCE_AUTHORITY_ORDER)
    return idx, str(r.url)


def build_references(drafts: list[SectionDraft]) -> list[Reference]:
    by_url: dict[str, Reference] = {}
    for d in drafts:
        for s in d.snippets_used:
            url_str = str(s.url)
            if url_str in by_url:
                if d.section_name not in by_url[url_str].used_in_sections:
                    by_url[url_str].used_in_sections.append(d.section_name)
            else:
                by_url[url_str] = Reference(
                    url=s.url,
                    source_name=s.source_name,
                    title=s.title or s.source_name,
                    retrieved_at=s.retrieved_at,
                    used_in_sections=[d.section_name],
                )
    return sorted(by_url.values(), key=_ref_sort_key)


async def _default_llm_complete(*, llm_config, system, user, schema, prompt_version):
    client = LLMClient(llm_config)
    return await client.complete_with_safety(
        system=system, user=user, schema=schema, prompt_version=prompt_version
    )


async def assemble_report(
    *,
    user_query: str,
    normalized_term: str | None,
    section_drafts: dict,
    llm_config: LLMConfig,
    prompts_dir: Path,
    total_latency_ms: int,
    llm_complete: Callable | None = None,
) -> Report:
    """Assemble 7 SectionDrafts into a Report with ES + References."""
    ordered = [section_drafts[name] for name in SectionNames if name in section_drafts]

    registry = PromptRegistry(in_repo_dir=prompts_dir)
    es_prompt = registry.get("executive_summary", version="v1")
    user_prompt = es_prompt.render(
        user_query=user_query,
        normalized_term=normalized_term or "",
        sections=ordered,
    )
    sys_prompt = "You output strictly valid JSON. No commentary."
    if llm_complete is None:
        es_out, _used, _trace = await _default_llm_complete(
            llm_config=llm_config,
            system=sys_prompt,
            user=user_prompt,
            schema=_ESLLMOut,
            prompt_version=es_prompt.version,
        )
    else:
        es_out, _used, _trace = await llm_complete(
            llm_config=llm_config,
            system=sys_prompt,
            user=user_prompt,
            schema=_ESLLMOut,
            prompt_version=es_prompt.version,
        )

    references = build_references(ordered)
    safety_retries_total = sum(d.safety_retries for d in ordered)
    snippets_total = sum(len(d.snippets_used) for d in ordered)

    # Drain the per-pipeline cost accumulator (populated by every _call in llm.py).
    # Called AFTER the ES LLM call above so its cost is included.
    cost_usd = consume_pipeline_cost()

    metadata = ReportMetadata(
        mentis_version=MENTIS_VERSION,
        llm_provider=llm_config.providers[0],
        prompt_versions={
            "section_synthesizer": "v1",
            "query_planner": "v1",
            "executive_summary": "v1",
        },
        total_latency_ms=total_latency_ms,
        total_snippets_retrieved=snippets_total,
        total_safety_retries=safety_retries_total,
        cost_usd=cost_usd,
        generated_at=datetime.now(),
    )

    return Report(
        user_query=user_query,
        normalized_term=normalized_term,
        sections=ordered,
        executive_summary=es_out.executive_summary,
        references=references,
        metadata=metadata,
    )


SECTION_TITLES = {
    "product_profile": "Product Profile",
    "clinical_use": "Clinical Use & Mechanism",
    "market_demand": "Market Size & Demand Drivers",
    "manufacturers": "Top Manufacturers & Suppliers",
    "regulatory": "Regulatory & Compliance",
    "sourcing_pricing": "Sourcing & Pricing Channels",
    "risks_alternatives": "Risks & Alternatives",
}


def render_markdown(report: Report) -> str:
    md: list[str] = []
    md.append("# Mentis Procurement Brief")
    md.append("")
    md.append(f"**Substance:** {report.user_query}")
    if report.normalized_term:
        md.append(f"**Normalized:** {report.normalized_term}")
    m = report.metadata
    md.append(f"**Generated:** {m.generated_at.strftime('%Y-%m-%d %H:%M')}")
    md.append(f"**Model:** {m.llm_provider} · prompt versions: {m.prompt_versions}")
    md.append(
        f"**Sources retrieved:** {m.total_snippets_retrieved} · "
        f"Cost: ${m.cost_usd:.4f} · "
        f"Latency: {m.total_latency_ms} ms · "
        f"Safety retries: {m.total_safety_retries}"
    )
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Executive Summary")
    md.append("")
    md.append(report.executive_summary)
    md.append("")
    for s in report.sections:
        md.append(f"## {SECTION_TITLES.get(s.section_name, s.section_name)}")
        md.append("")
        md.append(s.prose)
        md.append("")
        if s.fallback_to_raw_snippets:
            md.append(
                "> *This section could not be synthesized due to provider "
                "safety filters; raw sources shown above.*"
            )
            md.append("")
    md.append("## References")
    md.append("")
    for i, r in enumerate(report.references, start=1):
        sections_str = ", ".join(r.used_in_sections)
        retrieved_str = r.retrieved_at.strftime("%Y-%m-%d")
        md.append(
            f"{i}. **{r.source_name}** — [{r.title}]({r.url})"
            f" — retrieved {retrieved_str} — used in: {sections_str}"
        )
    md.append("")
    return "\n".join(md)
