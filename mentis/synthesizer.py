"""Synthesizer: snippets → SectionDraft with citation-grounded prose.

The anti-hallucination spine. Post-processes LLM output to verify every
cited URL is in the input snippet set; retries once; final fallback to
raw-snippet rendering with fallback_to_raw_snippets=True.
"""
from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from mentis.llm import LLMClient, LLMConfig, SafetyBlockedException
from mentis.prompts import PromptRegistry
from mentis.schemas import SectionDraft, SectionName, Snippet

logger = logging.getLogger(__name__)

SYNTH_VERSION = "v1"
SYSTEM_PROMPT = (
    "You output strictly valid JSON. Cite only the URLs provided in the snippets list. "
    "Never invent URLs or claims."
)

_URL_RE = re.compile(r"\]\((https?://[^)\s]+)\)")


class _SynthLLMOut(BaseModel):
    prose: str


def extract_cited_urls(prose: str) -> set[str]:
    return set(_URL_RE.findall(prose))


def _raw_snippets_render(snippets: list[Snippet]) -> str:
    lines = [
        "*Note: the AI synthesizer could not produce grounded prose; raw retrieved sources below.*",
        "",
    ]
    for s in snippets:
        title = s.title or s.source_name
        lines.append(f"- **{title}** — [{s.source_name}]({s.url})")
        lines.append(f"  {s.text[:300]}")
        lines.append("")
    return "\n".join(lines)


async def _default_llm_complete(
    *,
    llm_config: LLMConfig,
    system: str,
    user: str,
    schema,
    prompt_version: str,
    reframe_template: str | None,
):
    client = LLMClient(llm_config)
    return await client.complete_with_safety(
        system=system,
        user=user,
        schema=schema,
        prompt_version=prompt_version,
        reframe_template=reframe_template,
    )


async def synthesize_section(
    *,
    section_name: SectionName,
    snippets: list[Snippet],
    user_query: str,
    normalized_term: str | None,
    llm_config: LLMConfig,
    prompts_dir: Path,
    llm_complete: Callable | None = None,
) -> SectionDraft:
    """Run synthesizer with citation grounding + safety chain + retry-on-hallucination."""
    if not snippets:
        return SectionDraft(
            section_name=section_name,
            prose="*No sources retrieved for this section.*",
            snippets_used=[],
            citations=[],
            synthesizer_version=SYNTH_VERSION,
            safety_retries=0,
            fallback_to_raw_snippets=True,
        )

    registry = PromptRegistry(in_repo_dir=prompts_dir)
    prompt = registry.get("section_synthesizer", version=SYNTH_VERSION)
    reframe_template = None
    with contextlib.suppress(FileNotFoundError):
        reframe_template = registry.get("safety_reframe", version="v1").template_text

    user_prompt = prompt.render(
        section_name=section_name,
        user_query=user_query,
        normalized_term=normalized_term or "",
        snippets=snippets,
    )

    allowed_urls = {str(s.url) for s in snippets}

    async def _attempt():
        if llm_complete is None:
            return await _default_llm_complete(
                llm_config=llm_config,
                system=SYSTEM_PROMPT,
                user=user_prompt,
                schema=_SynthLLMOut,
                prompt_version=prompt.version,
                reframe_template=reframe_template,
            )
        return await llm_complete(
            llm_config=llm_config,
            system=SYSTEM_PROMPT,
            user=user_prompt,
            schema=_SynthLLMOut,
            prompt_version=prompt.version,
            reframe_template=reframe_template,
        )

    # Attempt 1
    safety_retries = 0
    try:
        out, _used, trace = await _attempt()
        safety_retries = trace.retry_count
        cited = extract_cited_urls(out.prose)
        if cited.issubset(allowed_urls):
            return SectionDraft(
                section_name=section_name,
                prose=out.prose,
                snippets_used=snippets,
                citations=list(cited),
                synthesizer_version=SYNTH_VERSION,
                safety_retries=safety_retries,
                fallback_to_raw_snippets=False,
            )
        logger.warning(
            "section %s: LLM cited unknown URLs %s; retrying",
            section_name,
            cited - allowed_urls,
        )
    except SafetyBlockedException:
        logger.warning("section %s: all safety paths blocked; raw snippets", section_name)
        return SectionDraft(
            section_name=section_name,
            prose=_raw_snippets_render(snippets),
            snippets_used=snippets,
            citations=[s.url for s in snippets],
            synthesizer_version=SYNTH_VERSION,
            safety_retries=4,
            fallback_to_raw_snippets=True,
        )

    # Attempt 2 (one retry on hallucinated URLs)
    try:
        out2, _used2, trace2 = await _attempt()
        safety_retries = max(safety_retries, trace2.retry_count)
        cited2 = extract_cited_urls(out2.prose)
        if cited2.issubset(allowed_urls):
            return SectionDraft(
                section_name=section_name,
                prose=out2.prose,
                snippets_used=snippets,
                citations=list(cited2),
                synthesizer_version=SYNTH_VERSION,
                safety_retries=safety_retries,
                fallback_to_raw_snippets=False,
            )
    except SafetyBlockedException:
        pass

    # Final fallback: raw snippets
    return SectionDraft(
        section_name=section_name,
        prose=_raw_snippets_render(snippets),
        snippets_used=snippets,
        citations=[s.url for s in snippets],
        synthesizer_version=SYNTH_VERSION,
        safety_retries=safety_retries,
        fallback_to_raw_snippets=True,
    )
