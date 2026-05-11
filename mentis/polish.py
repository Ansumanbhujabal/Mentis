"""Post-processing layer: takes an existing report's markdown and re-renders it
with tables + tighter structure, WITHOUT re-fetching data or re-synthesising.

One LLM call (sees the full report). Hard constraint via prompt: no new claims,
no new citations, no section reordering — just restructure each section to lead
with a headline + add a markdown table where data permits.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from mentis.llm import LLMClient, LLMConfig
from mentis.prompts import PromptRegistry

logger = logging.getLogger(__name__)

POLISH_VERSION = "v1"
SYSTEM_PROMPT = "You output strictly valid JSON. No commentary outside JSON."


class _PolishLLMOut(BaseModel):
    polished_markdown: str


async def _default_llm_complete(
    *, llm_config: LLMConfig, system: str, user: str, schema, prompt_version: str
):
    client = LLMClient(llm_config)
    return await client.complete_with_safety(
        system=system, user=user, schema=schema, prompt_version=prompt_version
    )


def _extract_allowed_urls(markdown: str) -> set[str]:
    """All URLs in the source markdown — used to verify the polish step didn't invent any."""
    return set(re.findall(r"\]\((https?://[^)\s]+)\)", markdown))


async def polish_report_markdown(
    *,
    user_query: str,
    original_markdown: str,
    llm_config: LLMConfig,
    prompts_dir: Path,
    llm_complete: Callable | None = None,
) -> str:
    """Run the polish prompt on existing report markdown.

    Returns the polished markdown. On hallucinated-URL detection, falls back
    to returning the original markdown (no silent corruption).
    """
    registry = PromptRegistry(in_repo_dir=prompts_dir)
    prompt = registry.get("report_polisher", version=POLISH_VERSION)
    allowed_urls = sorted(_extract_allowed_urls(original_markdown))
    user_prompt = prompt.render(
        user_query=user_query,
        original_markdown=original_markdown,
        allowed_urls=allowed_urls,
    )

    if llm_complete is None:
        out, _used, _trace = await _default_llm_complete(
            llm_config=llm_config,
            system=SYSTEM_PROMPT,
            user=user_prompt,
            schema=_PolishLLMOut,
            prompt_version=prompt.version,
        )
    else:
        out, _used, _trace = await llm_complete(
            llm_config=llm_config,
            system=SYSTEM_PROMPT,
            user=user_prompt,
            schema=_PolishLLMOut,
            prompt_version=prompt.version,
        )

    polished = out.polished_markdown.strip()

    # Anti-hallucination guard: every URL in the polished output must already
    # exist in the original. If the LLM invented citations, fall back.
    original_urls = _extract_allowed_urls(original_markdown)
    polished_urls = _extract_allowed_urls(polished)
    invented = polished_urls - original_urls
    if invented:
        logger.warning(
            f"polish step invented {len(invented)} URL(s); reverting to original markdown: "
            f"{list(invented)[:3]}"
        )
        return original_markdown

    return polished


def slug_filename(query: str, kind: str = "pdf") -> str:
    """Convert a substance query to a {Substance}_mentis_report.{kind} filename."""
    # Keep alphanumerics + percent signs; replace whitespace and punctuation with _
    cleaned = re.sub(r"[^A-Za-z0-9%]+", "_", query.strip()).strip("_")
    # Title-case the first letter for display
    if cleaned and cleaned[0].isalpha():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return f"{cleaned}_mentis_report.{kind}"
