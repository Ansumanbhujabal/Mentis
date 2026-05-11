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


async def polish_section_prose(
    *,
    section_name: str,
    original_prose: str,
    user_query: str,
    llm_config: LLMConfig,
    prompts_dir: Path,
    llm_complete: Callable | None = None,
) -> str:
    """Polish ONE section: add a headline + table on top, keep prose verbatim.

    Returns the polished section markdown. On URL hallucination, returns the
    original prose unchanged.
    """
    allowed_urls = sorted(_extract_allowed_urls(original_prose))
    registry = PromptRegistry(in_repo_dir=prompts_dir)
    prompt = registry.get("section_polisher", version="v1")
    user_prompt = prompt.render(
        section_name=section_name,
        user_query=user_query,
        original_prose=original_prose,
        allowed_urls=allowed_urls,
    )

    class _SectionPolishOut(BaseModel):
        polished: str

    if llm_complete is None:
        out, _used, _trace = await _default_llm_complete(
            llm_config=llm_config,
            system=SYSTEM_PROMPT,
            user=user_prompt,
            schema=_SectionPolishOut,
            prompt_version=prompt.version,
        )
    else:
        out, _used, _trace = await llm_complete(
            llm_config=llm_config,
            system=SYSTEM_PROMPT,
            user=user_prompt,
            schema=_SectionPolishOut,
            prompt_version=prompt.version,
        )

    polished = out.polished.strip()

    # Hallucination guard
    if (_extract_allowed_urls(polished) - set(allowed_urls)):
        logger.warning(f"polish_section_prose for {section_name}: hallucinated URL; reverting")
        return original_prose

    return polished


async def polish_report_per_section(
    *,
    user_query: str,
    original_markdown: str,
    llm_config: LLMConfig,
    prompts_dir: Path,
    section_titles_to_keys: dict[str, str] | None = None,
    llm_complete: Callable | None = None,
) -> str:
    """Parse a rendered report markdown, polish each content section, reassemble.

    Each section gets its own LLM call. Header + Executive Summary + References
    pass through unchanged. Cheaper-per-section than a single whole-report call
    AND more reliable at preserving prose because each call is narrowly scoped.
    """
    # Default mapping from display title back to section_name keys
    if section_titles_to_keys is None:
        section_titles_to_keys = {
            "Product Profile": "product_profile",
            "Clinical Use & Mechanism": "clinical_use",
            "Market Size & Demand Drivers": "market_demand",
            "Top Manufacturers & Suppliers": "manufacturers",
            "Regulatory & Compliance": "regulatory",
            "Sourcing & Pricing Channels": "sourcing_pricing",
            "Risks & Alternatives": "risks_alternatives",
        }

    # Split the markdown at every "## " heading
    parts = re.split(r"(?m)^## ", original_markdown)
    # parts[0] = everything before the first ##; subsequent = "<title>\n<body>"
    out_parts = [parts[0]]
    for chunk in parts[1:]:
        if not chunk.strip():
            continue
        # First line is the section title
        title_end = chunk.find("\n")
        if title_end == -1:
            out_parts.append("## " + chunk)
            continue
        title = chunk[:title_end].strip()
        body = chunk[title_end + 1 :]

        # Only polish content sections — skip Executive Summary and References
        section_key = section_titles_to_keys.get(title)
        if section_key is None:
            out_parts.append("## " + chunk)
            continue

        # Polish this section's body
        try:
            polished_body = await polish_section_prose(
                section_name=section_key,
                original_prose=body,
                user_query=user_query,
                llm_config=llm_config,
                prompts_dir=prompts_dir,
                llm_complete=llm_complete,
            )
        except Exception as e:
            logger.warning(f"polish failed for {title}: {e!r}; keeping original")
            polished_body = body
        out_parts.append(f"## {title}\n{polished_body}\n")

    return "\n".join(out_parts) if out_parts[0] == "" else "## ".join(out_parts) if False else "".join(
        [out_parts[0]] + ["## " + p for p in out_parts[1:] if not p.startswith("## ")] + [p for p in out_parts[1:] if p.startswith("## ")]
    ) if False else _rejoin(out_parts)


def _rejoin(parts: list[str]) -> str:
    """Rejoin parts back into a markdown doc. First part has no leading '## ';
    subsequent parts already include '## ' from the polish path above."""
    if not parts:
        return ""
    first = parts[0]
    rest = parts[1:]
    return first + "".join(rest)


def slug_filename(query: str, kind: str = "pdf") -> str:
    """Convert a substance query to a {Substance}_mentis_report.{kind} filename."""
    # Keep alphanumerics + percent signs; replace whitespace and punctuation with _
    cleaned = re.sub(r"[^A-Za-z0-9%]+", "_", query.strip()).strip("_")
    # Title-case the first letter for display
    if cleaned and cleaned[0].isalpha():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return f"{cleaned}_mentis_report.{kind}"
