from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from mentis.llm import LLMConfig, SafetyTrace
from mentis.schemas import Snippet
from mentis.synthesizer import (
    extract_cited_urls,
    synthesize_section,
)


class _SynthOut(BaseModel):
    prose: str


def _snip(url: str) -> Snippet:
    return Snippet(
        text=f"fact at {url}",
        url=url,
        source_name="X",
        source_kind="market",
        retrieved_at=datetime.now(),
    )


def test_extract_cited_urls() -> None:
    text = "Studies have shown [a 12% rise](https://a.org/x) and [also Y](https://b.org/y) here."
    assert extract_cited_urls(text) == {"https://a.org/x", "https://b.org/y"}


@pytest.mark.asyncio
async def test_synthesize_section_grounded(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    tmpl = (
        "Write {{ section_name }}. Use snippets only."
        " {% for s in snippets %}{{ s.url }}{% endfor %}"
    )
    (prompts_dir / "section_synthesizer.v1.j2").write_text(tmpl)
    snips = [_snip("https://a.org/x"), _snip("https://b.org/y")]
    valid_output = _SynthOut(
        prose="A finding from [source A](https://a.org/x) and a different [source B](https://b.org/y)."
    )
    fake_llm = AsyncMock(return_value=(valid_output, "gemini/gemini-2.0-flash", SafetyTrace()))

    draft = await synthesize_section(
        section_name="market_demand",
        snippets=snips,
        user_query="ranitidine",
        normalized_term="ranitidine HCl",
        llm_config=LLMConfig(),
        prompts_dir=prompts_dir,
        llm_complete=fake_llm,
    )
    assert draft.section_name == "market_demand"
    assert "source A" in draft.prose
    assert len(draft.citations) == 2
    assert draft.fallback_to_raw_snippets is False


@pytest.mark.asyncio
async def test_synthesize_falls_back_on_hallucinated_url(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "section_synthesizer.v1.j2").write_text("placeholder")
    snips = [_snip("https://a.org/x")]
    hallucinated = _SynthOut(
        prose="Found at [evil](https://hallucinated.org/zzz)."
    )

    call_count = {"n": 0}

    async def fake_llm(*args, **kwargs):
        call_count["n"] += 1
        return hallucinated, "gemini/gemini-2.0-flash", SafetyTrace()

    draft = await synthesize_section(
        section_name="market_demand",
        snippets=snips,
        user_query="ranitidine",
        normalized_term="ranitidine HCl",
        llm_config=LLMConfig(),
        prompts_dir=prompts_dir,
        llm_complete=fake_llm,
    )
    assert call_count["n"] == 2  # retried once
    assert draft.fallback_to_raw_snippets is True
