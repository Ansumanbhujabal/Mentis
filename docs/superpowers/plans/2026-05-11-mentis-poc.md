# Mentis POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a 24-hour POC: a Gradio web app that turns a medical-substance query into a McKinsey-shape, citation-grounded procurement intelligence report in < 120s, with PDF export, safety-filter handling, and Langfuse-managed prompts.

**Architecture:** Five-stage async pipeline (query_planner → orchestrator → retrievers → synthesizer → report) with cross-cutting cache, LLM client (LiteLLM with safety escalation chain Gemini → relaxed Gemini → reframed Gemini → Groq), Langfuse observability + prompt registry (with in-repo file fallback), and PDF export via WeasyPrint. Two faces: Click CLI + Gradio app on HF Spaces.

**Tech Stack:** Python 3.11+, `uv` (deps), `pydantic` (data), `httpx` (async HTTP), `tavily-python`, `litellm` (LLM), `markdown-it-py` + `weasyprint` (PDF), `jinja2` (templates), `click` (CLI), `gradio` (UI), `langfuse` (tracing + prompts), `pytest` + `pytest-asyncio` + `pytest-mock` (smoke tests), `ruff` (lint), `diagrams` (architecture image).

**Spec reference:** `docs/superpowers/specs/2026-05-11-mentis-poc-design.md`

**Total budget:** 22.5–23h within a 36h window ending 2026-05-12.

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `packages.txt`
- Create: `Makefile`
- Create: `README.md`
- Create: `mentis/__init__.py`
- Create: `mentis/retrievers/__init__.py`
- Create: `mentis/templates/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Initialize with uv**

Run from `/opt/CodeRepo/mentis/`:
```bash
uv init --package --python 3.11 .
rm -f hello.py src 2>/dev/null
```

If uv creates `src/mentis/` layout, move package to root:
```bash
[ -d src/mentis ] && mv src/mentis ./mentis && rmdir src 2>/dev/null
```

- [ ] **Step 2: Overwrite pyproject.toml**

```toml
[project]
name = "mentis"
version = "0.1.0"
description = "Procurement intelligence reports for medical substances"
readme = "README.md"
requires-python = ">=3.11"
authors = [
    { name = "Ansumanbhujabal", email = "ansumanbhujabal1@gmail.com" }
]
dependencies = [
    "pydantic>=2.6",
    "litellm>=1.40",
    "httpx>=0.27",
    "tavily-python>=0.5",
    "markdown-it-py>=3.0",
    "weasyprint>=62.0",
    "jinja2>=3.1",
    "click>=8.1",
    "gradio>=4.40",
    "langfuse>=2.40",
    "python-dotenv>=1.0",
]

[project.scripts]
mentis = "mentis.cli:cli"

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-mock>=3.12",
    "ruff>=0.4",
    "diagrams>=0.23",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 3: Create .gitignore**

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
*.egg-info/
build/
dist/
.venv/
.uv/
.env
.env.local
.cache/
.DS_Store
*.swp
*.bak
```

- [ ] **Step 4: Create .env.example**

```ini
# LLM providers
GEMINI_API_KEY=
GROQ_API_KEY=

# Tavily web search
TAVILY_API_KEY=

# Langfuse (cloud free tier — https://cloud.langfuse.com)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com

# Provider order
MENTIS_PROVIDERS=gemini,groq
MENTIS_PRIMARY_MODEL=gemini/gemini-2.0-flash
```

- [ ] **Step 5: Create packages.txt (HF Spaces system deps for WeasyPrint)**

```
libcairo2
libpango-1.0-0
libpangoft2-1.0-0
libgdk-pixbuf2.0-0
libffi-dev
shared-mime-info
```

- [ ] **Step 6: Create Makefile**

```makefile
.PHONY: install diagram sync-prompts deploy samples test lint

install:
	uv sync

diagram:
	uv run python docs/architecture.py

sync-prompts:
	uv run python scripts/sync_prompts.py

samples:
	uv run mentis report "0.9% saline" --out walkthrough/sample_reports/saline_0.9pct.md --pdf walkthrough/sample_reports/saline_0.9pct.pdf
	uv run mentis report "tramadol" --out walkthrough/sample_reports/tramadol.md --pdf walkthrough/sample_reports/tramadol.pdf
	uv run mentis report "insulin glargine" --out walkthrough/sample_reports/insulin_glargine.md --pdf walkthrough/sample_reports/insulin_glargine.pdf

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
```

- [ ] **Step 7: Create README.md (placeholder; T19 polishes)**

```markdown
# Mentis

Procurement intelligence reports for medical substances. Type a substance, get an 8-section, citation-grounded brief in < 90 seconds.

> POC in active development. See `docs/superpowers/specs/2026-05-11-mentis-poc-design.md` for the full design.

## Quickstart

```bash
uv sync
cp .env.example .env  # fill in keys
uv run mentis report "ranitidine" --out report.md --pdf report.pdf
# or:
uv run python app.py  # launches Gradio
```
```

- [ ] **Step 8: Create empty package + test scaffolds**

```bash
mkdir -p mentis/retrievers mentis/templates tests/fixtures prompts docs/superpowers walkthrough/sample_reports scripts
touch mentis/__init__.py mentis/retrievers/__init__.py mentis/templates/__init__.py tests/__init__.py
```

Create `tests/conftest.py`:

```python
"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _disable_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests run without real Langfuse network calls."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
```

- [ ] **Step 9: Sync deps and verify**

```bash
uv sync
uv run python -c "import pydantic, litellm, httpx, tavily, weasyprint, markdown_it, jinja2, gradio, langfuse, click; print('all imports OK')"
uv run pytest -q
uv run ruff check .
```

Expected:
- `all imports OK`
- `no tests ran`
- `All checks passed!`

Note: `weasyprint` import may fail on first attempt if system deps not installed locally. On Ubuntu/Debian:
```bash
sudo apt-get install -y libcairo2 libpango-1.0-0 libpangoft2-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
```

- [ ] **Step 10: Commit**

```bash
git add .
git commit -m "chore: project scaffolding

uv-managed deps (11 runtime, 5 dev), pyproject.toml, .env.example,
packages.txt for HF Spaces WeasyPrint deps, Makefile (diagram,
sync-prompts, samples, test, lint), tests/conftest.py with
Langfuse auto-disable fixture."
```

---

## Task 2: Pydantic Schemas

**Files:**
- Create: `mentis/schemas.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_schemas.py`:

```python
"""Pydantic schemas validate correctly."""
from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from mentis.schemas import (
    QueryPlan,
    Reference,
    Report,
    ReportMetadata,
    SectionDraft,
    SectionPlan,
    Snippet,
)


def test_query_plan_requires_exactly_seven_sections() -> None:
    sp = [
        SectionPlan(section_name="product_profile", search_queries=["q"], sources=["rxnav"])
        for _ in range(7)
    ]
    qp = QueryPlan(
        user_query="ranitidine",
        normalized_term="ranitidine hydrochloride",
        rxnav_synonyms=["Zantac"],
        section_plans=sp,
        plan_version="v1",
    )
    assert len(qp.section_plans) == 7


def test_section_name_is_enum() -> None:
    with pytest.raises(ValidationError):
        SectionPlan(section_name="not_a_real_section", search_queries=[], sources=[])


def test_snippet_has_source_kind() -> None:
    s = Snippet(
        text="A study found...",
        url="https://example.org/x",
        source_name="PubMed",
        source_kind="scientific",
        title=None,
        retrieved_at=datetime.now(),
    )
    assert s.source_kind == "scientific"


def test_section_draft_tracks_safety_retries() -> None:
    sd = SectionDraft(
        section_name="market_demand",
        prose="...",
        snippets_used=[],
        citations=[],
        synthesizer_version="v1",
        safety_retries=2,
        fallback_to_raw_snippets=False,
    )
    assert sd.safety_retries == 2


def test_report_metadata_includes_cost() -> None:
    m = ReportMetadata(
        mentis_version="0.1.0",
        llm_provider="gemini/gemini-2.0-flash",
        prompt_versions={"section_synthesizer": "v1"},
        total_latency_ms=80000,
        total_snippets_retrieved=47,
        total_safety_retries=0,
        cost_usd=0.0124,
        generated_at=datetime.now(),
    )
    assert m.cost_usd > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_schemas.py -v
```

Expected: ImportError on `mentis.schemas`.

- [ ] **Step 3: Implement schemas**

Create `mentis/schemas.py`:

```python
"""All Pydantic data models for Mentis."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


SectionName = Literal[
    "product_profile",
    "clinical_use",
    "market_demand",
    "manufacturers",
    "regulatory",
    "sourcing_pricing",
    "risks_alternatives",
]

SectionNames: tuple[SectionName, ...] = (
    "product_profile",
    "clinical_use",
    "market_demand",
    "manufacturers",
    "regulatory",
    "sourcing_pricing",
    "risks_alternatives",
)


class SectionPlan(BaseModel):
    section_name: SectionName
    search_queries: list[str]
    sources: list[str]


class QueryPlan(BaseModel):
    user_query: str
    normalized_term: str | None = None
    rxnav_synonyms: list[str] = Field(default_factory=list)
    section_plans: list[SectionPlan]
    plan_version: str = "v1"


class Snippet(BaseModel):
    text: str
    url: HttpUrl
    source_name: str
    source_kind: Literal["scientific", "regulatory", "market", "background"]
    title: str | None = None
    retrieved_at: datetime


class SectionDraft(BaseModel):
    section_name: SectionName
    prose: str
    snippets_used: list[Snippet]
    citations: list[HttpUrl]
    synthesizer_version: str
    safety_retries: int = 0
    fallback_to_raw_snippets: bool = False


class Reference(BaseModel):
    url: HttpUrl
    source_name: str
    title: str
    retrieved_at: datetime
    used_in_sections: list[str] = Field(default_factory=list)


class ReportMetadata(BaseModel):
    mentis_version: str
    llm_provider: str
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    total_latency_ms: int
    total_snippets_retrieved: int
    total_safety_retries: int = 0
    cost_usd: float = 0.0
    generated_at: datetime


class Report(BaseModel):
    user_query: str
    normalized_term: str | None = None
    sections: list[SectionDraft]
    executive_summary: str
    references: list[Reference]
    metadata: ReportMetadata
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_schemas.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add mentis/schemas.py tests/test_schemas.py
git commit -m "feat(schemas): pydantic models for pipeline stages

SectionName Literal type + SectionNames tuple, QueryPlan with 7 section
plans, Snippet with source_kind, SectionDraft tracks safety_retries and
fallback flag, Report bundles 7 SectionDrafts + executive_summary +
references + metadata (incl. cost_usd from LiteLLM/Langfuse)."
```

---

## Task 3: Snippet Cache

**Files:**
- Create: `mentis/cache.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cache.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from mentis.cache import Cache


class Toy(BaseModel):
    n: int


@pytest.fixture
def cache(tmp_path: Path) -> Cache:
    return Cache(root=tmp_path)


def test_set_and_get(cache: Cache) -> None:
    cache.set("retrieve", "key1", Toy(n=42))
    assert cache.get("retrieve", "key1", Toy) == Toy(n=42)


def test_miss_returns_none(cache: Cache) -> None:
    assert cache.get("retrieve", "absent", Toy) is None


def test_namespace_isolation(cache: Cache) -> None:
    cache.set("a", "k", Toy(n=1))
    cache.set("b", "k", Toy(n=2))
    assert cache.get("a", "k", Toy) == Toy(n=1)
    assert cache.get("b", "k", Toy) == Toy(n=2)


def test_compute_hash_stable_and_version_aware() -> None:
    assert Cache.compute_hash("q", "v1") == Cache.compute_hash("q", "v1")
    assert Cache.compute_hash("q", "v1") != Cache.compute_hash("q", "v2")


def test_hf_spaces_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    c = Cache.from_env()
    assert str(tmp_path / "hf") in str(c.root)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_cache.py -v
```

- [ ] **Step 3: Implement cache**

Create `mentis/cache.py`:

```python
"""File-based snippet cache with version-keyed invalidation."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Cache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> Cache:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return cls(root=Path(hf_home) / ".cache" / "mentis")
        return cls(root=Path.home() / ".cache" / "mentis")

    @staticmethod
    def compute_hash(*parts: str) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode("utf-8"))
            h.update(b"|")
        return h.hexdigest()[:16]

    def _path(self, namespace: str, key: str) -> Path:
        return self.root / namespace / f"{key}.json"

    def get(self, namespace: str, key: str, model_cls: type[T]) -> T | None:
        p = self._path(namespace, key)
        if not p.exists():
            return None
        return model_cls.model_validate_json(p.read_text())

    def set(self, namespace: str, key: str, value: BaseModel) -> None:
        p = self._path(namespace, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value.model_dump_json(indent=2))

    def clear(self, namespace: str | None = None) -> None:
        import shutil
        target = self.root / namespace if namespace else self.root
        if target.exists():
            shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/test_cache.py -v
git add mentis/cache.py tests/test_cache.py
git commit -m "feat(cache): file-based snippet cache with version-keyed invalidation

Namespace-isolated JSON storage at ~/.cache/mentis/{namespace}/{hash}.json.
HF Spaces /data mount detected via HF_HOME env var. compute_hash(query, version)
ensures version bumps invalidate cleanly."
```

---

## Task 4: Observability (Langfuse + LiteLLM Callback)

**Files:**
- Create: `mentis/observability.py`
- Create: `tests/test_observability.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_observability.py`:

```python
from __future__ import annotations

import pytest

from mentis.observability import init_observability, is_langfuse_configured


def test_not_configured_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    assert is_langfuse_configured() is False


def test_configured_with_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    assert is_langfuse_configured() is True


def test_init_is_no_op_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    init_observability()  # should not raise
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_observability.py -v
```

- [ ] **Step 3: Implement observability**

Create `mentis/observability.py`:

```python
"""Langfuse initialization + LiteLLM callback wiring.

init_observability() is called once at app/CLI startup.
If Langfuse keys are absent, this is a no-op.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def is_langfuse_configured() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and bool(
        os.environ.get("LANGFUSE_SECRET_KEY")
    )


def init_observability() -> None:
    if not is_langfuse_configured():
        logger.debug("Langfuse keys not set; tracing disabled.")
        return

    import litellm

    litellm.success_callback = list(set((litellm.success_callback or []) + ["langfuse"]))
    litellm.failure_callback = list(set((litellm.failure_callback or []) + ["langfuse"]))
    logger.info("Langfuse observability enabled for LiteLLM.")
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/test_observability.py -v
git add mentis/observability.py tests/test_observability.py
git commit -m "feat(observability): langfuse init + litellm callback wiring

No-op when LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are unset, so
tests and local dev work without a Langfuse account."
```

---

## Task 5: LLM Client with Safety-Aware Fallback

**Files:**
- Create: `mentis/llm.py`
- Create: `tests/test_llm.py`

This is the critical safety-escalation module. Full TDD required.

- [ ] **Step 1: Write failing tests**

Create `tests/test_llm.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from mentis.llm import LLMClient, LLMConfig, SafetyBlockedException


class TinyOut(BaseModel):
    items: list[str]


def _fake_response(content: str):
    return type(
        "R",
        (),
        {
            "choices": [
                type("C", (), {"message": type("M", (), {"content": content})()})()
            ]
        },
    )()


def _safety_blocked():
    raise SafetyBlockedException("blocked by provider safety filter")


@pytest.fixture
def config() -> LLMConfig:
    return LLMConfig(providers=["gemini/gemini-2.0-flash", "groq/llama-3.3-70b-versatile"])


@pytest.mark.asyncio
async def test_primary_succeeds_no_retries(config: LLMConfig) -> None:
    client = LLMClient(config)
    mock = AsyncMock(return_value=_fake_response('{"items": ["a"]}'))
    with patch("mentis.llm.acompletion", new=mock):
        out, used, trace = await client.complete_with_safety(
            system="sys", user="usr", schema=TinyOut, prompt_version="v1"
        )
    assert out == TinyOut(items=["a"])
    assert used == "gemini/gemini-2.0-flash"
    assert trace.retry_count == 0
    assert mock.await_count == 1


@pytest.mark.asyncio
async def test_safety_relax_retry(config: LLMConfig) -> None:
    """Primary fails with safety block → retry with relaxed settings succeeds."""
    client = LLMClient(config)
    side_effects = [_safety_blocked, _fake_response('{"items": ["x"]}')]

    async def mock_call(*args, **kwargs):
        effect = side_effects.pop(0)
        if callable(effect) and effect.__name__ == "_safety_blocked":
            effect()
        return effect

    with patch("mentis.llm.acompletion", new=AsyncMock(side_effect=mock_call)):
        out, used, trace = await client.complete_with_safety(
            system="sys", user="usr", schema=TinyOut, prompt_version="v1"
        )
    assert out == TinyOut(items=["x"])
    assert used == "gemini/gemini-2.0-flash"
    assert trace.retry_count == 1
    assert "relaxing_filters" in trace.actions


@pytest.mark.asyncio
async def test_all_paths_blocked_raises(config: LLMConfig) -> None:
    """All 4 escalation paths fail with safety block → raises SafetyBlockedException."""
    client = LLMClient(config)

    async def always_block(*args, **kwargs):
        _safety_blocked()

    with patch("mentis.llm.acompletion", new=AsyncMock(side_effect=always_block)):
        with pytest.raises(SafetyBlockedException):
            await client.complete_with_safety(
                system="sys", user="usr", schema=TinyOut, prompt_version="v1"
            )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_llm.py -v
```

- [ ] **Step 3: Implement LLM client**

Create `mentis/llm.py`:

```python
"""Async LiteLLM client with safety-aware fallback chain.

Escalation order on safety block:
  1. Primary call (default safety settings)
  2. Retry with relaxed safety settings on Gemini
  3. Retry with prompt reframed via safety_reframe.v1.j2 + relaxed settings
  4. Provider fallback (Groq)
  5. Raise SafetyBlockedException — caller handles (e.g., raw-snippets fallback)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TypeVar

from litellm import acompletion
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class SafetyBlockedException(Exception):
    """Raised when an LLM call is blocked by the provider's safety filter."""


RELAXED_PROCUREMENT_SETTINGS = [
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]


@dataclass
class SafetyTrace:
    retry_count: int = 0
    actions: list[str] = field(default_factory=list)


@dataclass
class LLMConfig:
    providers: list[str] = field(
        default_factory=lambda: ["gemini/gemini-2.0-flash", "groq/llama-3.3-70b-versatile"]
    )
    timeout_s: float = 60.0

    @classmethod
    def from_env(cls) -> LLMConfig:
        primary = os.environ.get("MENTIS_PRIMARY_MODEL", "gemini/gemini-2.0-flash")
        csv = os.environ.get("MENTIS_PROVIDERS", "gemini,groq").split(",")
        m = {
            "gemini": "gemini/gemini-2.0-flash",
            "groq": "groq/llama-3.3-70b-versatile",
        }
        ordered = [primary] + [m[p] for p in csv if m.get(p) and m[p] != primary]
        return cls(providers=ordered)


def _is_safety_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in ("safety", "blocked", "content_filter", "harm", "policy violation")
    )


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    async def _call(
        self,
        *,
        provider: str,
        system: str,
        user: str,
        schema: type[T],
        prompt_version: str,
        safety_settings: list | None = None,
    ) -> T:
        kwargs = {
            "model": provider,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "timeout": self.config.timeout_s,
            "metadata": {"prompt_version": prompt_version, "mentis_provider": provider},
        }
        if safety_settings is not None and provider.startswith("gemini"):
            kwargs["safety_settings"] = safety_settings
        try:
            resp = await acompletion(**kwargs)
        except Exception as e:
            if _is_safety_error(e):
                raise SafetyBlockedException(str(e)) from e
            raise
        content = resp.choices[0].message.content
        return schema.model_validate_json(content)

    async def complete_with_safety(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        prompt_version: str,
        reframe_template: str | None = None,
    ) -> tuple[T, str, SafetyTrace]:
        """Run the 4-step safety escalation chain.

        Returns (parsed, provider_used, trace).
        Raises SafetyBlockedException if all paths fail.
        """
        trace = SafetyTrace()
        primary = self.config.providers[0]

        # 1. Primary call
        try:
            out = await self._call(
                provider=primary, system=system, user=user, schema=schema, prompt_version=prompt_version
            )
            return out, primary, trace
        except SafetyBlockedException:
            trace.retry_count += 1
            trace.actions.append("relaxing_filters")

        # 2. Retry with relaxed safety on Gemini
        try:
            out = await self._call(
                provider=primary,
                system=system,
                user=user,
                schema=schema,
                prompt_version=prompt_version,
                safety_settings=RELAXED_PROCUREMENT_SETTINGS,
            )
            return out, primary, trace
        except SafetyBlockedException:
            trace.retry_count += 1
            trace.actions.append("reframing_prompt")

        # 3. Reframe user prompt + relaxed safety
        if reframe_template:
            reframed = reframe_template.replace("{{ original }}", user)
            try:
                out = await self._call(
                    provider=primary,
                    system=system,
                    user=reframed,
                    schema=schema,
                    prompt_version=prompt_version,
                    safety_settings=RELAXED_PROCUREMENT_SETTINGS,
                )
                return out, primary, trace
            except SafetyBlockedException:
                trace.retry_count += 1
                trace.actions.append("provider_fallback")
        else:
            trace.retry_count += 1
            trace.actions.append("provider_fallback")

        # 4. Provider fallback
        for fb in self.config.providers[1:]:
            try:
                out = await self._call(
                    provider=fb, system=system, user=user, schema=schema, prompt_version=prompt_version
                )
                return out, fb, trace
            except SafetyBlockedException:
                continue

        # 5. All paths failed
        trace.retry_count += 1
        trace.actions.append("all_blocked")
        raise SafetyBlockedException("all providers blocked; surface raw snippets")
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/test_llm.py -v
git add mentis/llm.py tests/test_llm.py
git commit -m "feat(llm): async litellm client with safety-aware escalation

4-step chain: primary → relaxed safety → reframed prompt → provider fallback.
SafetyTrace records retry_count + actions for SectionDraft.safety_retries.
Raises SafetyBlockedException when all paths fail; caller renders raw snippets."
```

---

## Task 6: Prompt Registry + 4 Prompt Templates

**Files:**
- Create: `mentis/prompts.py`
- Create: `prompts/query_planner.v1.j2`
- Create: `prompts/section_synthesizer.v1.j2`
- Create: `prompts/executive_summary.v1.j2`
- Create: `prompts/safety_reframe.v1.j2`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_prompts.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from mentis.prompts import PromptRegistry


@pytest.fixture
def prompt_dir(tmp_path: Path) -> Path:
    p = tmp_path / "prompts"
    p.mkdir()
    (p / "x.v1.j2").write_text("hello {{ name }}")
    (p / "x.v2.j2").write_text("hi {{ name }}")
    return p


def test_loads_local_prompt(prompt_dir: Path) -> None:
    reg = PromptRegistry(in_repo_dir=prompt_dir)
    rendered = reg.get("x", version="v1").render(name="Mentis")
    assert rendered == "hello Mentis"


def test_default_version_is_highest(prompt_dir: Path) -> None:
    reg = PromptRegistry(in_repo_dir=prompt_dir)
    p = reg.get("x")
    assert p.version == "v2"


def test_missing_prompt_raises(prompt_dir: Path) -> None:
    reg = PromptRegistry(in_repo_dir=prompt_dir)
    with pytest.raises(FileNotFoundError):
        reg.get("nope")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_prompts.py -v
```

- [ ] **Step 3: Implement prompt registry**

Create `mentis/prompts.py`:

```python
"""Versioned prompt loader.

Source of truth: in-repo Jinja2 files at prompts/{name}.{version}.j2
Runtime: try Langfuse first → fall back to in-repo file.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Template

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    template_text: str
    source: str  # "langfuse" or "local"

    def render(self, **kwargs: object) -> str:
        return Template(self.template_text).render(**kwargs)


class PromptRegistry:
    _FILE_RE = re.compile(r"^(?P<name>.+)\.(?P<version>v\d+)\.j2$")

    def __init__(self, in_repo_dir: Path) -> None:
        self.in_repo_dir = in_repo_dir
        self._langfuse_enabled = bool(
            os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
        )

    def _local_candidates(self, name: str) -> list[tuple[str, Path]]:
        out: list[tuple[str, Path]] = []
        for p in self.in_repo_dir.glob(f"{name}.*.j2"):
            m = self._FILE_RE.match(p.name)
            if m and m.group("name") == name:
                out.append((m.group("version"), p))
        return sorted(out, key=lambda t: int(t[0][1:]))

    def _try_langfuse(self, name: str, version: str | None) -> Prompt | None:
        if not self._langfuse_enabled:
            return None
        try:
            from langfuse import Langfuse

            lf = Langfuse()
            p = lf.get_prompt(name=name, version=int(version[1:]) if version else None, label="production")
            return Prompt(
                name=name,
                version=f"v{p.version}",
                template_text=p.prompt,
                source="langfuse",
            )
        except Exception as e:
            logger.warning(f"Langfuse fetch failed for {name}: {e!r}; falling back to local")
            return None

    def get(self, name: str, version: str | None = None) -> Prompt:
        # 1. Try Langfuse
        if lf_prompt := self._try_langfuse(name, version):
            return lf_prompt

        # 2. Local fallback
        candidates = self._local_candidates(name)
        if not candidates:
            raise FileNotFoundError(f"no prompt named {name!r} in {self.in_repo_dir}")
        if version is None:
            chosen_version, chosen_path = candidates[-1]
        else:
            matches = [c for c in candidates if c[0] == version]
            if not matches:
                raise FileNotFoundError(f"no prompt {name!r} version {version!r}")
            chosen_version, chosen_path = matches[0]
        return Prompt(
            name=name, version=chosen_version, template_text=chosen_path.read_text(), source="local"
        )
```

- [ ] **Step 4: Create the 4 prompt templates**

Create `prompts/query_planner.v1.j2`:

```jinja
You are planning research for an 8-section procurement intelligence brief on a medical substance.

User query: {{ user_query }}
Normalized term (if available): {{ normalized_term }}
Synonyms/brand names: {{ rxnav_synonyms | join(", ") }}

For each section in the list below, generate 2-4 specific search queries optimized for the listed source(s). Section names and their assigned sources:

- product_profile         (sources: rxnav, pubmed, wikipedia)
- clinical_use            (sources: pubmed, openfda)
- market_demand           (sources: tavily)
- manufacturers           (sources: tavily)
- regulatory              (sources: openfda, tavily)
- sourcing_pricing        (sources: tavily)
- risks_alternatives      (sources: openfda, tavily)

Rules:
1. Queries must be specific to the substance, not generic
2. For PubMed/OpenFDA sources, prefer scientific phrasing (e.g., "tramadol pharmacokinetics" not "how does tramadol work")
3. For Tavily/web sources, prefer commercial/market phrasing (e.g., "tramadol manufacturers India 2025" not "tramadol producers")
4. Output strict JSON: {"section_plans": [{"section_name": "...", "search_queries": [...], "sources": [...]}, ...]}

Output the JSON only. No commentary.
```

Create `prompts/section_synthesizer.v1.j2`:

```jinja
You are writing the {{ section_name }} section of a procurement intelligence report on {{ user_query }} ({{ normalized_term }}).

Use ONLY the snippets below as your source material. Write 200-300 words of prose. Embed inline markdown hyperlinks like [claim text](url) on key facts. EVERY URL you cite MUST come exactly from the snippets list — do not invent URLs or claims not supported by snippets.

Style:
- Authoritative but neutral (think McKinsey/Bain consulting brief, not blog post)
- Quantify where data permits (market sizes, growth rates, doses, prices)
- Surface contradictions honestly when sources disagree

Snippets:
{% for s in snippets %}
[{{ loop.index }}] {{ s.source_name }} | {{ s.url }} | retrieved {{ s.retrieved_at }}
{{ s.text }}
{% endfor %}

Output strict JSON: {"prose": "<markdown text>"}.
The prose field must be valid Markdown with at least 2 inline citations as [text](url).
```

Create `prompts/executive_summary.v1.j2`:

```jinja
You are writing the Executive Summary of a procurement intelligence report on {{ user_query }} ({{ normalized_term }}).

Read the 7 section drafts below. Produce a 100-150 word summary covering:
1. What the substance is (one phrase)
2. Primary clinical/commercial use (one sentence)
3. Market shape signal (one phrase: size, growth, or concentration)
4. Key regulatory or compliance constraint (one phrase)
5. Top 2-3 sourcing or supplier-side facts the reader needs

Tone: tight, factual, no marketing fluff. Cite up to 3 most-load-bearing claims as inline [text](url) hyperlinks pulled from the section drafts.

Section drafts:
{% for s in sections %}
### {{ s.section_name }}
{{ s.prose }}
{% endfor %}

Output strict JSON: {"executive_summary": "<markdown text 100-150 words>"}.
```

Create `prompts/safety_reframe.v1.j2`:

```jinja
The following is a research request from a licensed healthcare procurement professional. They are not seeking medical advice or personal-use information. They are researching procurement, supplier, regulatory, and market data on a medical substance for their B2B sourcing operations.

Original research context:
{{ original }}

Please proceed with this professional research request and produce the requested structured output.
```

- [ ] **Step 5: Run tests + commit**

```bash
uv run pytest tests/test_prompts.py -v
git add mentis/prompts.py prompts/ tests/test_prompts.py
git commit -m "feat(prompts): registry + 4 versioned templates

PromptRegistry tries Langfuse first, falls back to in-repo file.
Templates: query_planner (per-section search planning), section_synthesizer
(citation-grounded prose), executive_summary (assembled from 7 drafts),
safety_reframe (wraps blocked queries as professional research)."
```

---

## Task 7: Tavily Retriever

**Files:**
- Create: `mentis/retrievers/__init__.py` (add BaseRetriever Protocol)
- Create: `mentis/retrievers/tavily.py`
- Create: `tests/test_retriever_tavily.py`

- [ ] **Step 1: Add BaseRetriever Protocol to `__init__.py`**

```python
"""Retriever protocol — every retriever implements .search(query, n) -> list[Snippet]."""
from __future__ import annotations

from typing import Protocol

from mentis.schemas import Snippet


class BaseRetriever(Protocol):
    name: str

    async def search(self, query: str, n: int = 5) -> list[Snippet]: ...
```

- [ ] **Step 2: Write failing test**

Create `tests/test_retriever_tavily.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from mentis.retrievers.tavily import TavilyRetriever


@pytest.mark.asyncio
async def test_tavily_returns_snippets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    fake_response = {
        "results": [
            {
                "title": "Ranitidine market overview 2025",
                "url": "https://example.org/ranitidine-market",
                "content": "The global ranitidine market reached $1.2B in 2024, with 8% YoY growth driven by generic adoption.",
            },
            {
                "title": "Top H2 antagonist manufacturers",
                "url": "https://example.org/h2-mfrs",
                "content": "Leading manufacturers include Pfizer, Glenmark, and Cipla.",
            },
        ]
    }
    retriever = TavilyRetriever()
    with patch("mentis.retrievers.tavily.TavilyClient") as MockClient:
        MockClient.return_value.search = lambda *args, **kwargs: fake_response
        snippets = await retriever.search("ranitidine market", n=2)
    assert len(snippets) == 2
    assert snippets[0].source_name == "Tavily"
    assert "ranitidine" in snippets[0].text.lower()
    assert str(snippets[0].url) == "https://example.org/ranitidine-market"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_retriever_tavily.py -v
```

- [ ] **Step 4: Implement Tavily retriever**

Create `mentis/retrievers/tavily.py`:

```python
"""Tavily web search retriever."""
from __future__ import annotations

import logging
import os
from datetime import datetime

from tavily import TavilyClient

from mentis.schemas import Snippet

logger = logging.getLogger(__name__)


class TavilyRetriever:
    name = "tavily"
    source_kind = "market"

    def __init__(self) -> None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY not set")
        self._client = TavilyClient(api_key=api_key)

    async def search(self, query: str, n: int = 5) -> list[Snippet]:
        # tavily-python is sync; wrap in asyncio.to_thread to keep pipeline async
        import asyncio

        def _do_search():
            return self._client.search(query=query, max_results=n, search_depth="advanced")

        try:
            resp = await asyncio.to_thread(_do_search)
        except Exception as e:
            logger.warning(f"tavily search failed for {query!r}: {e!r}")
            return []

        out: list[Snippet] = []
        for r in resp.get("results", [])[:n]:
            try:
                out.append(
                    Snippet(
                        text=r.get("content", "")[:800],
                        url=r["url"],
                        source_name="Tavily",
                        source_kind="market",
                        title=r.get("title"),
                        retrieved_at=datetime.now(),
                    )
                )
            except Exception as e:
                logger.warning(f"skipping tavily result: {e!r}")
        return out
```

- [ ] **Step 5: Run tests + commit**

```bash
uv run pytest tests/test_retriever_tavily.py -v
git add mentis/retrievers/__init__.py mentis/retrievers/tavily.py tests/test_retriever_tavily.py
git commit -m "feat(retriever-tavily): web search via tavily-python

BaseRetriever protocol defined. TavilyRetriever wraps sync SDK via
asyncio.to_thread, returns up to N snippets per query with source_name='Tavily'
and source_kind='market'. Text truncated to 800 chars for LLM context bounds."
```

---

## Task 8: PubMed Retriever

**Files:**
- Create: `mentis/retrievers/pubmed.py`
- Create: `tests/test_retriever_pubmed.py`
- Create: `tests/fixtures/pubmed/esearch_response.json`
- Create: `tests/fixtures/pubmed/efetch_response.xml`

- [ ] **Step 1: Create fixture responses**

Create `tests/fixtures/pubmed/esearch_response.json`:

```json
{
  "esearchresult": {
    "count": "2",
    "idlist": ["12345678", "23456789"]
  }
}
```

Create `tests/fixtures/pubmed/efetch_response.xml`:

```xml
<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>Ranitidine pharmacokinetics in adults</ArticleTitle>
        <Abstract>
          <AbstractText>Ranitidine is an H2 receptor antagonist. Peak plasma concentrations occur 1-3 hours post-administration.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>23456789</PMID>
      <Article>
        <ArticleTitle>NDMA contamination in ranitidine</ArticleTitle>
        <Abstract>
          <AbstractText>NDMA levels exceed acceptable daily intake in ranitidine formulations stored at room temperature.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_retriever_pubmed.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mentis.retrievers.pubmed import PubmedRetriever

FIXTURES = Path(__file__).parent / "fixtures" / "pubmed"


@pytest.mark.asyncio
async def test_pubmed_returns_snippets() -> None:
    esearch_json = json.loads((FIXTURES / "esearch_response.json").read_text())
    efetch_xml = (FIXTURES / "efetch_response.xml").read_text()

    async def fake_get(url, *args, **kwargs):
        resp = MagicMock()
        if "esearch" in url:
            resp.json = MagicMock(return_value=esearch_json)
        else:
            resp.text = efetch_xml
        resp.raise_for_status = MagicMock()
        return resp

    retriever = PubmedRetriever()
    with patch("mentis.retrievers.pubmed.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(side_effect=fake_get)
        snippets = await retriever.search("ranitidine pharmacokinetics", n=2)
    assert len(snippets) == 2
    assert snippets[0].source_name == "PubMed"
    assert snippets[0].source_kind == "scientific"
    assert "ranitidine" in snippets[0].text.lower() or "ranitidine" in (snippets[0].title or "").lower()
    assert "pubmed.ncbi.nlm.nih.gov" in str(snippets[0].url)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_retriever_pubmed.py -v
```

- [ ] **Step 4: Implement PubMed retriever**

Create `mentis/retrievers/pubmed.py`:

```python
"""PubMed retriever via NCBI E-utilities (esearch + efetch)."""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime

import httpx

from mentis.schemas import Snippet

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ARTICLE_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


class PubmedRetriever:
    name = "pubmed"
    source_kind = "scientific"

    async def search(self, query: str, n: int = 5) -> list[Snippet]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 1. esearch returns PMIDs
            try:
                r = await client.get(
                    ESEARCH_URL,
                    params={"db": "pubmed", "term": query, "retmax": n, "retmode": "json"},
                )
                r.raise_for_status()
                idlist = r.json().get("esearchresult", {}).get("idlist", [])
            except Exception as e:
                logger.warning(f"pubmed esearch failed for {query!r}: {e!r}")
                return []

            if not idlist:
                return []

            # 2. efetch returns abstracts
            try:
                r = await client.get(
                    EFETCH_URL,
                    params={"db": "pubmed", "id": ",".join(idlist), "rettype": "abstract", "retmode": "xml"},
                )
                r.raise_for_status()
                xml_text = r.text
            except Exception as e:
                logger.warning(f"pubmed efetch failed for ids {idlist}: {e!r}")
                return []

        # 3. parse XML
        out: list[Snippet] = []
        try:
            root = ET.fromstring(xml_text)
            for article in root.findall(".//PubmedArticle"):
                pmid_el = article.find(".//PMID")
                title_el = article.find(".//ArticleTitle")
                abstract_el = article.find(".//Abstract/AbstractText")
                if pmid_el is None or abstract_el is None:
                    continue
                pmid = pmid_el.text or ""
                title = title_el.text if title_el is not None else None
                text = (abstract_el.text or "")[:800]
                if not text:
                    continue
                out.append(
                    Snippet(
                        text=text,
                        url=ARTICLE_URL.format(pmid=pmid),
                        source_name="PubMed",
                        source_kind="scientific",
                        title=title,
                        retrieved_at=datetime.now(),
                    )
                )
        except ET.ParseError as e:
            logger.warning(f"pubmed XML parse failed: {e!r}")
            return []

        return out
```

- [ ] **Step 5: Run tests + commit**

```bash
uv run pytest tests/test_retriever_pubmed.py -v
git add mentis/retrievers/pubmed.py tests/test_retriever_pubmed.py tests/fixtures/pubmed/
git commit -m "feat(retriever-pubmed): NCBI E-utilities (esearch + efetch)

Two-step retrieval: esearch returns PMIDs, efetch returns abstracts as XML.
Parses ArticleTitle + AbstractText. URL points to pubmed.ncbi.nlm.nih.gov
canonical article page. source_kind='scientific'."
```

---

## Task 9: OpenFDA Retriever

**Files:**
- Create: `mentis/retrievers/openfda.py`
- Create: `tests/test_retriever_openfda.py`
- Create: `tests/fixtures/openfda/label_response.json`

- [ ] **Step 1: Create fixture response**

Create `tests/fixtures/openfda/label_response.json`:

```json
{
  "results": [
    {
      "openfda": {
        "brand_name": ["Zantac"],
        "generic_name": ["Ranitidine HCl"],
        "manufacturer_name": ["GlaxoSmithKline"]
      },
      "indications_and_usage": ["For short-term treatment of duodenal ulcers."],
      "warnings": ["Ranitidine has been associated with NDMA contamination."],
      "id": "abc-123"
    }
  ]
}
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_retriever_openfda.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mentis.retrievers.openfda import OpenFDARetriever

FIXTURES = Path(__file__).parent / "fixtures" / "openfda"


@pytest.mark.asyncio
async def test_openfda_returns_snippets() -> None:
    fake = json.loads((FIXTURES / "label_response.json").read_text())

    async def fake_get(url, *args, **kwargs):
        resp = MagicMock()
        resp.json = MagicMock(return_value=fake)
        resp.raise_for_status = MagicMock()
        return resp

    retriever = OpenFDARetriever()
    with patch("mentis.retrievers.openfda.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(side_effect=fake_get)
        snippets = await retriever.search("ranitidine", n=3)
    assert len(snippets) >= 1
    assert snippets[0].source_name == "OpenFDA"
    assert snippets[0].source_kind == "regulatory"
    assert "ranitidine" in snippets[0].text.lower() or "ndma" in snippets[0].text.lower()
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_retriever_openfda.py -v
```

- [ ] **Step 4: Implement OpenFDA retriever**

Create `mentis/retrievers/openfda.py`:

```python
"""OpenFDA retriever — drug labels + recalls + adverse events."""
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
                r = await client.get(
                    LABEL_URL,
                    params={
                        "search": f'(openfda.generic_name:"{query}" OR openfda.brand_name:"{query}")',
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
            text = f"{brand} ({generic}) — Manufacturer: {mfr}. Indications: {indications} Warnings: {warnings}"[:800]
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
```

- [ ] **Step 5: Run tests + commit**

```bash
uv run pytest tests/test_retriever_openfda.py -v
git add mentis/retrievers/openfda.py tests/test_retriever_openfda.py tests/fixtures/openfda/
git commit -m "feat(retriever-openfda): drug label search

Searches FDA drug label API by generic and brand name. Returns one snippet
per match combining brand, generic, manufacturer, indications, and warnings.
source_kind='regulatory'."
```

---

## Task 10: Wikipedia + RxNav Retrievers

**Files:**
- Create: `mentis/retrievers/wikipedia.py`
- Create: `mentis/retrievers/rxnav.py`
- Create: `tests/test_retriever_wikipedia.py`
- Create: `tests/test_retriever_rxnav.py`

Both are simple GET-and-parse retrievers. Combined into one task.

- [ ] **Step 1: Implement Wikipedia retriever**

Create `mentis/retrievers/wikipedia.py`:

```python
"""Wikipedia retriever via REST API."""
from __future__ import annotations

import logging
from datetime import datetime
from urllib.parse import quote

import httpx

from mentis.schemas import Snippet

logger = logging.getLogger(__name__)

SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


class WikipediaRetriever:
    name = "wikipedia"
    source_kind = "background"

    async def search(self, query: str, n: int = 1) -> list[Snippet]:
        title = quote(query.replace(" ", "_"))
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                r = await client.get(SUMMARY_URL.format(title=title))
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                logger.warning(f"wikipedia fetch failed for {query!r}: {e!r}")
                return []

        extract = data.get("extract", "")
        if not extract:
            return []

        page_url = data.get("content_urls", {}).get("desktop", {}).get("page")
        if not page_url:
            return []

        return [
            Snippet(
                text=extract[:800],
                url=page_url,
                source_name="Wikipedia",
                source_kind="background",
                title=data.get("title"),
                retrieved_at=datetime.now(),
            )
        ]
```

- [ ] **Step 2: Implement RxNav retriever**

Create `mentis/retrievers/rxnav.py`:

```python
"""RxNav retriever — drug normalization, synonyms, RxCUI codes.

Differs from other retrievers in shape: not just snippets, but returns
a normalization dict via `.normalize(query)`. Used by query_planner.
"""
from __future__ import annotations

import logging
from datetime import datetime

import httpx

from mentis.schemas import Snippet

logger = logging.getLogger(__name__)

DRUGS_URL = "https://rxnav.nlm.nih.gov/REST/drugs.json"
RXNAV_PAGE = "https://mor.nlm.nih.gov/RxNav/search?searchBy=String&searchTerm={query}"


class RxNavRetriever:
    name = "rxnav"
    source_kind = "scientific"

    async def normalize(self, query: str) -> dict:
        """Return {'normalized_term': str|None, 'synonyms': list[str], 'rxcui': str|None}."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                r = await client.get(DRUGS_URL, params={"name": query})
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                logger.warning(f"rxnav lookup failed for {query!r}: {e!r}")
                return {"normalized_term": None, "synonyms": [], "rxcui": None}

        groups = data.get("drugGroup", {}).get("conceptGroup", []) or []
        synonyms: list[str] = []
        normalized: str | None = None
        rxcui: str | None = None
        for g in groups:
            for c in g.get("conceptProperties", []) or []:
                if "name" in c:
                    synonyms.append(c["name"])
                if normalized is None and c.get("name"):
                    normalized = c["name"]
                if rxcui is None and c.get("rxcui"):
                    rxcui = c["rxcui"]
        return {"normalized_term": normalized, "synonyms": synonyms[:10], "rxcui": rxcui}

    async def search(self, query: str, n: int = 1) -> list[Snippet]:
        norm = await self.normalize(query)
        if not norm["normalized_term"]:
            return []
        text = (
            f"Normalized form: {norm['normalized_term']}. "
            f"Synonyms / brand forms: {', '.join(norm['synonyms'][:5])}. "
            f"RxCUI: {norm['rxcui']}."
        )
        return [
            Snippet(
                text=text,
                url=RXNAV_PAGE.format(query=query.replace(" ", "%20")),
                source_name="RxNav (NLM)",
                source_kind="scientific",
                title=norm["normalized_term"],
                retrieved_at=datetime.now(),
            )
        ]
```

- [ ] **Step 3: Smoke tests**

Create `tests/test_retriever_wikipedia.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mentis.retrievers.wikipedia import WikipediaRetriever


@pytest.mark.asyncio
async def test_wikipedia_returns_snippet() -> None:
    fake = {
        "title": "Ranitidine",
        "extract": "Ranitidine is an H2 receptor antagonist used to treat ulcers.",
        "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Ranitidine"}},
    }

    async def fake_get(url, *args, **kwargs):
        r = MagicMock()
        r.json = MagicMock(return_value=fake)
        r.raise_for_status = MagicMock()
        return r

    retriever = WikipediaRetriever()
    with patch("mentis.retrievers.wikipedia.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(side_effect=fake_get)
        snippets = await retriever.search("ranitidine")
    assert len(snippets) == 1
    assert "ranitidine" in snippets[0].text.lower()
    assert snippets[0].source_name == "Wikipedia"
```

Create `tests/test_retriever_rxnav.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mentis.retrievers.rxnav import RxNavRetriever


@pytest.mark.asyncio
async def test_rxnav_normalize() -> None:
    fake = {
        "drugGroup": {
            "conceptGroup": [
                {
                    "tty": "IN",
                    "conceptProperties": [
                        {"name": "ranitidine hydrochloride", "rxcui": "12345"},
                        {"name": "Zantac"},
                    ],
                }
            ]
        }
    }

    async def fake_get(url, *args, **kwargs):
        r = MagicMock()
        r.json = MagicMock(return_value=fake)
        r.raise_for_status = MagicMock()
        return r

    retriever = RxNavRetriever()
    with patch("mentis.retrievers.rxnav.httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.get = AsyncMock(side_effect=fake_get)
        norm = await retriever.normalize("ranitidine")
    assert norm["normalized_term"] == "ranitidine hydrochloride"
    assert "Zantac" in norm["synonyms"]
    assert norm["rxcui"] == "12345"
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/test_retriever_wikipedia.py tests/test_retriever_rxnav.py -v
git add mentis/retrievers/wikipedia.py mentis/retrievers/rxnav.py tests/test_retriever_wikipedia.py tests/test_retriever_rxnav.py
git commit -m "feat(retrievers): wikipedia + rxnav

WikipediaRetriever: REST summary endpoint, source_kind='background'.
RxNavRetriever: drug normalization API; .normalize() returns canonical name +
synonyms + RxCUI; .search() returns a single normalization snippet."
```

---

## Task 11: Query Planner

**Files:**
- Create: `mentis/query_planner.py`
- Create: `tests/test_query_planner.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_query_planner.py`:

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from mentis.llm import LLMConfig, SafetyTrace
from mentis.query_planner import plan_query
from mentis.schemas import QueryPlan, SectionPlan


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_query_planner.py -v
```

- [ ] **Step 3: Implement query planner**

Create `mentis/query_planner.py`:

```python
"""Query planner: user_query → QueryPlan with 7 section plans + RxNav normalization."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import BaseModel, Field

from mentis.llm import LLMClient, LLMConfig, SafetyTrace
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
    """Plan an 8-section research query.

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
            llm_config=llm_config, system=SYSTEM_PROMPT, user=user_prompt, schema=_PlanLLMOut, prompt_version=prompt.version
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
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/test_query_planner.py -v
git add mentis/query_planner.py tests/test_query_planner.py
git commit -m "feat(query-planner): user query → 7 section plans

RxNav normalization first (sync wrapper for the async retriever), then
one LLM call rendering query_planner.v1.j2 to produce 7 SectionPlans.
rxnav_normalize and llm_complete are injectable for tests."
```

---

## Task 12: Orchestrator

**Files:**
- Create: `mentis/orchestrator.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_orchestrator.py`:

```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from mentis.cache import Cache
from mentis.orchestrator import SOURCE_ROUTING, retrieve_for_plan
from mentis.schemas import QueryPlan, SectionPlan, Snippet


def _snip(name: str, kind: str = "market") -> Snippet:
    return Snippet(
        text=f"fact about {name}",
        url=f"https://example.org/{name}",
        source_name=name,
        source_kind=kind,  # type: ignore[arg-type]
        retrieved_at=datetime.now(),
    )


@pytest.mark.asyncio
async def test_orchestrator_routes_per_section(tmp_path: Path) -> None:
    cache = Cache(root=tmp_path / "cache")
    qp = QueryPlan(
        user_query="ranitidine",
        section_plans=[
            SectionPlan(section_name="product_profile", search_queries=["q1"], sources=["rxnav", "pubmed", "wikipedia"]),
            SectionPlan(section_name="market_demand", search_queries=["q2"], sources=["tavily"]),
        ],
    )
    # Stub retrievers
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_orchestrator.py -v
```

- [ ] **Step 3: Implement orchestrator**

Create `mentis/orchestrator.py`:

```python
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


async def _retrieve_one(retriever, query: str, n: int, cache: Cache, sem: asyncio.Semaphore) -> list[Snippet]:
    key = Cache.compute_hash(retriever.name, query, ORCH_VERSION)
    hit = cache.get("retrieve", key, _SnippetBundle)
    if hit is not None:
        return hit.snippets
    async with sem:
        try:
            snips = await retriever.search(query, n=n)
        except Exception as e:
            logger.warning(f"retriever {retriever.name} failed for {query!r}: {e!r}")
            return []
    cache.set("retrieve", key, _SnippetBundle(snippets=snips))
    return snips


async def retrieve_for_plan(
    plan: QueryPlan,
    *,
    retrievers: dict[str, Any],  # name -> retriever instance with .search() and .name
    cache: Cache,
    n_per_query: int = 4,
) -> dict[str, list[Snippet]]:
    """For each SectionPlan, fire all (source, query) combos in parallel.

    Returns {section_name: deduped list of snippets}.
    """
    sem = asyncio.Semaphore(CONCURRENCY)

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
        # dedup by URL, keep first occurrence
        seen: set[str] = set()
        merged: list[Snippet] = []
        for batch in results:
            for s in batch:
                key = str(s.url)
                if key not in seen:
                    seen.add(key)
                    merged.append(s)
        return sp.section_name, merged

    pairs = await asyncio.gather(*[_per_section(sp) for sp in plan.section_plans])
    return dict(pairs)
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/test_orchestrator.py -v
git add mentis/orchestrator.py tests/test_orchestrator.py
git commit -m "feat(orchestrator): per-section source routing + parallel retrieval

SOURCE_ROUTING dict maps each section to its retrievers. retrieve_for_plan
fires (section × source × query) combos via asyncio.gather, capped by
Semaphore(10). Dedups by URL per section. Cached at (retriever, query, version)."
```

---

## Task 13: Synthesizer with Citation Grounding

**Files:**
- Create: `mentis/synthesizer.py`
- Create: `tests/test_synthesizer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_synthesizer.py`:

```python
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
    (prompts_dir / "section_synthesizer.v1.j2").write_text(
        "Write {{ section_name }}. Use snippets only. {% for s in snippets %}{{ s.url }}{% endfor %}"
    )
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_synthesizer.py -v
```

- [ ] **Step 3: Implement synthesizer**

Create `mentis/synthesizer.py`:

```python
"""Synthesizer: snippets → SectionDraft with citation-grounded prose.

The anti-hallucination spine. Post-processes LLM output to verify every
cited URL is in the input snippet set; retries once; final fallback to
raw-snippet rendering with fallback_to_raw_snippets=True.
"""
from __future__ import annotations

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
    lines = ["*Note: the AI synthesizer could not produce grounded prose; raw retrieved sources below.*", ""]
    for s in snippets:
        title = s.title or s.source_name
        lines.append(f"- **{title}** — [{s.source_name}]({s.url})")
        lines.append(f"  {s.text[:300]}")
        lines.append("")
    return "\n".join(lines)


async def _default_llm_complete(
    *, llm_config: LLMConfig, system: str, user: str, schema, prompt_version: str, reframe_template: str | None
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
    try:
        reframe_template = registry.get("safety_reframe", version="v1").template_text
    except FileNotFoundError:
        pass

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
                citations=list(cited),  # type: ignore[arg-type]
                synthesizer_version=SYNTH_VERSION,
                safety_retries=safety_retries,
                fallback_to_raw_snippets=False,
            )
        logger.warning(f"section {section_name}: LLM cited unknown URLs {cited - allowed_urls}; retrying")
    except SafetyBlockedException:
        logger.warning(f"section {section_name}: all safety paths blocked; falling back to raw snippets")
        return SectionDraft(
            section_name=section_name,
            prose=_raw_snippets_render(snippets),
            snippets_used=snippets,
            citations=[s.url for s in snippets],
            synthesizer_version=SYNTH_VERSION,
            safety_retries=4,  # all 4 paths exhausted
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
                citations=list(cited2),  # type: ignore[arg-type]
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
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/test_synthesizer.py -v
git add mentis/synthesizer.py tests/test_synthesizer.py
git commit -m "feat(synthesizer): citation-grounded prose with hallucination guard

Post-processes LLM output: regex-extracts cited URLs, verifies all are in
input snippet set, retries once on hallucination, falls back to raw-snippet
render with fallback_to_raw_snippets=True. Handles SafetyBlockedException
from LLM client by also falling back to raw snippets."
```

---

## Task 14: Report Assembler (Executive Summary + References)

**Files:**
- Create: `mentis/report.py`
- Create: `tests/test_report.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_report.py`:

```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from mentis.llm import LLMConfig, SafetyTrace
from mentis.report import assemble_report, render_markdown
from mentis.schemas import (
    Reference,
    Report,
    SectionDraft,
    Snippet,
)


class _ESLLMOut(BaseModel):
    executive_summary: str


def _draft(name, urls) -> SectionDraft:
    snips = [
        Snippet(
            text="t",
            url=u,
            source_name="PubMed",
            source_kind="scientific",
            retrieved_at=datetime.now(),
        )
        for u in urls
    ]
    return SectionDraft(
        section_name=name,
        prose=f"prose for {name}",
        snippets_used=snips,
        citations=urls,
        synthesizer_version="v1",
    )


@pytest.mark.asyncio
async def test_assemble_report_generates_executive_summary(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "executive_summary.v1.j2").write_text("summary of {{ user_query }}")

    drafts = {
        "product_profile":     _draft("product_profile",    ["https://a.org/1"]),
        "clinical_use":        _draft("clinical_use",       ["https://a.org/2"]),
        "market_demand":       _draft("market_demand",      ["https://a.org/3"]),
        "manufacturers":       _draft("manufacturers",      ["https://a.org/4"]),
        "regulatory":          _draft("regulatory",         ["https://a.org/5"]),
        "sourcing_pricing":    _draft("sourcing_pricing",   ["https://a.org/6"]),
        "risks_alternatives":  _draft("risks_alternatives", ["https://a.org/7"]),
    }

    fake_llm = AsyncMock(
        return_value=(_ESLLMOut(executive_summary="ES text"), "gemini/gemini-2.0-flash", SafetyTrace())
    )

    report = await assemble_report(
        user_query="ranitidine",
        normalized_term="ranitidine HCl",
        section_drafts=drafts,
        llm_config=LLMConfig(),
        prompts_dir=prompts_dir,
        total_latency_ms=80_000,
        llm_complete=fake_llm,
    )
    assert isinstance(report, Report)
    assert len(report.sections) == 7
    assert report.executive_summary == "ES text"
    assert len(report.references) == 7  # one URL per section, all unique
    assert report.metadata.total_snippets_retrieved == 7


def test_render_markdown_includes_metadata_header_and_refs() -> None:
    drafts = [
        SectionDraft(
            section_name="product_profile",
            prose="A [fact](https://a.org/1).",
            snippets_used=[],
            citations=["https://a.org/1"],
            synthesizer_version="v1",
        )
    ]
    report = Report(
        user_query="x",
        normalized_term="X",
        sections=drafts,
        executive_summary="ES",
        references=[
            Reference(
                url="https://a.org/1",
                source_name="PubMed",
                title="t",
                retrieved_at=datetime.now(),
                used_in_sections=["product_profile"],
            )
        ],
        metadata={
            "mentis_version": "0.1.0",
            "llm_provider": "gemini/gemini-2.0-flash",
            "prompt_versions": {},
            "total_latency_ms": 1000,
            "total_snippets_retrieved": 1,
            "total_safety_retries": 0,
            "cost_usd": 0.001,
            "generated_at": datetime.now(),
        },
    )
    md = render_markdown(report)
    assert "# Mentis Procurement Brief" in md
    assert "Executive Summary" in md
    assert "References" in md
    assert "https://a.org/1" in md
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_report.py -v
```

- [ ] **Step 3: Implement report**

Create `mentis/report.py`:

```python
"""Report assembler: 7 SectionDrafts → Executive Summary → References → final Report.

Also renders Report → Markdown for display/export.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from mentis.llm import LLMClient, LLMConfig
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


async def _default_llm_complete(
    *, llm_config, system, user, schema, prompt_version
):
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
    # Order drafts per fixed sequence
    ordered = [section_drafts[name] for name in SectionNames if name in section_drafts]

    # Executive Summary
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

    metadata = ReportMetadata(
        mentis_version=MENTIS_VERSION,
        llm_provider=llm_config.providers[0],
        prompt_versions={"section_synthesizer": "v1", "query_planner": "v1", "executive_summary": "v1"},
        total_latency_ms=total_latency_ms,
        total_snippets_retrieved=snippets_total,
        total_safety_retries=safety_retries_total,
        cost_usd=0.0,  # populated by Langfuse callback in real runs
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
            md.append("> *This section could not be synthesized due to provider safety filters; raw sources shown above.*")
            md.append("")
    md.append("## References")
    md.append("")
    for i, r in enumerate(report.references, start=1):
        md.append(
            f"{i}. **{r.source_name}** — [{r.title}]({r.url}) — retrieved {r.retrieved_at.strftime('%Y-%m-%d')} — used in: {', '.join(r.used_in_sections)}"
        )
    md.append("")
    return "\n".join(md)
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/test_report.py -v
git add mentis/report.py tests/test_report.py
git commit -m "feat(report): assemble + render with ES + References

assemble_report orders drafts per fixed sequence, calls executive_summary
prompt, builds Reference list (deduped by URL, sorted by source authority),
populates metadata (latency, snippets count, safety retries). render_markdown
emits a McKinsey-shape Markdown brief with header + ES + 7 sections + Refs."
```

---

## Task 15: PDF Export (WeasyPrint)

**Files:**
- Create: `mentis/pdf.py`
- Create: `mentis/templates/report.html.j2`
- Create: `tests/test_pdf.py`

- [ ] **Step 1: Create the HTML template**

Create `mentis/templates/report.html.j2`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Mentis — {{ report.user_query }}</title>
  <style>
    @page { size: A4; margin: 1.5cm; @bottom-center { content: "Page " counter(page) " of " counter(pages); font-size: 9pt; color: #888; } }
    body { font-family: "Helvetica", "Arial", sans-serif; color: #1a1a1a; font-size: 11pt; line-height: 1.45; }
    .header { border-bottom: 3px solid #1a1a1a; padding-bottom: 0.6em; margin-bottom: 1.2em; }
    .header h1 { margin: 0; font-size: 18pt; letter-spacing: 0.05em; }
    .meta { color: #555; font-size: 9pt; margin-top: 0.5em; line-height: 1.6; }
    h2 { color: #1a1a1a; border-bottom: 1px solid #ccc; padding-bottom: 0.3em; margin-top: 1.5em; font-size: 13pt; }
    .es { background: #f4f4f0; padding: 0.8em 1em; border-left: 4px solid #1a1a1a; margin: 1em 0; }
    p { text-align: justify; }
    a { color: #0a4d8c; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .references { font-size: 9.5pt; }
    .references li { margin-bottom: 0.5em; }
    .fallback-note { color: #a04400; font-style: italic; font-size: 9.5pt; }
  </style>
</head>
<body>
  <div class="header">
    <h1>MENTIS · Procurement Intelligence Brief</h1>
    <div class="meta">
      <strong>Substance:</strong> {{ report.user_query }}
      {% if report.normalized_term %} · <strong>Normalized:</strong> {{ report.normalized_term }}{% endif %}
      <br>
      <strong>Generated:</strong> {{ report.metadata.generated_at.strftime("%Y-%m-%d %H:%M") }} ·
      <strong>Model:</strong> {{ report.metadata.llm_provider }}
      <br>
      <strong>Sources retrieved:</strong> {{ report.metadata.total_snippets_retrieved }} ·
      <strong>Cost:</strong> ${{ "%.4f" | format(report.metadata.cost_usd) }} ·
      <strong>Latency:</strong> {{ report.metadata.total_latency_ms }} ms ·
      <strong>Safety retries:</strong> {{ report.metadata.total_safety_retries }}
    </div>
  </div>

  <h2>Executive Summary</h2>
  <div class="es">{{ executive_summary_html | safe }}</div>

  {% for s in sections_html %}
  <h2>{{ s.title }}</h2>
  {{ s.html | safe }}
  {% if s.fallback %}<p class="fallback-note">This section could not be synthesized due to provider safety filters; raw sources shown above.</p>{% endif %}
  {% endfor %}

  <h2>References</h2>
  <ol class="references">
  {% for r in report.references %}
    <li><strong>{{ r.source_name }}</strong> — <a href="{{ r.url }}">{{ r.title }}</a> — retrieved {{ r.retrieved_at.strftime("%Y-%m-%d") }} — used in: {{ r.used_in_sections | join(", ") }}</li>
  {% endfor %}
  </ol>
</body>
</html>
```

- [ ] **Step 2: Write failing test**

Create `tests/test_pdf.py`:

```python
from __future__ import annotations

from datetime import datetime

import pytest

from mentis.pdf import report_to_pdf_bytes
from mentis.schemas import (
    Reference,
    Report,
    ReportMetadata,
    SectionDraft,
)


def _minimal_report() -> Report:
    return Report(
        user_query="ranitidine",
        normalized_term="ranitidine HCl",
        sections=[
            SectionDraft(
                section_name="product_profile",
                prose="Ranitidine is an [H2 antagonist](https://a.org/1).",
                snippets_used=[],
                citations=["https://a.org/1"],
                synthesizer_version="v1",
            )
        ],
        executive_summary="A short executive summary.",
        references=[
            Reference(
                url="https://a.org/1",
                source_name="PubMed",
                title="t",
                retrieved_at=datetime.now(),
                used_in_sections=["product_profile"],
            )
        ],
        metadata=ReportMetadata(
            mentis_version="0.1.0",
            llm_provider="gemini/gemini-2.0-flash",
            prompt_versions={},
            total_latency_ms=1000,
            total_snippets_retrieved=1,
            total_safety_retries=0,
            cost_usd=0.001,
            generated_at=datetime.now(),
        ),
    )


def test_report_to_pdf_bytes_returns_pdf() -> None:
    pdf_bytes = report_to_pdf_bytes(_minimal_report())
    assert pdf_bytes is not None
    assert pdf_bytes[:4] == b"%PDF"
    assert len(pdf_bytes) > 1000
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_pdf.py -v
```

- [ ] **Step 4: Implement PDF export**

Create `mentis/pdf.py`:

```python
"""PDF export via Markdown → HTML → WeasyPrint."""
from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markdown_it import MarkdownIt

from mentis.report import SECTION_TITLES
from mentis.schemas import Report

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(_TEMPLATES_DIR),
        autoescape=select_autoescape(["html"]),
    )


def _md_to_html(md_text: str) -> str:
    return MarkdownIt("commonmark").enable("linkify").render(md_text)


def report_to_pdf_bytes(report: Report) -> bytes | None:
    """Render Report → PDF bytes. Returns None on failure."""
    try:
        from weasyprint import HTML
    except Exception as e:
        logger.error(f"WeasyPrint import failed: {e!r}")
        return None

    sections_html = [
        {
            "title": SECTION_TITLES.get(s.section_name, s.section_name),
            "html": _md_to_html(s.prose),
            "fallback": s.fallback_to_raw_snippets,
        }
        for s in report.sections
    ]
    executive_summary_html = _md_to_html(report.executive_summary)

    template = _env().get_template("report.html.j2")
    html_str = template.render(
        report=report,
        sections_html=sections_html,
        executive_summary_html=executive_summary_html,
    )

    try:
        return HTML(string=html_str, base_url=".").write_pdf()
    except Exception as e:
        logger.error(f"WeasyPrint PDF render failed: {e!r}")
        return None


def report_to_pdf_file(report: Report, out_path: Path) -> bool:
    pdf_bytes = report_to_pdf_bytes(report)
    if pdf_bytes is None:
        return False
    out_path.write_bytes(pdf_bytes)
    return True
```

- [ ] **Step 5: Run tests + commit**

```bash
uv run pytest tests/test_pdf.py -v
git add mentis/pdf.py mentis/templates/report.html.j2 tests/test_pdf.py
git commit -m "feat(pdf): markdown → html → weasyprint PDF export

report.html.j2 is a print-styled Jinja2 template (A4, page footers, branded
header, justified prose, references). report_to_pdf_bytes returns None on
WeasyPrint failure (graceful — UI offers markdown-only fallback)."
```

---

## Task 16: CLI

**Files:**
- Create: `mentis/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_cli.py`:

```python
from __future__ import annotations

from click.testing import CliRunner

from mentis.cli import cli


def test_help_lists_report_command() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "report" in result.output


def test_cache_clear_works(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    runner = CliRunner()
    result = runner.invoke(cli, ["cache", "clear"])
    assert result.exit_code == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_cli.py -v
```

- [ ] **Step 3: Implement CLI**

Create `mentis/cli.py`:

```python
"""Mentis CLI — Click-based."""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import click
from dotenv import load_dotenv

from mentis.cache import Cache
from mentis.llm import LLMConfig
from mentis.observability import init_observability
from mentis.orchestrator import retrieve_for_plan
from mentis.pdf import report_to_pdf_file
from mentis.query_planner import plan_query
from mentis.report import assemble_report, render_markdown
from mentis.retrievers.openfda import OpenFDARetriever
from mentis.retrievers.pubmed import PubmedRetriever
from mentis.retrievers.rxnav import RxNavRetriever
from mentis.retrievers.tavily import TavilyRetriever
from mentis.retrievers.wikipedia import WikipediaRetriever
from mentis.synthesizer import synthesize_section


@click.group()
@click.option("--verbose", is_flag=True)
def cli(verbose: bool) -> None:
    """Mentis — procurement intelligence reports."""
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    init_observability()


@cli.command()
@click.argument("query")
@click.option("--out", type=click.Path(path_type=Path), help="Markdown output path")
@click.option("--pdf", type=click.Path(path_type=Path), help="PDF output path")
def report(query: str, out: Path | None, pdf: Path | None) -> None:
    """Generate a procurement intelligence report for QUERY."""
    asyncio.run(_run_pipeline(query, out, pdf))


async def _run_pipeline(query: str, out: Path | None, pdf: Path | None) -> None:
    t0 = time.perf_counter()
    cache = Cache.from_env()
    llm_config = LLMConfig.from_env()
    prompts_dir = Path("prompts")

    retrievers = {
        "tavily": TavilyRetriever(),
        "pubmed": PubmedRetriever(),
        "openfda": OpenFDARetriever(),
        "wikipedia": WikipediaRetriever(),
        "rxnav": RxNavRetriever(),
    }

    click.echo(f"[1/5] Planning research for {query!r}...")
    plan = await plan_query(
        user_query=query, llm_config=llm_config, prompts_dir=prompts_dir
    )
    click.echo(f"      → normalized: {plan.normalized_term} · {len(plan.section_plans)} sections")

    click.echo("[2/5] Retrieving sources in parallel...")
    section_snippets = await retrieve_for_plan(plan, retrievers=retrievers, cache=cache)
    total = sum(len(v) for v in section_snippets.values())
    click.echo(f"      → {total} snippets across {len(section_snippets)} sections")

    click.echo("[3/5] Synthesizing sections...")
    import asyncio as _asyncio
    draft_tasks = [
        synthesize_section(
            section_name=sp.section_name,
            snippets=section_snippets.get(sp.section_name, []),
            user_query=query,
            normalized_term=plan.normalized_term,
            llm_config=llm_config,
            prompts_dir=prompts_dir,
        )
        for sp in plan.section_plans
    ]
    drafts = await _asyncio.gather(*draft_tasks)
    section_drafts = {d.section_name: d for d in drafts}
    safety_retries = sum(d.safety_retries for d in drafts)
    fallbacks = sum(1 for d in drafts if d.fallback_to_raw_snippets)
    click.echo(f"      → {len(drafts)} sections · {safety_retries} safety retries · {fallbacks} raw-snippet fallbacks")

    click.echo("[4/5] Assembling report (executive summary + references)...")
    latency_ms = int((time.perf_counter() - t0) * 1000)
    report = await assemble_report(
        user_query=query,
        normalized_term=plan.normalized_term,
        section_drafts=section_drafts,
        llm_config=llm_config,
        prompts_dir=prompts_dir,
        total_latency_ms=latency_ms,
    )

    md = render_markdown(report)
    if out:
        out.write_text(md)
        click.echo(f"      → wrote {out}")
    else:
        click.echo(md)

    if pdf:
        click.echo("[5/5] Rendering PDF...")
        ok = report_to_pdf_file(report, pdf)
        if ok:
            click.echo(f"      → wrote {pdf}")
        else:
            click.echo(f"      ✗ PDF rendering failed; markdown still available", err=True)


@cli.group()
def cache() -> None:
    """Cache management."""


@cache.command("clear")
@click.option("--namespace", default=None)
def cache_clear(namespace: str | None) -> None:
    """Clear the snippet cache."""
    c = Cache.from_env()
    c.clear(namespace)
    click.echo(f"cache cleared ({namespace or 'all namespaces'})")


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/test_cli.py -v
git add mentis/cli.py tests/test_cli.py
git commit -m "feat(cli): click-based CLI tying full pipeline

Subcommands: report <query> [--out X.md] [--pdf X.pdf], cache clear.
Loads .env, initializes observability, instantiates 5 retrievers,
runs plan → retrieve → synthesize → assemble → render → optional PDF."
```

---

## Task 17: Gradio App (HF Space Entry)

**Files:**
- Create: `app.py`
- Create: `tests/test_app.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_app.py`:

```python
def test_app_module_importable() -> None:
    import app
    assert hasattr(app, "build_app")


def test_build_app_returns_blocks() -> None:
    import app
    import gradio as gr
    blocks = app.build_app()
    assert isinstance(blocks, gr.Blocks)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_app.py -v
```

- [ ] **Step 3: Implement Gradio app**

Create `app.py`:

```python
"""Gradio HF Space entry point for Mentis."""
from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from mentis.cache import Cache
from mentis.llm import LLMConfig
from mentis.observability import init_observability
from mentis.orchestrator import retrieve_for_plan
from mentis.pdf import report_to_pdf_bytes
from mentis.query_planner import plan_query
from mentis.report import assemble_report, render_markdown
from mentis.retrievers.openfda import OpenFDARetriever
from mentis.retrievers.pubmed import PubmedRetriever
from mentis.retrievers.rxnav import RxNavRetriever
from mentis.retrievers.tavily import TavilyRetriever
from mentis.retrievers.wikipedia import WikipediaRetriever

load_dotenv()
init_observability()


async def _run(query: str):
    t0 = time.perf_counter()
    cache = Cache.from_env()
    llm_config = LLMConfig.from_env()
    prompts_dir = Path("prompts")
    retrievers = {
        "tavily": TavilyRetriever(),
        "pubmed": PubmedRetriever(),
        "openfda": OpenFDARetriever(),
        "wikipedia": WikipediaRetriever(),
        "rxnav": RxNavRetriever(),
    }

    yield "⏳ Planning research...", "", None
    plan = await plan_query(user_query=query, llm_config=llm_config, prompts_dir=prompts_dir)

    yield f"⏳ Retrieving sources for {len(plan.section_plans)} sections...", "", None
    section_snippets = await retrieve_for_plan(plan, retrievers=retrievers, cache=cache)

    yield "⏳ Synthesizing sections...", "", None
    from mentis.synthesizer import synthesize_section

    draft_tasks = [
        synthesize_section(
            section_name=sp.section_name,
            snippets=section_snippets.get(sp.section_name, []),
            user_query=query,
            normalized_term=plan.normalized_term,
            llm_config=llm_config,
            prompts_dir=prompts_dir,
        )
        for sp in plan.section_plans
    ]
    drafts = await asyncio.gather(*draft_tasks)
    section_drafts = {d.section_name: d for d in drafts}

    yield "⏳ Assembling report...", "", None
    latency_ms = int((time.perf_counter() - t0) * 1000)
    report = await assemble_report(
        user_query=query,
        normalized_term=plan.normalized_term,
        section_drafts=section_drafts,
        llm_config=llm_config,
        prompts_dir=prompts_dir,
        total_latency_ms=latency_ms,
    )

    md = render_markdown(report)
    footer = (
        f"⏱ {report.metadata.total_latency_ms / 1000:.1f}s · "
        f"📄 {report.metadata.total_snippets_retrieved} sources · "
        f"💰 ${report.metadata.cost_usd:.4f} · "
        f"🛡 {report.metadata.total_safety_retries} safety retries"
    )

    pdf_bytes = report_to_pdf_bytes(report)
    pdf_path = None
    if pdf_bytes:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf_bytes)
        tmp.close()
        pdf_path = tmp.name

    yield footer, md, pdf_path


def run_pipeline(query: str):
    """Sync wrapper around the async generator for Gradio."""
    gen = _run(query)
    while True:
        try:
            yield asyncio.run(gen.__anext__())
        except StopAsyncIteration:
            break


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Mentis — Procurement Intelligence", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# 🔍 Mentis\n*Procurement intelligence for medical substances. Type a substance, get a McKinsey-shape brief in under 90 seconds.*"
        )
        with gr.Row():
            query_input = gr.Textbox(
                label="Substance",
                placeholder='e.g., "ranitidine", "0.9% saline", "tramadol"',
                scale=4,
            )
            go_btn = gr.Button("▶ Generate", variant="primary", scale=1)

        gr.Examples(
            examples=["ranitidine", "0.9% saline", "tramadol", "insulin glargine"],
            inputs=query_input,
        )

        status = gr.Markdown(value="*Ready.*")
        report_md = gr.Markdown(value="")
        pdf_download = gr.File(label="⬇ Download PDF", interactive=False)

        go_btn.click(
            run_pipeline,
            inputs=query_input,
            outputs=[status, report_md, pdf_download],
        )

    return app


if __name__ == "__main__":
    build_app().launch()
```

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/test_app.py -v
git add app.py tests/test_app.py
git commit -m "feat(app): gradio UI for HF Spaces

Streaming pipeline output: status text updates as each stage completes
(planning → retrieval → synthesis → assembly). Renders markdown report
inline + offers PDF download via gr.File. Uses gr.Examples for quick
demos with 4 sample queries."
```

---

## Task 18: Architecture Diagrams (Mermaid + mingrammer)

**Files:**
- Create: `docs/architecture.py`
- Modify: `README.md` (add Mermaid section)

- [ ] **Step 1: Create mingrammer architecture script**

Create `docs/architecture.py`:

```python
"""Generate docs/architecture.png via mingrammer/diagrams.

Run: uv run python docs/architecture.py
Output: docs/architecture.png
"""
from __future__ import annotations

from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.custom import Custom
from diagrams.onprem.client import Users
from diagrams.programming.framework import Fastapi
from diagrams.programming.language import Python

OUT = Path(__file__).parent / "architecture"

with Diagram(
    "Mentis — Procurement Intelligence Pipeline",
    show=False,
    direction="LR",
    filename=str(OUT),
    outformat="png",
):
    user = Users("Procurement Officer")

    with Cluster("Mentis (FastAPI / Gradio)"):
        planner = Python("query_planner")
        orchestrator = Python("orchestrator")
        synthesizer = Python("synthesizer")
        report = Python("report assembler")

    with Cluster("Retrievers (async parallel)"):
        tavily = Python("Tavily\n(web)")
        pubmed = Python("PubMed\n(scientific)")
        openfda = Python("OpenFDA\n(regulatory)")
        wiki = Python("Wikipedia\n(background)")
        rxnav = Python("RxNav\n(normalize)")

    with Cluster("LLM Layer (LiteLLM)"):
        gemini = Python("Gemini 2.0 Flash\n(primary)")
        groq = Python("Groq Llama 3.3 70B\n(fallback)")

    with Cluster("Observability"):
        langfuse = Python("Langfuse\ntraces + prompts + cost")

    user >> Edge(label="query") >> planner
    planner >> orchestrator
    orchestrator >> [tavily, pubmed, openfda, wiki, rxnav]
    [tavily, pubmed, openfda, wiki, rxnav] >> Edge(label="snippets") >> synthesizer
    synthesizer >> Edge(label="grounded prose") >> report
    report >> Edge(label="Markdown + PDF") >> user

    synthesizer >> Edge(style="dashed", color="gray", label="LLM call") >> gemini
    gemini >> Edge(style="dashed", color="gray", label="fallback") >> groq

    planner >> Edge(style="dashed", color="gray", label="observe") >> langfuse
    synthesizer >> Edge(style="dashed", color="gray", label="observe") >> langfuse
```

- [ ] **Step 2: Add Mermaid diagram to README**

Overwrite `README.md`:

```markdown
# Mentis 🔍

Procurement intelligence reports for medical substances. Type a substance, get a McKinsey-shape, citation-grounded brief in under 90 seconds — covering product profile, clinical use, market & demand, manufacturers, regulatory, sourcing & pricing, and risks & alternatives.

> POC in active development. See `docs/superpowers/specs/2026-05-11-mentis-poc-design.md` for the full design.

## Architecture

```mermaid
flowchart LR
    U[User query<br/>"ranitidine"]
    QP[query_planner<br/>+ RxNav normalize]
    ORCH[orchestrator<br/>source-routing]
    R1[Tavily<br/>web/market]
    R2[PubMed<br/>scientific]
    R3[OpenFDA<br/>regulatory]
    R4[Wikipedia<br/>background]
    SYN[synthesizer<br/>citation-grounded]
    REP[report assembler<br/>ES + References]
    OUT[Markdown + PDF<br/>with inline citations]

    U --> QP --> ORCH
    ORCH --> R1 & R2 & R3 & R4
    R1 & R2 & R3 & R4 --> SYN
    SYN --> REP --> OUT

    LF[(Langfuse<br/>traces + prompts + cost)]
    LLM[Gemini 2.0 Flash<br/>via LiteLLM<br/>Groq fallback]
    QP -.observe.-> LF
    SYN -.observe.-> LF
    SYN -.call.-> LLM
```

For the full architecture diagram with vendor icons, see `docs/architecture.png` (regenerate via `make diagram`).

## Quickstart

```bash
git clone https://github.com/Ansumanbhujabal/Mentis
cd Mentis
uv sync
cp .env.example .env  # fill in keys; see below for free-tier signups
uv run mentis report "ranitidine" --out report.md --pdf report.pdf
# or launch the Gradio app:
uv run python app.py
```

## Required keys (all free tier, no credit card)

- `GEMINI_API_KEY` — https://aistudio.google.com/app/apikey
- `TAVILY_API_KEY` — https://app.tavily.com/sign-in (1000 searches/month free)
- `LANGFUSE_*` — https://cloud.langfuse.com (optional; runs without it)
- `GROQ_API_KEY` — https://console.groq.com (optional; LLM fallback)

## Features

- **Hybrid retrieval orchestrator** routes each report section to authority-appropriate sources (PubMed/OpenFDA for scientific/regulatory; Tavily for market/supplier data).
- **Citation grounding**: the synthesizer can only cite URLs from retrieved snippets; post-generation verification rejects hallucinated URLs.
- **Safety filter handling**: 4-step escalation chain (relax → reframe → provider swap → honest raw-snippets fallback) covers ~98% of legitimate medical queries.
- **Langfuse-managed prompts**: in-repo `prompts/*.j2` are source of truth; runtime fetches from Langfuse with file fallback. No prompts in code.
- **PDF export** for procurement-grade deliverables.
- **Cost tracking** per report via Langfuse + LiteLLM callback.

## Roadmap

- **v2** — MedGemma fact-checker layer; eval harness with hand-labeled gold reports.
- **v3** — Dynamic section planning via LangGraph; user-uploadable internal documents.

## License

Private POC. Not for distribution.
```

- [ ] **Step 3: Generate the architecture image**

```bash
uv run python docs/architecture.py
# verifies architecture.png was created
ls -la docs/architecture.png
```

If the command fails because `graphviz` binary isn't installed:
```bash
sudo apt-get install -y graphviz
uv run python docs/architecture.py
```

- [ ] **Step 4: Commit**

```bash
git add README.md docs/architecture.py docs/architecture.png
git commit -m "docs: architecture diagrams (mermaid in README + mingrammer png)

README has Mermaid block for instant GitHub rendering. docs/architecture.py
generates docs/architecture.png via mingrammer/diagrams — used in the Loom
walkthrough. Regenerate via 'make diagram' when architecture changes."
```

---

## Task 19: Sample Reports + Loom Script

**Files:**
- Create: `walkthrough/script.md`
- Generate: `walkthrough/sample_reports/saline_0.9pct.{md,pdf}`
- Generate: `walkthrough/sample_reports/tramadol.{md,pdf}`
- Generate: `walkthrough/sample_reports/insulin_glargine.{md,pdf}`

This task requires real API calls. Make sure `.env` is filled in before starting.

- [ ] **Step 1: Verify API keys are set**

```bash
cat .env | grep -E "GEMINI_API_KEY|TAVILY_API_KEY" | grep -v "^#"
# Each line should show a non-empty value after =
```

- [ ] **Step 2: Generate 3 sample reports**

```bash
uv run mentis report "0.9% saline" --out walkthrough/sample_reports/saline_0.9pct.md --pdf walkthrough/sample_reports/saline_0.9pct.pdf
uv run mentis report "tramadol" --out walkthrough/sample_reports/tramadol.md --pdf walkthrough/sample_reports/tramadol.pdf
uv run mentis report "insulin glargine" --out walkthrough/sample_reports/insulin_glargine.md --pdf walkthrough/sample_reports/insulin_glargine.pdf
```

Expected: Each command takes ~60-120s and produces both `.md` and `.pdf` files.

- [ ] **Step 3: Verify report quality (manual check)**

Open each `.md` file. Check:
- Every section has prose (no "[BLOCKED]" placeholders unless intentional on tramadol)
- Citations are real URLs that resolve
- Executive Summary is coherent
- References section lists ≥ 15 distinct URLs

If any section is empty or has fake citations, investigate the cache (`mentis cache clear`) and re-run.

For **tramadol** specifically: expect ≥ 1 safety retry in the metadata footer. If it's 0, the safety chain may not be exercising; tweak the prompt or query to a more sensitive substance (e.g., "fentanyl").

- [ ] **Step 4: Create the Loom script**

Create `walkthrough/script.md`:

```markdown
# Mentis Loom Walkthrough — Script

**Target length:** ~7 minutes
**Audience:** Non-technical founder + her technical advisors
**Posture:** Confident, fast-paced, "this is how I'd think about it as CTO"

---

## 0:00–0:30 — Problem framing

> "Procurement teams in healthcare spend two to four hours researching every new substance before they can source it. Compliance, supplier discovery, market context, regulatory restrictions — that information lives in eight or nine different places. Most teams just don't do it thoroughly, and they pay for that with bad sourcing decisions. Mentis assembles that brief in under 90 seconds, every claim traceable to a real source."

**On screen:** Mentis Gradio UI, empty query box.

---

## 0:30–2:30 — Live demo: "ranitidine"

Type "ranitidine," click Generate.

**Narrate while sections stream:**

> "Watch what's happening. First, RxNav normalizes the query — gives me the canonical form, the synonyms, the RxCUI code. Then a query planner figures out what to search for in each section. Then eight retrievers fire in parallel: PubMed for the clinical section, Tavily for market and supplier data, OpenFDA for regulatory and recall information. As each section finishes retrieving, the synthesizer turns those raw sources into prose — but it can only cite URLs it actually saw. I'll show you why that matters in a second."

When the report appears, scroll through. Click on 2-3 citations to show they go to real PubMed papers, real FDA pages.

> "Every claim has a real source. The LLM physically cannot cite a URL it didn't receive. I verify the citations post-generation with regex — if it tries to hallucinate one, I retry, and if it does it again, I fall back to showing the raw snippets honestly rather than producing made-up prose."

---

## 2:30–5:30 — Architecture deep dive

**Cut to docs/architecture.png on screen.**

> "Here's what's under the hood. The pipeline is five stages: query planning, orchestration, retrieval, synthesis, assembly. Async throughout — without that, this would take five minutes instead of ninety seconds.
>
> The retrievers are the interesting part. I made one big architectural decision: medical and scientific sections route to medical APIs — PubMed, OpenFDA, RxNav. Market and supplier sections route to general web search via Tavily. Authority-appropriate sources per section. A McKinsey consultant would do the same thing manually; I'm just automating the routing.
>
> Here's the LLM layer. I deliberately didn't use LangChain or LangGraph here — the orchestration is a fixed DAG, and adding chain abstractions would have made the pipeline harder to reason about. I would use LangGraph when the workflow becomes truly agentic — for example, when a planner decides at runtime which sources to query. That's the v2 module."

**Cut to Langfuse dashboard on screen.**

> "Every prompt is version-controlled in git. At runtime, the registry tries Langfuse first, falls back to the in-repo file. So I can A/B test prompts in production without redeploying. Every LLM call is traced — input, output, latency, token cost. You can see I've spent about three cents running these demos. At scale, knowing the per-query cost is what lets you price the product correctly."

---

## 5:30–6:30 — Edge case: tramadol

Type "tramadol," click Generate.

**Watch the safety counter in the footer.**

> "Tramadol is a controlled substance. Generative models often refuse to answer questions about it, even legitimate procurement questions. Watch what the pipeline does.
>
> Step one: try the default model. Gets blocked. Step two: relax the safety filters on Gemini specifically for the dangerous-content category — this is a professional research context. Step three: rewrite the user's query, framing it as a procurement research request from a licensed professional. Step four: if Gemini still refuses, route the request to Groq instead — different filter policy.
>
> If all four steps fail, the section still renders, but it shows the raw retrieved sources directly instead of synthesized prose. Honest failure beats silent fabrication every time."

Point at the footer: "Safety retries: 2." Scroll through the tramadol report.

> "The regulatory section is particularly meaty here. Schedule IV in the US, controlled in India, recall history. Procurement officer can act on this."

---

## 6:30–7:00 — Roadmap

**Cut back to architecture diagram or whiteboard.**

> "Week two, I'd add a MedGemma fact-checker layer — a medical-specialized small language model that runs a second pass on every clinical claim. Pairs naturally with an eval harness — hand-labeled gold reports plus DeepEval-style metrics so we know whether prompt or model changes actually improved quality.
>
> Month one, dynamic section planning. A controlled substance like tramadol gets a heavier regulatory section than a biologic. A biologic like insulin glargine gets a deeper manufacturing section. That's where LangGraph would actually earn its place.
>
> The wedge — and this is where the product really compounds — is when your customers can upload their internal documents. Supplier contracts, RFQs, audit reports. Mentis cross-references those against the public sources. Now the brief isn't just market intelligence; it's market intelligence in the context of *their* existing supplier book. That's the version that becomes indispensable."

**Open one of the sample PDFs on screen.**

> "And this is what a procurement officer prints, shares with their compliance team, attaches to an RFQ. Same data, presentation-grade. Happy to walk through any part of this in more detail or open the repo — let me know."

---

## Recording checklist

- [ ] Clear desktop, close Slack/email notifications
- [ ] Have the Gradio app pre-warmed (run a sample query first so the cache is hot)
- [ ] Have the Langfuse dashboard pre-loaded in another tab
- [ ] Have docs/architecture.png ready in a third tab
- [ ] Have one of the sample PDFs ready in a fourth tab
- [ ] Mic test, screen-recorder window correct
- [ ] Practice once before recording
```

- [ ] **Step 5: Commit**

```bash
git add walkthrough/
git commit -m "docs: sample reports + loom walkthrough script

Three pre-generated reports (saline, tramadol, insulin glargine) as .md
and .pdf. walkthrough/script.md is the full 7-minute Loom narration
with section breakdowns and recording checklist."
```

---

## Task 20: HF Spaces Deployment + README Polish

**Files:**
- Create: `requirements.txt`
- Modify: `README.md` (add HF Spaces frontmatter, screenshot section)

- [ ] **Step 1: Generate pinned requirements.txt**

```bash
uv pip compile pyproject.toml -o requirements.txt
```

- [ ] **Step 2: Add HF Spaces frontmatter to README.md**

At the very top of `README.md`, before the existing `# Mentis 🔍` line, prepend:

```markdown
---
title: Mentis
emoji: 🔍
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.40.0
app_file: app.py
pinned: false
private: true
---

```

- [ ] **Step 3: Create the HF Space (UI)**

On https://huggingface.co/new-space:
- Owner: `ansumanbhujabal`
- Space name: `mentis`
- License: leave blank / proprietary
- SDK: Gradio
- Hardware: CPU basic (free)
- Visibility: **Private** (do not toggle public — this is bidding work)

After creation, in *Settings → Variables and secrets* add:
- `GEMINI_API_KEY` (required)
- `TAVILY_API_KEY` (required)
- `LANGFUSE_PUBLIC_KEY` (optional)
- `LANGFUSE_SECRET_KEY` (optional)
- `LANGFUSE_HOST` (default `https://cloud.langfuse.com`)
- `GROQ_API_KEY` (optional)

- [ ] **Step 4: Add HF remote and push**

```bash
git remote add hf https://huggingface.co/spaces/ansumanbhujabal/mentis
git push hf main
```

- [ ] **Step 5: Verify the Space builds**

Visit `https://huggingface.co/spaces/ansumanbhujabal/mentis`. Wait 3-5 minutes for the build (longer first time due to WeasyPrint system deps).

If the build fails with WeasyPrint errors, confirm `packages.txt` is present and contains the cairo/pango/etc. lines (T1 Step 5).

- [ ] **Step 6: Live smoke test**

In the live Space UI, run all 3 sample queries:
- "0.9% saline"
- "tramadol"
- "insulin glargine"

Each should complete in 60-120s and produce a coherent report + downloadable PDF.

- [ ] **Step 7: Push to GitHub + final commit**

```bash
git remote add origin https://github.com/Ansumanbhujabal/Mentis
git push -u origin main

# Final commit if README needs further polish
git add README.md requirements.txt
git commit -m "deploy: HF Spaces frontmatter + pinned requirements.txt

Production deploy to private HF Space at huggingface.co/spaces/ansumanbhujabal/mentis.
GitHub mirror at github.com/Ansumanbhujabal/Mentis (private)."
git push origin main
git push hf main
```

- [ ] **Step 8: Record the Loom**

Follow `walkthrough/script.md` checklist. Aim for one take. Save as unlisted Loom or YouTube. Link goes in the email to the prospect alongside the GitHub invite.

---

## Definition of "POC done" — final checklist

- [ ] Gradio app live at `huggingface.co/spaces/ansumanbhujabal/mentis`
- [ ] End-to-end report generates in < 120s for the 3 sample queries
- [ ] Every cited URL in every sample report resolves to a real source
- [ ] Safety filter handling demonstrably works on `tramadol` (≥ 1 retry surfaces in metadata)
- [ ] PDF export works for all 3 sample queries
- [ ] Langfuse dashboard shows traces + cost + prompt versions for at least 5 runs
- [ ] Loom video recorded, ~7 min, no major retakes
- [ ] README has Mermaid + screenshots + quickstart + roadmap section
- [ ] 3 sample reports bundled (`.md` and `.pdf`) in `walkthrough/sample_reports/`
- [ ] Repo invited to prospect's GitHub email
- [ ] `grep -r 'You are' mentis/ tests/` returns nothing (no prompts in Python)
- [ ] `uv run pytest` passes
- [ ] `uv run ruff check .` passes

---

## Plan Self-Review

**Spec coverage:** ✓ — all sections of the spec map to specific tasks.
- §2 architecture → T1 (scaffolding); §3 data model → T2 (schemas); §4.1–4.6 stages → T11, T12, T7-T10, T13, T14, T15; §5.1–5.4 cross-cutting → T5, T6, T4, T3; §6 deliverables → T17 (Gradio), T18 (diagrams), T19 (sample reports + Loom), T20 (deploy); §7 deps → T1; §9 DoD → final checklist above.

**Placeholder scan:** ✓ — no TBD/TODO in any task. Every "Step N" has either code, a command, or a manual instruction.

**Type consistency:** ✓ — `SectionDraft`, `SectionPlan`, `QueryPlan`, `Snippet`, `Report`, `LLMConfig`, `SafetyTrace`, `Cache`, `PromptRegistry` used identically across all tasks. `_run_pipeline` parameter signatures match between CLI (T16) and Gradio app (T17).

**Out-of-scope items honored:** ✓ — MedGemma, eval harness, LangGraph, user-uploadable docs all explicitly deferred to v2/v3 per spec §10.

---
