# Mentis — Design Spec

**Date:** 2026-05-11
**Author:** Ansumanbhujabal
**Status:** Approved (brainstorming complete; ready for implementation plan)
**Workspace:** `/opt/CodeRepo/mentis/`
**Repo:** `github.com/Ansumanbhujabal/Mentis` (private)
**Deadline:** 2026-05-12

---

## 1. North Star

**Mentis turns a single medical-substance query into a McKinsey-style, citation-grounded procurement intelligence report in under 90 seconds.**

Built as a POC for a pre-MVP B2B healthcare procurement startup. Demonstrates a hybrid retrieval pipeline (medical APIs for scientific/regulatory sections, web search for market/supplier sections) feeding an LLM synthesizer that physically cannot cite a URL it didn't receive in its prompt — the anti-hallucination spine.

The POC's job is not to ship a finished product. Its job is to make a non-technical founder and her technical advisors leave the demo thinking: *"This person can architect the AI brain of a procurement marketplace."*

### Design tenets

1. **Static pipeline shape, dynamic content per query.** Predictable structure across reports (8 fixed sections + auto-References) so outputs are comparable; everything inside is real-time retrieval + real-time synthesis.
2. **Snippet-grounded citations.** The synthesizer prompt only sees URLs from retrieved snippets. Post-generation verification rejects any hallucinated URL. Citations are inline markdown hyperlinks + a deduped References section.
3. **No prompts in code.** All prompts live in `prompts/*.j2`. Runtime fetches from Langfuse with file fallback.
4. **Honest failure over silent fabrication.** Safety blocks and retrieval failures surface visibly in the UI (badge counts, `[BLOCKED]` placeholders with raw snippets). Better to show "this didn't work" than to invent prose.
5. **24-hour ship discipline.** Every architectural choice is justified by what we can defensibly demo in the time budget. MedGemma, LangGraph, dynamic section planning, eval harness — all explicitly deferred to v2 with named roadmap slots.

---

## 2. High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                              MENTIS                                 │
│                                                                     │
│  user query ──►  query_planner  ──►  orchestrator                  │
│  ("ranitidine")     │                    │                          │
│                     ▼                    ▼                          │
│              per-section search    parallel retrieval               │
│              queries (8)           per section → snippets           │
│                                          │                          │
│                                          ▼                          │
│                                    synthesizer                      │
│                                    (per section, with               │
│                                     snippet grounding)              │
│                                          │                          │
│                                          ▼                          │
│                                    report assembler                 │
│                                    (sections + ES + Refs)           │
│                                          │                          │
│                                          ▼                          │
│                                     Markdown / HTML / PDF           │
│                                                                     │
│  Two faces, one core:                                              │
│    • CLI       mentis report "ranitidine" --pdf report.pdf          │
│    • Gradio    app.py on HF Spaces — live UI                        │
└────────────────────────────────────────────────────────────────────┘
```

### Stage latency budgets (target end-to-end ≤ 120s; aim ≤ 90s)

| Stage | Job | Budget |
|---|---|---|
| `query_planner` | RxNav normalize + 1 LLM call → per-section search queries | ~3s |
| `orchestrator` | Per-section source routing; fire all retrievals concurrently | parallel |
| `retrievers` | Tavily / PubMed / OpenFDA / Wikipedia / RxNav clients | ~5–15s parallel |
| `synthesizer` | 7 LLM calls in parallel, one per content section, snippet-grounded | ~15–30s |
| `report` | Assemble + Executive Summary LLM call + dedupe References | ~3s |
| `pdf` (on demand) | Markdown → HTML → WeasyPrint | ~2s |

### Module layout

```
mentis/                              # /opt/CodeRepo/mentis/
├── pyproject.toml                   # uv-managed
├── uv.lock                          # committed
├── Makefile                         # diagram, sync-prompts, deploy, samples
├── README.md                        # Mermaid arch + screenshots + quickstart
├── .env.example                     # GEMINI_API_KEY, TAVILY_API_KEY, LANGFUSE_*
├── .gitignore
├── packages.txt                     # HF Spaces system deps for WeasyPrint
├── mentis/
│   ├── __init__.py
│   ├── schemas.py                   # all Pydantic models
│   ├── llm.py                       # LiteLLM client + safety-aware fallback
│   ├── prompts.py                   # PromptRegistry: Langfuse → file fallback
│   ├── cache.py                     # snippet cache, 24h TTL
│   ├── observability.py             # Langfuse init + LiteLLM callback
│   ├── query_planner.py             # query → per-section search terms
│   ├── orchestrator.py              # source-routing + parallel retrieval
│   ├── synthesizer.py               # per-section prose w/ citation grounding
│   ├── report.py                    # assemble sections + ES + References
│   ├── pdf.py                       # Markdown → HTML → WeasyPrint PDF
│   ├── templates/
│   │   └── report.html.j2           # print-styled HTML template
│   ├── retrievers/
│   │   ├── __init__.py              # BaseRetriever protocol
│   │   ├── tavily.py
│   │   ├── pubmed.py
│   │   ├── openfda.py
│   │   ├── wikipedia.py
│   │   └── rxnav.py
│   └── cli.py                       # Click CLI: chapters, report, eval
├── prompts/
│   ├── query_planner.v1.j2
│   ├── section_synthesizer.v1.j2
│   ├── executive_summary.v1.j2
│   └── safety_reframe.v1.j2
├── app.py                           # Gradio HF Space entry
├── scripts/
│   └── sync_prompts.py              # push local prompts → Langfuse
├── docs/
│   ├── architecture.py              # mingrammer/diagrams source
│   ├── architecture.png             # generated
│   └── superpowers/
│       ├── specs/2026-05-11-mentis-poc-design.md
│       └── plans/2026-05-11-mentis-poc.md
├── walkthrough/
│   ├── script.md                    # Loom recording script
│   └── sample_reports/
│       ├── saline_0.9pct.{md,pdf}
│       ├── tramadol.{md,pdf}
│       └── insulin_glargine.{md,pdf}
└── tests/
    ├── test_schemas.py
    ├── test_retrievers_smoke.py
    └── test_orchestrator.py
```

---

## 3. Data Model

```python
# mentis/schemas.py — sketches, not final

class QueryPlan(BaseModel):
    user_query: str                       # raw input: "ranitidine"
    normalized_term: str | None           # "ranitidine hydrochloride" from RxNav
    rxnav_synonyms: list[str]             # additional drug names / brand names
    section_plans: list[SectionPlan]      # exactly 7 (Executive Summary generated separately, see §4.5)
    plan_version: str                     # for cache key invalidation

class SectionPlan(BaseModel):
    section_name: Literal[
        "product_profile",
        "clinical_use",
        "market_demand",
        "manufacturers",
        "regulatory",
        "sourcing_pricing",
        "risks_alternatives",
    ]
    search_queries: list[str]             # 2-4 sub-queries generated by planner
    sources: list[str]                    # subset of ["tavily","pubmed","openfda","wikipedia","rxnav"]

class Snippet(BaseModel):
    text: str                             # 200-500 word excerpt
    url: HttpUrl                          # exact, never paraphrased
    source_name: str                      # "PubMed", "OpenFDA", "Tavily"
    source_kind: Literal["scientific","regulatory","market","background"]
    title: str | None
    retrieved_at: datetime

class SectionDraft(BaseModel):
    section_name: str
    prose: str                            # markdown with inline [claim](url) hyperlinks
    snippets_used: list[Snippet]          # exactly which snippets the LLM saw
    citations: list[HttpUrl]              # URLs actually cited in prose (post-extract)
    synthesizer_version: str
    safety_retries: int                   # 0 if clean; >0 if safety filter tripped
    fallback_to_raw_snippets: bool        # True if synthesizer ultimately failed

class ReportMetadata(BaseModel):
    mentis_version: str
    llm_provider: str                     # "gemini/gemini-2.0-flash" (or fallback)
    prompt_versions: dict[str, str]       # {"section_synthesizer": "v1", ...}
    total_latency_ms: int
    total_snippets_retrieved: int
    total_safety_retries: int             # sum across all sections
    cost_usd: float                       # from Langfuse via LiteLLM
    generated_at: datetime

class Reference(BaseModel):
    url: HttpUrl
    source_name: str
    title: str
    retrieved_at: datetime
    used_in_sections: list[str]

class Report(BaseModel):
    user_query: str
    normalized_term: str | None
    sections: list[SectionDraft]          # exactly 7 content sections, ordered (ES held in separate field)
    executive_summary: str                # generated last
    references: list[Reference]           # deduped, sorted by source authority
    metadata: ReportMetadata
```

---

## 4. Pipeline Stages

### 4.1 `query_planner.py`

- Input: raw user query (e.g., `"ranitidine"`)
- Output: `QueryPlan`
- Steps:
  1. Call **RxNav** (`/REST/drugs.json?name=...`) to normalize the term + collect synonyms/brand names/RxCUI codes. This step alone massively improves PubMed/OpenFDA hit rates.
  2. One LLM call (Gemini Flash, via LiteLLM, prompt `query_planner.v1.j2`) that produces an 8-element JSON array of section plans, each with 2-4 search queries.
- Cache: `hash(user_query + plan_version)` → 24h TTL on the resulting `QueryPlan`.

### 4.2 `orchestrator.py`

- Input: `QueryPlan`
- Output: `dict[section_name, list[Snippet]]`
- Per-section source routing (encoded as a Python dict, not LLM-decided):
  ```python
  SOURCE_ROUTING = {
      "product_profile":     ["rxnav", "pubmed", "wikipedia"],
      "clinical_use":        ["pubmed", "openfda"],
      "market_demand":       ["tavily"],
      "manufacturers":       ["tavily"],
      "regulatory":          ["openfda", "tavily"],
      "sourcing_pricing":    ["tavily"],
      "risks_alternatives":  ["openfda", "tavily"],
  }
  ```
- For each `(section, source, query)` triple, fire the retriever. Limit per-source concurrency with `asyncio.Semaphore(10)`.
- Top 3-5 snippets per (section, source) → ~10-15 snippets per section after dedup.
- Snippet cache: `hash(source + query + retriever_version)` → 24h TTL.

### 4.3 Retrievers (`mentis/retrievers/*.py`)

Each retriever implements:
```python
class BaseRetriever(Protocol):
    async def search(self, query: str, n: int = 5) -> list[Snippet]: ...
```

Per-source:

- **`tavily.py`** — `tavily-python` SDK. Reads `TAVILY_API_KEY`. Returns top N web results with full snippets. Free tier: 1000 searches/month.
- **`pubmed.py`** — NCBI E-utilities (`esearch` + `efetch` over `pubmed` db). Two-step: search returns PMIDs; fetch returns abstracts. No auth needed, ~3 req/sec rate limit.
- **`openfda.py`** — `https://api.fda.gov/drug/{event,label,recall}.json`. No auth. Returns adverse event / label / recall data.
- **`wikipedia.py`** — Wikipedia REST `/api/rest_v1/page/summary/{title}`. Free, no auth. Falls back when other sources thin.
- **`rxnav.py`** — `https://rxnav.nlm.nih.gov/REST/`. Drug normalization, RxCUI codes, synonyms, brand names. Used by query_planner, not directly during retrieval orchestration.

All use `httpx.AsyncClient`. All retry with exponential backoff (3 attempts). All cap snippet text at 800 chars to keep LLM context bounded.

### 4.4 `synthesizer.py` — the anti-hallucination spine

- Input: `(section_name, list[Snippet])`
- Output: `SectionDraft`
- Behavior:
  1. Render `section_synthesizer.v1.j2` with `{section_name, snippets}`. Prompt instructs: write 200-300 words, use ONLY snippets below, embed `[claim](url)` markdown hyperlinks, every URL must come from snippets list.
  2. LLM call (Gemini Flash via LLMClient — includes safety escalation chain).
  3. **Post-process verification:**
     ```python
     cited_urls = set(re.findall(r'\]\((https?://[^)]+)\)', llm_output))
     allowed_urls = {str(s.url) for s in snippets}
     if cited_urls - allowed_urls:
         # LLM cited a URL not in our snippet set — reject and retry once
     ```
  4. If retry also fails: fall back to "raw snippets render" — section becomes a bulleted list of the snippets themselves with full attribution. `fallback_to_raw_snippets=True` and a visible footnote on the section.
- All 7 content sections fired via `asyncio.gather`. Executive Summary is generated separately in `report.py` after the 7 are complete (see §4.5).

### 4.5 `report.py`

- Input: `dict[section_name, SectionDraft]` (7 content sections)
- Output: `Report`
- Steps:
  1. Order section drafts per fixed sequence.
  2. Call `executive_summary.v1.j2` with the 7 drafts → 100-150 word Executive Summary. This is an 8th LLM call, executed sequentially after the 7 parallel section calls.
  3. Build `references` list: union of all `citations` across sections, deduped by URL, sorted by source authority (PubMed/OpenFDA first, Tavily/Wikipedia after), with `used_in_sections` annotation.
  4. Compute `metadata` (latency, cost from Langfuse, prompt versions, safety retries sum).
- The Markdown render template wraps: header with metadata → Executive Summary → 7 content sections → References. The reader sees 9 distinct blocks (1 ES + 7 sections + Refs); the synthesizer's parallel work targets the 7 content sections only.

### 4.6 `pdf.py`

- Input: `Report`
- Output: PDF bytes (returned in-memory or written to file)
- Pipeline: `Report → Markdown → markdown-it-py → HTML → Jinja2 (report.html.j2) → WeasyPrint → PDF`.
- HTML template uses print-friendly CSS (page breaks before sections, footer running, branded header).
- Fallback: if WeasyPrint raises, return None — Gradio UI offers only markdown download in that case.

---

## 5. Cross-Cutting Infrastructure

### 5.1 `llm.py` — Safety-aware LLM client

Single public method: `LLMClient.complete_with_safety(prompt, system, schema=None) -> tuple[T, SafetyTrace]`.

Escalation chain on safety block:

1. **Primary call** — Gemini 2.0 Flash with default safety settings.
2. **Retry 1** — Gemini with relaxed `safety_settings`: `HARM_CATEGORY_DANGEROUS_CONTENT` set to `BLOCK_ONLY_HIGH`. Logged.
3. **Retry 2** — Gemini with relaxed settings AND user prompt wrapped via `safety_reframe.v1.j2` ("research request for licensed healthcare procurement professionals..."). Logged.
4. **Provider fallback** — Groq Llama 3.3 70B (different filter policy). Logged.
5. **Final raise** — `SafetyBlockedException`. Synthesizer catches this, renders the section with raw snippets and marks `fallback_to_raw_snippets=True`.

`SafetyTrace` includes `retry_count` and the action history; flows into `SectionDraft.safety_retries` and aggregate `ReportMetadata.total_safety_retries`. UI shows this as a small badge.

### 5.2 `prompts.py` — Langfuse-first, file fallback. No prompts in code.

```python
class PromptRegistry:
    def get(self, name: str, version: str | None = None) -> RenderedPrompt:
        # 1. Try Langfuse with label="production"
        # 2. On any error → fall back to prompts/{name}.{version or 'v1'}.j2
```

Workflow:
1. Edit prompt in `prompts/{name}.{version}.j2`.
2. Run `make sync-prompts` → push to Langfuse via `scripts/sync_prompts.py`.
3. Optionally bump active version in Langfuse UI to A/B test in production.
4. Code calls `prompts.get(name)` — fetches from Langfuse with file fallback.

In-repo Jinja2 files are the source of truth; Langfuse is a runtime overlay for A/B testing + observability.

### 5.3 `observability.py` — Langfuse traces + cost

```python
def init_observability():
    if not is_langfuse_configured():
        return
    import litellm
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]
```

**Auto-captured via LiteLLM callback:** every LLM call's input, output, latency, token counts, **cost in USD**, model used, prompt version (passed in `metadata`).

**Manual spans** with `@langfuse.observe()` on retriever functions to capture URL, status, latency, snippet count.

**Cost surface for the prospect:** Langfuse dashboard shows $/day, $/report, top-cost sections. Report metadata footer embeds a one-line cost summary ("Generated using $0.0124 of LLM tokens").

### 5.4 `cache.py` — Snippet cache

- Storage: `~/.cache/mentis/snippets/{retriever}/{hash}.json`, or `/data/...` on HF Spaces (via `HF_HOME` env var).
- Key: `hash(retriever_name + query + retriever_version)`.
- TTL: 24h (configurable).
- **Caches only snippets (retrieval results).** Never caches LLM synthesis — same query rerun produces fresh prose. This is a feature for live demos ("see, every run is real-time").
- Manual cache-bust: `mentis cache clear`.

---

## 6. Deliverables

### Gradio UI (`app.py`)

Single page. No menu, no settings panel. Components:

- Title bar: "MENTIS · Procurement Intelligence" + model badge + prompt version
- Query input (textbox) + example chips ("ranitidine", "0.9% saline", "tramadol")
- "▶ Generate" button
- Live progress stream area: streams "✓ Section N/8: <name> ... K sources" as orchestrator completes each section
- Streaming markdown render of the assembled report
- Bottom action row:
  - "⬇ Download Markdown" button
  - "⬇ Download PDF" button (calls `pdf.report_to_pdf`)
- Footer: `<latency>s · <N> sources · $<cost> · <safety_retries> safety retries`

Uses Gradio's async generator pattern for streaming (`yield` from `run_pipeline`).

### Walkthrough Loom (~7 min)

Script (full in `walkthrough/script.md`):
- 0:00-0:30 — problem framing
- 0:30-2:30 — live demo on "ranitidine"
- 2:30-5:30 — architecture (mingrammer diagram + Langfuse dashboard)
- 5:30-6:30 — edge case: "tramadol" demonstrates safety filter handling
- 6:30-7:00 — roadmap (MedGemma, eval harness, internal data, LangGraph)
- Closes with opening a sample PDF on screen.

### Sample reports (bundled)

| Query | Why this query |
|---|---|
| `0.9% saline` | common chemical, all sections populated, baseline |
| `tramadol` | controlled substance — demonstrates safety filter handling |
| `insulin glargine` | biologic — demonstrates manufacturing/top-players depth |

Each saved as both `.md` and `.pdf` in `walkthrough/sample_reports/`.

### Architecture artifacts

- **`README.md`** — opens with Mermaid diagram (renders inline on GitHub); quickstart commands; screenshot of the Gradio UI; link to Loom; explicit roadmap section
- **`docs/architecture.py`** — mingrammer/diagrams source that generates `architecture.png` (the polished diagram shown in the Loom)
- **`Makefile`** — `make diagram`, `make sync-prompts`, `make deploy`, `make samples`

### HF Spaces deployment

- `requirements.txt` produced via `uv pip compile pyproject.toml`
- `packages.txt` lists system deps for WeasyPrint: `libcairo2 libpango-1.0-0 libpangoft2-1.0-0 libgdk-pixbuf2.0-0`
- README frontmatter has HF Spaces metadata (`sdk: gradio`, `app_file: app.py`)
- Secrets in HF Spaces UI: `GEMINI_API_KEY`, `TAVILY_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, `GROQ_API_KEY` (optional)

---

## 7. Dependencies (uv-managed)

```toml
[project]
name = "mentis"
version = "0.1.0"
requires-python = ">=3.11"
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

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-mock>=3.12",
    "ruff>=0.4",
    "diagrams>=0.23",   # mingrammer/diagrams for architecture.png
]
```

11 runtime deps, 5 dev deps. Lean.

---

## 8. Time Budget (Total: 22.5–23 working hours; window 36h)

| Block | Hours | Notes |
|---|---|---|
| Foundation: uv init, pyproject, schemas, cache, llm, prompts, observability | 2 | borrows heavily from Pugmark |
| Retrievers (5): Tavily, PubMed, OpenFDA, Wikipedia, RxNav | 3 | parallelize via subagents |
| Pipeline: query_planner, orchestrator, synthesizer, report | 4 | synthesizer is most polish-sensitive |
| UI + CLI: Gradio app, Click CLI | 2 | |
| PDF export: WeasyPrint + Jinja2 print template | 1.5 | new |
| First E2E + iterate synthesizer prompt | 3 | this is where quality is won |
| Sample reports + Mermaid + mingrammer | 2 | 3 md + 3 pdf, diagrams |
| HF Spaces deploy + packages.txt | 1 | |
| Loom recording + README polish | 2 | |
| Buffer | 1–2 | because things slip |
| **Total** | **22.5–23** | |

---

## 9. Definition of POC done

- [ ] Gradio app live at `huggingface.co/spaces/ansumanbhujabal/mentis`
- [ ] End-to-end report generates in < 120s for the 3 sample queries
- [ ] Every cited URL in every sample report resolves to a real source
- [ ] Safety filter handling demonstrably works on `tramadol` test query (≥ 1 retry surfaces in metadata)
- [ ] PDF export works for all 3 sample queries
- [ ] Langfuse dashboard shows traces + cost + prompt versions for at least 5 runs
- [ ] Loom video recorded, ~7 min, no major retakes
- [ ] README has Mermaid + screenshots + quickstart + roadmap section
- [ ] 3 sample reports bundled (`.md` and `.pdf`) in `walkthrough/sample_reports/`
- [ ] Repo shared with prospect (GitHub invite)
- [ ] No prompts strings in any `.py` file (`grep -r 'You are' mentis/` returns nothing in Python files)

---

## 10. Out of scope (deferred to v2 or v3 — explicit roadmap)

These are intentionally NOT in the 24h build. Each one becomes a talking point in the Loom roadmap.

- ❌ Authentication / user accounts
- ❌ Multi-language support
- ❌ **MedGemma fact-checker** — v2 module: run MedGemma-9B as a verifier pass on medical claims. Pairs with eval harness.
- ❌ **Eval harness** — hand-labeled "gold reports" + DeepEval-style metrics across queries. v2 priority.
- ❌ **Dynamic section planning** — LangGraph-based planner that decides which sections matter per query (controlled substance gets heavier Compliance, biologic gets heavier Manufacturing). v2.
- ❌ **User-uploadable internal documents** — customers upload their supplier contracts, RFQs; Mentis cross-references. **Huge future value** — pitch this as the "wedge" in the roadmap.
- ❌ Supplier database integration (when the prospect has one)
- ❌ Internationalization (US/India/EU regulatory differences)
- ❌ Caching beyond snippets (no synthesis cache — feature, not bug)
- ❌ Auth, billing, multi-tenant — not POC territory

---

## 11. Risks & Open Questions

### Risks

| Risk | Mitigation |
|---|---|
| Gemini free-tier rate limits during live demo | LiteLLM auto-falls back to Groq Llama 3.3 70B |
| WeasyPrint system-dep issues on HF Spaces | `packages.txt` lists cairo/pango/etc.; fallback offers markdown-only if PDF fails |
| Tavily 1000/mo limit | snippet cache means each query's retrieval is one-time; 1000 unique queries is plenty for POC + demo |
| PubMed/OpenFDA rate limits during high parallelism | Semaphore(10) concurrency cap + exponential backoff |
| Safety filter blocks legitimate medical content | 4-step escalation chain (relax → reframe → provider swap → honest surface) |
| LLM cites URLs not in snippets | Post-generation regex verification + retry; final fallback to raw-snippets section |
| Demo crashes during the prospect's live test | Pre-warm 3 sample queries' caches; have 3 pre-recorded markdown reports as offline fallback |

### Open Questions (resolve during planning or implementation)

1. Should the section order be customizable per query type (e.g., regulatory first for controlled substances)? — Deferred to v2 dynamic planning. v1 = fixed order.
2. Source authority weighting for the References sort — is PubMed always ahead of OpenFDA, or context-dependent? — Default: PubMed → OpenFDA → Wikipedia → Tavily. Tunable per section.
3. PDF page layout — A4 or Letter? Both? — Default A4 (international/India). Add Letter via CSS media query if T+22h has slack.
4. Loom hosting — Loom proper, or YouTube unlisted? — Loom has better analytics; YouTube embeds anywhere. Decide at recording time.

---

## 12. Decision Log

This spec is the result of 4 sequenced questions during brainstorming on 2026-05-11:

| # | Question | Choice | Rationale |
|---|---|---|---|
| 1 | Demo surface | A: Gradio web app on HF Spaces, staged delivery (Loom → live link) | Controlled first impression via video; interactivity follows |
| 2 | Data sources | C: Hybrid orchestrator routing per section | McKinsey-grade output requires authority-appropriate sources per section |
| 3 | LLM provider | Gemini 2.0 Flash via LiteLLM, Groq fallback; MedGemma as v2 roadmap | Free tier, no card, 1M context; MedGemma's value is for non-RAG medical reasoning, defer until v2 |
| 4 | Report shape + citations | 8 fixed sections + auto-References, inline hyperlinks + bottom References, snippet-grounded | Predictable shape for comparison; inline hyperlinks for credibility + References for formality |

Plus user-driven refinements:
- No LangChain — raw async Python (justified in Loom as architectural judgment)
- Project name: Mentis (Latin "of the mind"; avoids Mentimeter collision)
- Mermaid in README (always live) + mingrammer/diagrams for the polished Loom diagram
- **PDF export added** — recognized as a business artifact for procurement officers, not just a developer dump
- **Safety filter handling** — 4-step escalation chain
- **Langfuse for prompt mgmt + cost tracking, no prompts in code**

---
