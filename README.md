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

For the polished architecture image with vendor icons, see `docs/architecture.png` (regenerate via `make diagram`).

## Quickstart

```bash
git clone https://github.com/Ansumanbhujabal/Mentis
cd Mentis
uv sync
cp .env.example .env  # fill in keys; see below
uv run mentis report "ranitidine" --out report.md --pdf report.pdf
# or launch the Gradio app:
uv run python app.py
```

## Required keys (all free tier, no credit card)

- `GEMINI_API_KEY` — https://aistudio.google.com/app/apikey
- `TAVILY_API_KEY` — https://app.tavily.com/sign-in (1000 searches/month free)
- `LANGFUSE_*` — https://cloud.langfuse.com (optional; pipeline runs without it)
- `GROQ_API_KEY` — https://console.groq.com (optional; LLM fallback)

## Features

- **Hybrid retrieval orchestrator** routes each report section to authority-appropriate sources (PubMed/OpenFDA for scientific/regulatory; Tavily for market/supplier data).
- **Citation grounding**: the synthesizer can only cite URLs from retrieved snippets; post-generation verification rejects hallucinated URLs.
- **Safety filter handling**: 4-step escalation chain (relax → reframe → provider swap → honest raw-snippets fallback) covers most legitimate medical queries.
- **Langfuse-managed prompts**: in-repo `prompts/*.j2` are source of truth; runtime fetches from Langfuse with file fallback. No prompts in code.
- **PDF export** for procurement-grade deliverables.
- **Cost tracking** per report via Langfuse + LiteLLM callback.

## Roadmap

- **v2** — MedGemma fact-checker layer; eval harness with hand-labeled gold reports.
- **v3** — Dynamic section planning via LangGraph; user-uploadable internal documents.

## License

Proprietary POC.
