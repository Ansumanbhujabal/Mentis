"""Gradio HF Space entry point for Mentis."""
from __future__ import annotations

import asyncio
import contextlib
import tempfile
import time
from pathlib import Path

# Monkey-patch gradio_client schema bug BEFORE importing gradio.
# Gradio 4.40 + Pydantic v2 produces schema fragments that are bool (True/False)
# meaning "anything" / "nothing". gradio_client.utils.get_type does `"const" in schema`
# which raises TypeError when schema is a bool. We wrap the helpers to fall through
# gracefully on non-dict inputs.
import gradio_client.utils as _gc_utils  # noqa: E402

_orig_get_type = _gc_utils.get_type
_orig_json_schema = _gc_utils._json_schema_to_python_type


def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)


def _safe_json_schema(schema, defs):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_json_schema(schema, defs)


_gc_utils.get_type = _safe_get_type
_gc_utils._json_schema_to_python_type = _safe_json_schema

import gradio as gr  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

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
from mentis.synthesizer import synthesize_section

load_dotenv()
init_observability()

# Precomputed reports — instant load for demo queries that match.
# Falls through to live pipeline on cache miss.
_PRECOMPUTED_DIR = Path(__file__).parent / "walkthrough" / "sample_reports"
PRECOMPUTED = {
    "0.9% saline": "saline_0.9pct",
    "saline": "saline_0.9pct",
    "tramadol": "tramadol",
    "insulin glargine": "insulin_glargine",
    "insulin": "insulin_glargine",
}


def _precomputed_for(query: str) -> tuple[str, Path] | None:
    """Return (markdown_text, pdf_path) if query matches a precomputed sample."""
    key = query.strip().lower()
    slug = PRECOMPUTED.get(key)
    if slug is None:
        return None
    md_path = _PRECOMPUTED_DIR / f"{slug}.md"
    pdf_path = _PRECOMPUTED_DIR / f"{slug}.pdf"
    if not md_path.exists():
        return None
    md_text = md_path.read_text()
    return md_text, pdf_path if pdf_path.exists() else None


async def _run(query: str):
    t0 = time.perf_counter()

    # Precomputed-cache fast path: if the query matches a sample, return its
    # pre-rendered markdown + PDF instantly. Bypasses all retrieval + LLM calls.
    hit = _precomputed_for(query)
    if hit is not None:
        md_text, pdf_path = hit
        footer = (
            f"⚡ Precomputed sample · "
            f"📄 prebuilt PDF · "
            f"💰 $0.00 (cached) · "
            f"🛡 0 safety retries"
        )
        yield footer, md_text, str(pdf_path) if pdf_path else None
        return

    cache = Cache.from_env()
    llm_config = LLMConfig.from_env()
    prompts_dir = Path("prompts")

    retrievers: dict = {}
    with contextlib.suppress(RuntimeError):
        retrievers["tavily"] = TavilyRetriever()
    retrievers["pubmed"] = PubmedRetriever()
    retrievers["openfda"] = OpenFDARetriever()
    retrievers["wikipedia"] = WikipediaRetriever()
    retrievers["rxnav"] = RxNavRetriever()

    yield "⏳ Planning research...", "", None
    plan = await plan_query(user_query=query, llm_config=llm_config, prompts_dir=prompts_dir)

    yield f"⏳ Retrieving sources for {len(plan.section_plans)} sections...", "", None
    section_snippets = await retrieve_for_plan(plan, retrievers=retrievers, cache=cache)

    yield "⏳ Synthesizing sections...", "", None
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
    assembled = await assemble_report(
        user_query=query,
        normalized_term=plan.normalized_term,
        section_drafts=section_drafts,
        llm_config=llm_config,
        prompts_dir=prompts_dir,
        total_latency_ms=latency_ms,
    )

    md = render_markdown(assembled)
    footer = (
        f"⏱ {assembled.metadata.total_latency_ms / 1000:.1f}s · "
        f"📄 {assembled.metadata.total_snippets_retrieved} sources · "
        f"💰 ${assembled.metadata.cost_usd:.4f} · "
        f"🛡 {assembled.metadata.total_safety_retries} safety retries"
    )

    pdf_bytes = report_to_pdf_bytes(assembled)
    pdf_path = None
    if pdf_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            pdf_path = tmp.name

    yield footer, md, pdf_path


def run_pipeline(query: str):
    """Sync wrapper around the async generator for Gradio."""
    gen = _run(query)
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                yield loop.run_until_complete(gen.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.close()


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Mentis — Procurement Intelligence", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# 🔍 Mentis\n*Procurement intelligence for medical substances. "
            "Type a substance, get a structured, citation-grounded brief.*"
        )
        with gr.Row():
            query_input = gr.Textbox(
                label="Substance",
                placeholder='e.g., "0.9% saline", "tramadol", "insulin glargine"',
                scale=4,
            )
            go_btn = gr.Button("▶ Generate", variant="primary", scale=1)

        gr.Examples(
            examples=["0.9% saline", "tramadol", "insulin glargine"],
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


# Top-level demo so HF Spaces' runner can pick it up at import time.
demo = build_app()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
