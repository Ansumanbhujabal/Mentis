"""Gradio HF Space entry point for Mentis."""
from __future__ import annotations

import asyncio
import contextlib
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
from mentis.synthesizer import synthesize_section

load_dotenv()
init_observability()


async def _run(query: str):
    t0 = time.perf_counter()
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


# Top-level demo so HF Spaces' runner can pick it up at import time.
demo = build_app()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
