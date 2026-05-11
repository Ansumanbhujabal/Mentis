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
@click.option("--polish/--no-polish", default=False, help="Run the post-processing polish layer (adds tables, tighter structure)")
def report(query: str, out: Path | None, pdf: Path | None, polish: bool) -> None:
    """Generate a procurement intelligence report for QUERY.

    Default output filenames: {Substance}_mentis_report.md and .pdf in the current directory.
    """
    from mentis.polish import slug_filename
    if out is None:
        out = Path(slug_filename(query, "md"))
    if pdf is None:
        pdf = Path(slug_filename(query, "pdf"))
    asyncio.run(_run_pipeline(query, out, pdf, polish))


async def _run_pipeline(query: str, out: Path | None, pdf: Path | None, do_polish: bool = False) -> None:
    t0 = time.perf_counter()
    cache = Cache.from_env()
    llm_config = LLMConfig.from_env()
    prompts_dir = Path("prompts")

    retrievers: dict = {}
    try:
        retrievers["tavily"] = TavilyRetriever()
    except RuntimeError as e:
        click.echo(f"⚠ Tavily disabled: {e}", err=True)
    retrievers["pubmed"] = PubmedRetriever()
    retrievers["openfda"] = OpenFDARetriever()
    retrievers["wikipedia"] = WikipediaRetriever()
    retrievers["rxnav"] = RxNavRetriever()

    click.echo(f"[1/5] Planning research for {query!r}...")
    plan = await plan_query(
        user_query=query, llm_config=llm_config, prompts_dir=prompts_dir
    )
    click.echo(
        f"      → normalized: {plan.normalized_term} · {len(plan.section_plans)} sections"
    )

    click.echo("[2/5] Retrieving sources in parallel...")
    section_snippets = await retrieve_for_plan(plan, retrievers=retrievers, cache=cache)
    total = sum(len(v) for v in section_snippets.values())
    click.echo(f"      → {total} snippets across {len(section_snippets)} sections")

    click.echo("[3/5] Synthesizing sections...")
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
    safety_retries = sum(d.safety_retries for d in drafts)
    fallbacks = sum(1 for d in drafts if d.fallback_to_raw_snippets)
    click.echo(
        f"      → {len(drafts)} sections · {safety_retries} safety retries"
        f" · {fallbacks} raw-snippet fallbacks"
    )

    click.echo("[4/5] Assembling report (executive summary + references)...")
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

    # Always write the no-polish (raw) outputs first
    if out:
        out.write_text(md)
        click.echo(f"      → wrote {out}  (raw / no polish)")
    else:
        click.echo(md)
    if pdf:
        click.echo("[5/5] Rendering PDF (raw)...")
        ok = report_to_pdf_file(assembled, pdf)
        if ok:
            click.echo(f"      → wrote {pdf}  (raw / no polish)")
        else:
            click.echo("      ✗ PDF rendering failed; markdown still available", err=True)

    # If --polish, ALSO produce a polished variant alongside (does not replace the raw output)
    if do_polish:
        click.echo("[polish] Per-section polish (table + headline per section)...")
        from mentis.polish import polish_report_per_section
        polished_md = await polish_report_per_section(
            user_query=query,
            original_markdown=md,
            llm_config=llm_config,
            prompts_dir=prompts_dir,
        )
        if out:
            polished_md_path = out.with_name(out.stem + "_polished" + out.suffix)
            polished_md_path.write_text(polished_md)
            click.echo(f"      → wrote {polished_md_path}  (polished)")
        if pdf:
            polished_pdf_path = pdf.with_name(pdf.stem + "_polished" + pdf.suffix)
            from mentis.pdf import polished_markdown_to_pdf_bytes
            pdf_bytes = polished_markdown_to_pdf_bytes(polished_md)
            if pdf_bytes:
                Path(polished_pdf_path).write_bytes(pdf_bytes)
                click.echo(f"      → wrote {polished_pdf_path}  (polished)")
            else:
                click.echo("      ✗ Polished PDF render failed", err=True)


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
