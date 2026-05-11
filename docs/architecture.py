"""Generate docs/architecture.png via mingrammer/diagrams.

Run: uv run python docs/architecture.py
Output: docs/architecture.png
"""
from __future__ import annotations

from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.onprem.client import Users
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

    with Cluster("Mentis (Gradio / CLI)"):
        planner = Python("query_planner")
        orchestrator = Python("orchestrator")
        synthesizer = Python("synthesizer")
        report_node = Python("report assembler")

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
    synthesizer >> Edge(label="grounded prose") >> report_node
    report_node >> Edge(label="Markdown + PDF") >> user

    synthesizer >> Edge(style="dashed", color="gray", label="LLM call") >> gemini
    gemini >> Edge(style="dashed", color="gray", label="fallback") >> groq

    planner >> Edge(style="dashed", color="gray", label="observe") >> langfuse
    synthesizer >> Edge(style="dashed", color="gray", label="observe") >> langfuse
