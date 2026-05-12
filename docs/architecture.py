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
    graph_attr={"fontsize": "14", "splines": "spline", "ranksep": "1.0"},
):
    user = Users("Procurement Officer")

    with Cluster("Mentis (Gradio / CLI)"):
        planner = Python("query_planner\n(query → 7-section plan)")
        orchestrator = Python("orchestrator\n(source routing + cache)")
        synthesizer = Python("synthesizer × 7\n(citation-grounded)")
        polish = Python("polish (optional)\n(per-section tables)")
        report_node = Python("report assembler\n(ES + de-dup refs)")

    with Cluster("Retrievers (async parallel)"):
        tavily = Python("Tavily\n(web / market)")
        pubmed = Python("PubMed\n(scientific)")
        openfda = Python("OpenFDA\n(regulatory)")
        wiki = Python("Wikipedia\n(background)")
        rxnav = Python("RxNav (NLM)\n(normalize)")

    with Cluster("LLM Layer (LiteLLM)"):
        azure = Python("Azure OpenAI\ngpt-4o\n(primary)")
        gemini = Python("Gemini 2.0 Flash\n(safety-relaxed)")
        groq = Python("Groq Llama 3.3 70B\n(fallback)")

    with Cluster("Observability"):
        langfuse = Python("Langfuse\ntraces + prompts + cost")

    user >> Edge(label="query") >> planner
    planner >> orchestrator
    orchestrator >> [tavily, pubmed, openfda, wiki, rxnav]
    [tavily, pubmed, openfda, wiki, rxnav] >> Edge(label="snippets") >> synthesizer
    synthesizer >> Edge(label="grounded prose") >> polish
    polish >> Edge(label="polished") >> report_node
    synthesizer >> Edge(style="dotted", color="gray", label="raw path") >> report_node
    report_node >> Edge(label="Markdown + PDF\n(raw & polished)") >> user

    synthesizer >> Edge(style="dashed", color="gray", label="LLM call") >> azure
    polish >> Edge(style="dashed", color="gray", label="LLM call") >> azure
    planner >> Edge(style="dashed", color="gray", label="LLM call") >> azure
    azure >> Edge(style="dashed", color="gray", label="safety fallback") >> gemini
    gemini >> Edge(style="dashed", color="gray", label="fallback") >> groq

    planner >> Edge(style="dashed", color="gray", label="observe") >> langfuse
    synthesizer >> Edge(style="dashed", color="gray", label="observe") >> langfuse
    polish >> Edge(style="dashed", color="gray", label="observe") >> langfuse
