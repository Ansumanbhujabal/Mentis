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

            try:
                r = await client.get(
                    EFETCH_URL,
                    params={
                        "db": "pubmed",
                        "id": ",".join(idlist),
                        "rettype": "abstract",
                        "retmode": "xml",
                    },
                )
                r.raise_for_status()
                xml_text = r.text
            except Exception as e:
                logger.warning(f"pubmed efetch failed for ids {idlist}: {e!r}")
                return []

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
