"""Fetch and store PubMed abstracts via the NCBI E-utilities API.

Uses ``esearch`` to find PMIDs for a gene/keyword query, then ``efetch``
to retrieve full abstract metadata.  Results are stored as JSON for
later embedding by :mod:`perturbgpt.rag.literature_index`.

NCBI rate limits (as of 2024):
  - Without API key: **3 requests/second**
  - With API key (register at https://www.ncbi.nlm.nih.gov/account/):
    **10 requests/second**

Set the ``NCBI_API_KEY`` environment variable to use the higher rate.
This module sleeps between requests to respect the limit.
"""

from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence

import requests  # core dependency

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

RATE_LIMIT_INTERVAL_NO_KEY = 0.30   # ~3 req/s
RATE_LIMIT_INTERVAL_WITH_KEY = 0.10  # ~10 req/s


@dataclass
class PubMedRecord:
    """A single PubMed abstract with metadata."""

    pmid: str
    title: str
    abstract: str
    journal: str
    year: str


def _get_api_key() -> Optional[str]:
    return os.environ.get("NCBI_API_KEY")


def _rate_limit_sleep():
    """Sleep to respect NCBI rate limits."""
    interval = RATE_LIMIT_INTERVAL_WITH_KEY if _get_api_key() else RATE_LIMIT_INTERVAL_NO_KEY
    time.sleep(interval)


def search_pubmed(
    query: str,
    max_results: int = 10,
    api_key: Optional[str] = None,
) -> list[str]:
    """Search PubMed via esearch and return a list of PMID strings."""
    if api_key is None:
        api_key = _get_api_key()
    params = {"db": "pubmed", "term": query, "retmax": str(max_results), "retmode": "json"}
    if api_key:
        params["api_key"] = api_key
    resp = requests.get(ESEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    _rate_limit_sleep()
    return [str(x) for x in resp.json().get("esearchresult", {}).get("idlist", [])]


def _parse_efetch_xml(xml_text: str) -> list[PubMedRecord]:
    """Parse efetch XML response into PubMedRecord objects."""
    records: list[PubMedRecord] = []
    root = ET.fromstring(xml_text)
    for article in root.findall(".//PubmedArticle"):
        pmid_elem = article.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else ""
        title_elem = article.find(".//ArticleTitle")
        title = "".join(title_elem.itertext()) if title_elem is not None else ""
        abstract_parts = article.findall(".//Abstract/AbstractText")
        abstract = " ".join("".join(p.itertext()) for p in abstract_parts) if abstract_parts else ""
        journal_elem = article.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else ""
        year_elem = article.find(".//PubDate/Year")
        if year_elem is None:
            year_elem = article.find(".//PubDate/MedlineDate")
        year = year_elem.text if year_elem is not None else ""
        records.append(PubMedRecord(
            pmid=pmid, title=title.strip(), abstract=abstract.strip(),
            journal=journal.strip(), year=(year or "").strip(),
        ))
    return records


def fetch_abstracts(
    pmid_list: Sequence[str],
    api_key: Optional[str] = None,
) -> list[PubMedRecord]:
    """Fetch full abstract metadata for a list of PMIDs via efetch."""
    if not pmid_list:
        return []
    if api_key is None:
        api_key = _get_api_key()
    params = {"db": "pubmed", "id": ",".join(pmid_list), "retmode": "xml", "rettype": "abstract"}
    if api_key:
        params["api_key"] = api_key
    resp = requests.get(EFETCH_URL, params=params, timeout=60)
    resp.raise_for_status()
    _rate_limit_sleep()
    return _parse_efetch_xml(resp.text)


def ingest_gene_queries(
    queries: Sequence[str],
    output_dir: Path = Path("data/literature"),
    max_per_query: int = 10,
    api_key: Optional[str] = None,
) -> list[PubMedRecord]:
    """Fetch, deduplicate, and store PubMed abstracts for multiple queries.

    Writes ``abstracts.json`` in *output_dir*.  Returns all records.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    all_records: list[PubMedRecord] = []
    for query in queries:
        pmids = search_pubmed(query, max_results=max_per_query, api_key=api_key)
        new_pmids = [p for p in pmids if p not in seen]
        if not new_pmids:
            continue
        records = fetch_abstracts(new_pmids, api_key=api_key)
        for rec in records:
            if rec.pmid not in seen:
                seen.add(rec.pmid)
                all_records.append(rec)
    out_path = output_dir / "abstracts.json"
    with out_path.open("w") as fh:
        json.dump([asdict(r) for r in all_records], fh, indent=2)
    return all_records


def load_stored_abstracts(path: Path = Path("data/literature/abstracts.json")) -> list[PubMedRecord]:
    """Load previously stored abstracts from JSON."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as fh:
        raw = json.load(fh)
    return [PubMedRecord(**r) for r in raw]

