"""
Lightweight RAG layer: fetch recent PubMed abstracts on a mitigation topic
(e.g. "3-NOP methane dairy cattle") via NCBI E-utilities, then use Claude to
summarize the current evidence with citations.

Requires internet access (NCBI E-utilities) and an ANTHROPIC_API_KEY
environment variable. Run this on your own machine / CI, not inside a
network-restricted sandbox.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python src/rag_pubmed.py --topic "3-NOP methane dairy cattle" --max-results 8
"""

import argparse
import os
import time
import xml.etree.ElementTree as ET

import requests

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-4-6"


def esearch_pubmed(query: str, max_results: int = 8) -> list[str]:
    """Return a list of PubMed IDs matching the query, most recent first."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "sort": "date",
        "retmode": "json",
    }
    resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()["esearchresult"]["idlist"]


def efetch_abstracts(pmids: list[str]) -> list[dict]:
    """Fetch title + abstract text for a list of PubMed IDs."""
    if not pmids:
        return []
    params = {"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"}
    resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_texts = article.findall(".//AbstractText")
        year_el = article.find(".//PubDate/Year")

        pmid = pmid_el.text if pmid_el is not None else "unknown"
        title = "".join(title_el.itertext()) if title_el is not None else ""
        abstract = " ".join("".join(a.itertext()) for a in abstract_texts)
        year = year_el.text if year_el is not None else "n.d."

        records.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "year": year,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return records


def summarize_with_claude(topic: str, records: list[dict]) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY to use the summarization step.")

    sources_block = "\n\n".join(
        f"[{i+1}] {r['title']} ({r['year']}) - {r['url']}\nAbstract: {r['abstract']}"
        for i, r in enumerate(records)
    )

    system_prompt = (
        "You are a scientific evidence summarizer for an animal nutrition research tool. "
        "Summarize the CURRENT evidence on the given mitigation topic using ONLY the "
        "provided abstracts. Cite sources by their bracket number [1], [2], etc. "
        "If evidence is thin, contradictory, or insufficient, say so explicitly rather "
        "than overstating confidence. Do not invent findings not present in the abstracts. "
        "End with a one-line evidence-strength rating: Strong / Moderate / Weak / Insufficient."
    )

    user_prompt = f"Mitigation topic: {topic}\n\nSources:\n\n{sources_block}"

    resp = requests.post(
        ANTHROPIC_API_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": 800,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return "".join(block.get("text", "") for block in data.get("content", []))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help='e.g. "3-NOP methane dairy cattle"')
    parser.add_argument("--max-results", type=int, default=8)
    args = parser.parse_args()

    print(f"Searching PubMed for: {args.topic}")
    pmids = esearch_pubmed(args.topic, args.max_results)
    print(f"Found {len(pmids)} candidate articles, fetching abstracts...")
    time.sleep(0.34)  # be polite to NCBI's rate limit (max ~3 req/sec without an API key)
    records = efetch_abstracts(pmids)
    records = [r for r in records if r["abstract"]]  # drop entries with no abstract text

    if not records:
        print("No abstracts with usable text found. Try a broader topic.")
        return

    print(f"Summarizing {len(records)} abstracts with Claude...\n")
    summary = summarize_with_claude(args.topic, records)
    print(summary)


if __name__ == "__main__":
    main()
