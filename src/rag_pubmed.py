"""
PubMed evidence-retrieval layer for the enteric methane AI project.

This script searches PubMed for scientific abstracts related to an enteric
methane mitigation topic and saves a citation-aware evidence brief.

It supports two modes:

    1. Retrieval-only mode
       - Searches PubMed
       - Fetches article metadata and abstracts
       - Saves records as CSV
       - Creates a transparent non-LLM evidence brief

    2. Optional LLM summary mode
       - Uses the retrieved abstracts only
       - Requires ANTHROPIC_API_KEY
       - Requires an Anthropic model name via --anthropic-model or ANTHROPIC_MODEL
       - Produces a citation-based evidence summary

Example:

    python src/rag_pubmed.py --topic "3-NOP methane dairy cattle" --max-results 8

Optional LLM mode:

    set ANTHROPIC_API_KEY=your_key_here
    set ANTHROPIC_MODEL=your_model_name_here
    python src/rag_pubmed.py --topic "3-NOP methane dairy cattle" --max-results 8 --use-llm

Important:
    - This layer uses PubMed abstracts, not full-text papers.
    - The summary should not be treated as a systematic review.
    - Any mitigation recommendation must be checked against animal performance,
      health, feed formulation, farm context, and full scientific evidence.
"""

from __future__ import annotations

import argparse
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "rag"

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

DEFAULT_MAX_RESULTS = 8
REQUEST_TIMEOUT = 30
NCBI_DELAY_SECONDS = 0.34


# ---------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------

@dataclass
class PubMedRecord:
    """Container for one PubMed article record."""

    pmid: str
    title: str
    abstract: str
    year: str
    journal: str
    doi: str
    url: str


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def safe_filename(text: str, max_length: int = 70) -> str:
    """
    Create a filesystem-safe name from a topic string.
    """

    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:max_length] or "pubmed_topic"


def xml_text(element: Optional[ET.Element]) -> str:
    """
    Extract all text from an XML element safely.
    """

    if element is None:
        return ""

    return " ".join("".join(element.itertext()).split())


def parse_year(article: ET.Element) -> str:
    """
    Extract publication year from a PubMedArticle XML element.
    """

    year_el = article.find(".//PubDate/Year")
    if year_el is not None and year_el.text:
        return year_el.text

    article_date_year = article.find(".//ArticleDate/Year")
    if article_date_year is not None and article_date_year.text:
        return article_date_year.text

    medline_date = article.find(".//PubDate/MedlineDate")
    if medline_date is not None and medline_date.text:
        match = re.search(r"(19|20)\d{2}", medline_date.text)
        if match:
            return match.group(0)

    return "n.d."


def parse_doi(article: ET.Element) -> str:
    """
    Extract DOI from a PubMedArticle XML element when available.
    """

    for el in article.findall(".//ELocationID"):
        if el.attrib.get("EIdType", "").lower() == "doi" and el.text:
            return el.text.strip()

    for el in article.findall(".//ArticleId"):
        if el.attrib.get("IdType", "").lower() == "doi" and el.text:
            return el.text.strip()

    return ""


def build_eutils_params(extra_params: Dict[str, object]) -> Dict[str, object]:
    """
    Build NCBI E-utilities parameters.

    Users can optionally set:
        NCBI_EMAIL
        NCBI_API_KEY
    """

    params: Dict[str, object] = {
        "tool": "enteric_methane_ai_portfolio",
    }

    email = os.environ.get("NCBI_EMAIL")
    api_key = os.environ.get("NCBI_API_KEY")

    if email:
        params["email"] = email

    if api_key:
        params["api_key"] = api_key

    params.update(extra_params)
    return params


def get_with_retries(
    url: str,
    params: Dict[str, object],
    timeout: int = REQUEST_TIMEOUT,
    max_retries: int = 3,
) -> requests.Response:
    """
    GET request with lightweight retry logic.
    """

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt == max_retries:
                break
            sleep_time = attempt * 1.5
            print(f"Request failed; retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    raise RuntimeError(f"Request failed after {max_retries} attempts: {last_error}")


# ---------------------------------------------------------------------
# PubMed retrieval
# ---------------------------------------------------------------------

def esearch_pubmed(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> List[str]:
    """
    Return PubMed IDs matching a query, sorted by most recent first.
    """

    if max_results < 1:
        raise ValueError("max_results must be at least 1.")

    params = build_eutils_params(
        {
            "db": "pubmed",
            "term": query,
            "retmax": int(max_results),
            "sort": "date",
            "retmode": "json",
        }
    )

    response = get_with_retries(f"{EUTILS_BASE}/esearch.fcgi", params=params)
    data = response.json()

    return data.get("esearchresult", {}).get("idlist", [])


def efetch_abstracts(pmids: List[str]) -> List[PubMedRecord]:
    """
    Fetch article metadata and abstracts for a list of PubMed IDs.
    """

    if not pmids:
        return []

    params = build_eutils_params(
        {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
        }
    )

    response = get_with_retries(f"{EUTILS_BASE}/efetch.fcgi", params=params)
    root = ET.fromstring(response.content)

    records: List[PubMedRecord] = []

    for article in root.findall(".//PubmedArticle"):
        pmid = xml_text(article.find(".//PMID")) or "unknown"
        title = xml_text(article.find(".//ArticleTitle"))
        journal = xml_text(article.find(".//Journal/Title"))
        year = parse_year(article)
        doi = parse_doi(article)

        abstract_parts = []
        for abstract_el in article.findall(".//AbstractText"):
            label = abstract_el.attrib.get("Label", "")
            text = xml_text(abstract_el)

            if not text:
                continue

            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)

        abstract = " ".join(abstract_parts)

        records.append(
            PubMedRecord(
                pmid=pmid,
                title=title,
                abstract=abstract,
                year=year,
                journal=journal,
                doi=doi,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            )
        )

    return records


def filter_records_with_abstracts(records: List[PubMedRecord]) -> List[PubMedRecord]:
    """
    Keep only records with usable abstract text.
    """

    return [record for record in records if record.abstract.strip()]


# ---------------------------------------------------------------------
# Evidence brief generation
# ---------------------------------------------------------------------

def format_sources_block(records: List[PubMedRecord]) -> str:
    """
    Format retrieved records for LLM summarization.
    """

    chunks = []

    for index, record in enumerate(records, start=1):
        chunks.append(
            "\n".join(
                [
                    f"[{index}] {record.title} ({record.year})",
                    f"Journal: {record.journal}",
                    f"PMID: {record.pmid}",
                    f"DOI: {record.doi or 'not available'}",
                    f"URL: {record.url}",
                    f"Abstract: {record.abstract}",
                ]
            )
        )

    return "\n\n".join(chunks)


def build_retrieval_only_brief(topic: str, records: List[PubMedRecord]) -> str:
    """
    Build a transparent evidence brief without using an LLM.

    This is intentionally conservative: it lists the retrieved evidence and
    avoids generating claims that are not manually verified.
    """

    lines = [
        f"# PubMed Evidence Brief: {topic}",
        "",
        "## Scope",
        "",
        "This brief was generated from retrieved PubMed abstracts only. It is not a systematic review and does not replace reading the full papers.",
        "",
        "## Retrieved Sources",
        "",
    ]

    for index, record in enumerate(records, start=1):
        lines.extend(
            [
                f"### [{index}] {record.title}",
                "",
                f"- **Year:** {record.year}",
                f"- **Journal:** {record.journal or 'not available'}",
                f"- **PMID:** {record.pmid}",
                f"- **DOI:** {record.doi or 'not available'}",
                f"- **URL:** {record.url}",
                "",
                "**Abstract excerpt:**",
                "",
                record.abstract[:1200] + ("..." if len(record.abstract) > 1200 else ""),
                "",
            ]
        )

    lines.extend(
        [
            "## Conservative Interpretation Notes",
            "",
            "- The retrieved abstracts can help identify relevant scientific evidence, but full-text review is needed before drawing firm conclusions.",
            "- Evidence strength depends on study design, animal category, diet context, dose, duration, methane measurement method, and production-performance outcomes.",
            "- The project should report insufficient or mixed evidence when abstracts do not clearly support a mitigation recommendation.",
            "",
            "## Evidence-Strength Rating",
            "",
            "Not rated automatically in retrieval-only mode.",
        ]
    )

    return "\n".join(lines)


def summarize_with_anthropic(
    topic: str,
    records: List[PubMedRecord],
    model_name: Optional[str] = None,
    max_tokens: int = 900,
) -> str:
    """
    Summarize retrieved abstracts using Anthropic Messages API.

    The model must use only the provided abstracts and cite sources by bracket
    number.
    """

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Use retrieval-only mode or set the key."
        )

    selected_model = model_name or os.environ.get("ANTHROPIC_MODEL")

    if not selected_model:
        raise RuntimeError(
            "No Anthropic model was provided. Set ANTHROPIC_MODEL or pass "
            "--anthropic-model."
        )

    sources_block = format_sources_block(records)

    system_prompt = (
        "You are a scientific evidence summarizer for an animal nutrition "
        "research tool. Use ONLY the provided PubMed abstracts. Cite sources "
        "by bracket number such as [1] or [2]. If evidence is thin, mixed, "
        "indirect, or insufficient, say so explicitly. Do not invent findings, "
        "do not cite sources that were not provided, and do not present the "
        "summary as a systematic review."
    )

    user_prompt = (
        f"Mitigation topic: {topic}\n\n"
        "Prepare a concise evidence summary with these sections:\n"
        "1. Key findings\n"
        "2. Evidence supporting methane reduction\n"
        "3. Animal performance or safety considerations\n"
        "4. Important limitations\n"
        "5. Practical interpretation for a research prototype\n"
        "6. Evidence-strength rating: Strong / Moderate / Weak / Insufficient\n\n"
        f"Sources:\n\n{sources_block}"
    )

    response = requests.post(
        ANTHROPIC_API_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": selected_model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        timeout=60,
    )

    response.raise_for_status()
    data = response.json()

    content_blocks = data.get("content", [])
    summary_parts = []

    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            summary_parts.append(block.get("text", ""))

    return "\n".join(summary_parts).strip()


def build_llm_brief(topic: str, records: List[PubMedRecord], summary: str) -> str:
    """
    Wrap LLM summary with source metadata and responsible-use notes.
    """

    lines = [
        f"# PubMed Evidence Summary: {topic}",
        "",
        "## LLM Summary",
        "",
        summary,
        "",
        "## Retrieved Sources",
        "",
    ]

    for index, record in enumerate(records, start=1):
        lines.extend(
            [
                f"- [{index}] {record.title} ({record.year}). PMID: {record.pmid}. {record.url}",
            ]
        )

    lines.extend(
        [
            "",
            "## Responsible-Use Notes",
            "",
            "- This summary is based only on retrieved abstracts.",
            "- It is not a systematic review.",
            "- Full-text review is required before making scientific, commercial, or on-farm decisions.",
            "- Recommendations should consider animal performance, health, economics, and local feed context.",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Saving outputs
# ---------------------------------------------------------------------

def save_records(records: List[PubMedRecord], path: Path) -> None:
    """
    Save PubMed records to CSV.
    """

    records_df = pd.DataFrame([asdict(record) for record in records])
    records_df.to_csv(path, index=False)


def save_text(text: str, path: Path) -> None:
    """
    Save text output with UTF-8 encoding.
    """

    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------
# CLI workflow
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Retrieve PubMed abstracts and create an evidence brief."
    )

    parser.add_argument(
        "--topic",
        required=True,
        help='Search topic, e.g. "3-NOP methane dairy cattle".',
    )

    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help=f"Maximum number of PubMed records to retrieve. Default: {DEFAULT_MAX_RESULTS}.",
    )

    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use Anthropic API to summarize retrieved abstracts.",
    )

    parser.add_argument(
        "--anthropic-model",
        default=None,
        help="Anthropic model name. Alternatively set ANTHROPIC_MODEL.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help=f"Output directory. Default: {OUTPUT_DIR}",
    )

    return parser.parse_args()


def main() -> None:
    """
    Run PubMed retrieval and evidence-brief generation.
    """

    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topic_slug = safe_filename(args.topic)
    records_path = output_dir / f"pubmed_records_{topic_slug}.csv"
    brief_path = output_dir / f"evidence_brief_{topic_slug}.md"

    print(f"Searching PubMed for: {args.topic}")
    pmids = esearch_pubmed(args.topic, max_results=args.max_results)

    if not pmids:
        print("No PubMed records found. Try a broader or different topic.")
        return

    print(f"Found {len(pmids)} candidate articles.")
    print("Fetching abstracts...")

    time.sleep(NCBI_DELAY_SECONDS)
    records = efetch_abstracts(pmids)
    records = filter_records_with_abstracts(records)

    if not records:
        print("No records with usable abstract text found. Try a broader topic.")
        return

    save_records(records, records_path)

    if args.use_llm:
        print("Creating LLM evidence summary from retrieved abstracts...")
        summary = summarize_with_anthropic(
            topic=args.topic,
            records=records,
            model_name=args.anthropic_model,
        )
        brief = build_llm_brief(args.topic, records, summary)
    else:
        print("Creating retrieval-only evidence brief...")
        brief = build_retrieval_only_brief(args.topic, records)

    save_text(brief, brief_path)

    print("")
    print(f"Saved PubMed records to: {records_path}")
    print(f"Saved evidence brief to: {brief_path}")
    print("")
    print("Preview:")
    print("-" * 72)
    print(brief[:2000])
    if len(brief) > 2000:
        print("...")


if __name__ == "__main__":
    main()
