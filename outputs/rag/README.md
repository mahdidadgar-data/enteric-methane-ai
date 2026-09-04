# Saved literature snapshots

The three `evidence_brief_*.md` files are curated literature snapshots checked
on 4 September 2026 against six selected PubMed records. Abstracts were accessed
through the project's existing `rag_pubmed.efetch_abstracts` function and NCBI
E-utilities. The briefs contain concise paraphrases and direct source links;
full abstracts are not redistributed here.

Selection is illustrative: two dairy 3-NOP experiments, beef and dairy
Asparagopsis experiments, and two dairy nitrate experiments. This is not a
systematic or comprehensive review, a current regulatory assessment, a formal
evidence grade, or automatic validation of the saved model. No full-text
risk-of-bias assessment has been completed for these snapshots.

The Streamlit app reads the committed Markdown files. It performs no live
PubMed search and makes no paid API calls. `src/rag_pubmed.py` remains available
as a separate retrieval workflow. New retrieval results require review before
publication; do not replace these snapshots with unreviewed full abstracts.

## Source identifiers

| Brief | PubMed IDs | DOIs |
| --- | --- | --- |
| 3-NOP | 26229078; 33131815 | 10.1073/pnas.1504124112; 10.3168/jds.2020-18908 |
| Asparagopsis | 33730064; 33516546 | 10.1371/journal.pone.0247820; 10.3168/jds.2020-19686 |
| Nitrate | 21787938; 27236758 | 10.3168/jds.2011-4236; 10.3168/jds.2015-10691 |
