# Scientific AI Assistant for Sustainable Ruminant Nutrition — Enteric Methane Prediction, Explainability & Evidence Retrieval

A three-layer AI system that predicts enteric methane (CH4) emissions from ruminant diet
and animal traits, explains *why* the model predicts what it predicts, and grounds
mitigation recommendations in current scientific literature via retrieval-augmented
generation (RAG).

Built to demonstrate the intersection of PhD-level animal nutrition domain knowledge with
modern data science / AI engineering.

## Why this project

Enteric fermentation from ruminant livestock is the single largest agricultural source of
methane, a potent greenhouse gas. Reducing it — without hurting milk/meat yield — is one of
the most active applied research areas in animal science right now. Most published models
either (a) use classic empirical/mechanistic equations (IPCC Tier 2 style) or (b) use ML
without explaining or contextualizing predictions. This project combines all three:

1. **Predict** — ML models (Random Forest, Gradient Boosting / XGBoost) trained on diet
   composition and animal traits, benchmarked against a published empirical equation.
2. **Explain** — feature-importance / SHAP analysis to rank which diet and animal variables
   drive methane output, turned into a ranked "mitigation opportunity" table.
3. **Ground in evidence** — a lightweight RAG pipeline that pulls recent PubMed abstracts on
   whatever mitigation strategy the model flags (e.g., 3-NOP, seaweed/Asparagopsis, tannins,
   nitrate) and summarizes current evidence with citations, so recommendations don't go stale.

## Data

This project is designed around the structure of a real, published dataset:

> **A global dataset of enteric methane mitigation experiments with lactating and
> non-lactating dairy cows conducted from 1963 to 2022** (797 records, 162 variables,
> compiled from 213 peer-reviewed publications). Published as a data article
> (ScienceDirect / Mendeley Data, 2023).

**You need to download that dataset yourself** (this environment has no internet access)
and place the raw Excel/CSV file in `data/raw/`. Search "A global dataset of enteric methane
mitigation experiments dairy cows Mendeley" to find the download page — it's open access.

Until then, `src/generate_synthetic_data.py` creates a synthetic dataset with the same
column names and realistic value ranges (drawn from the published literature — e.g. CP
42–276 g/kg DM, NDF 93–776 g/kg DM) so you can run and test the entire pipeline right now
and swap in real data later without changing any downstream code.

## Project structure

```
methane-ai/
├── data/
│   ├── raw/              # put the real Mendeley dataset here (not included)
│   ├── processed/        # cleaned data, written by data_prep.py
│   └── README.md         # data dictionary + provenance notes
├── src/
│   ├── generate_synthetic_data.py   # placeholder data so pipeline runs end-to-end today
│   ├── data_prep.py                 # cleaning, missing values, feature engineering
│   ├── benchmark_equation.py        # published empirical CH4 equation for comparison
│   ├── train_models.py              # RF + Gradient Boosting/XGBoost training & evaluation
│   ├── explain.py                   # feature importance / SHAP + mitigation ranking table
│   └── rag_pubmed.py                # PubMed fetch + retrieval + LLM evidence summary
├── app/
│   └── app.py             # Streamlit app tying all three layers together
├── notebooks/
│   └── 01_eda.py           # exploratory data analysis (run as script or paste into Jupyter)
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt

# 1. Get data (synthetic for now, real dataset when you have it)
python src/generate_synthetic_data.py

# 2. Clean + engineer features
python src/data_prep.py

# 3. Train and benchmark models
python src/train_models.py

# 4. Explainability + mitigation ranking
python src/explain.py

# 5. (optional, needs internet + an Anthropic API key) evidence retrieval
export ANTHROPIC_API_KEY=...
python src/rag_pubmed.py --topic "3-NOP methane dairy cattle"

# 6. Launch the app
streamlit run app/app.py
```

## Responsible AI notes

- Predictions are only as good as the training data's diet/breed/region coverage — the app
  should flag low-confidence predictions when inputs fall outside the training distribution.
- The RAG layer must always cite its sources and should say "insufficient evidence" rather
  than fabricate a recommendation when literature coverage is thin.
- This is a research/portfolio tool, not a validated on-farm decision system — the README and
  app UI say so explicitly.

## Next extensions (stretch goals)

- Add a rumen microbiome module (biomarker discovery) if a public microbiome dataset is
  obtained — several published studies (Holstein cow datasets) use Random Forest regression
  to link microbial taxa to methane output.
- Add a poultry track (feed efficiency / nitrogen excretion) — a much less saturated research
  niche than ruminant methane, and a good way to show breadth across species.
- Add MIR-spectroscopy-based prediction as a proxy layer, since milk fatty acid profiles from
  routine mid-infrared spectroscopy have been shown to predict methane output and could be a
  cheap, deployable version of this model for commercial dairies.
