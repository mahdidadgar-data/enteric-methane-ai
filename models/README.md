# Saved demonstration model

`best_model.joblib` is the existing trained model used by `app/app.py`.
It is included so the Streamlit demonstration can run without training a
model on the hosting server. The artifact has not been retrained or modified.

- Model: XGBoost regressor, 300 boosting rounds
- Input features: 28, with their names and order stored in the bundle
- Target: `ch4_g_kg_dmi`
- XGBoost version recorded in the artifact: 3.3.0
- Deployment Python version: 3.12
- SHA-256: `c3b3080de60f0f0e587385c2d275936ae604c143280ec53ebe5787ed4956e355`

Install `app/requirements.txt` in a separate environment for inference.
The CPU-only XGBoost package provides the same Python import, `xgboost`,
without requiring GPU support.

## Scope and limitations

This artifact is for a research and portfolio demonstration, not validated
on-farm advice. The bundle does not record the training dataset's provenance.
The repository includes a synthetic-data workflow; the presence of a saved
model must not be interpreted as evidence of validation on real experiments.
Runtime tests establish that the app can load the model and make predictions,
not that those predictions are scientifically accurate.

The optional explainability CSVs and saved evidence briefs are not embedded
in this file. Until those outputs are supplied, their app sections display
availability messages.

Only load Joblib/pickle artifacts from a trusted source. Do not add an
unrestricted model-upload feature to the public app. If upgrading XGBoost,
first verify compatibility with this artifact; consider exporting it with
XGBoost's native `save_model` format for longer-term portability.
