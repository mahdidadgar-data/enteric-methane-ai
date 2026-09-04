# Saved-model demonstration outputs

These outputs were newly generated for the hosted demo; they are not recovered
copies of a previous desktop analysis. The saved model was not retrained.

## Feature importance

`feature_importance.csv` contains all 28 input features. Importance is average
XGBoost split gain, normalized to sum to 1 (100%). This is not SHAP, a local
explanation, an effect size or evidence of causality. Correlated predictors can
share or compete for importance, and an unused feature can have zero gain.

## Model sensitivity

`mitigation_opportunities.csv` contains 10 model-input probes applied to 27
synthetic reference profiles. The grid combines three intake levels, three
forage proportions and three milk yields. Fixed values, grid values, the model
checksum and limitations are recorded in `demo_metadata.json`.

The table reports scenario mean minus baseline mean in **g CH4/kg DMI**.
Negative means a lower model prediction. Rank is only a sort order, not a
ranking of proven mitigation efficacy. Every scenario starts from the same
no-additive reference profiles; scenarios are not applied cumulatively.

Forage and concentrate remain complementary. Milk-per-DMI is recalculated by
the same input encoder used in the app. These are illustrative probes, not
fully balanced rations, real farm observations, training rows, or a sample from
which to infer population outcomes. A normalized additive dose of 1 is not a
common physical dose and does not make additives biologically comparable.

The outputs are precomputed and do not respond to the prediction form. They do
not validate the model against experimental methane measurements.

## Reproduce

Use Python 3.12 and `app/requirements.txt`, then run from the repository root:

```bash
python src/build_demo_outputs.py
```

The script verifies the model checksum and writes only the two CSVs and their
metadata. It does not read private data, retrain the model, or change literature
briefs. Review the provenance if replacing the model.
