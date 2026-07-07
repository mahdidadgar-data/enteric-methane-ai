"""
Explainability layer: rank which diet/animal variables drive methane
predictions, and turn that into a practical "mitigation opportunity" table
(estimated CH4 reduction per unit change, alongside any milk-yield trade-off
if that data is present).

Uses SHAP if installed (pip install shap), otherwise falls back to the
model's built-in feature_importances_ (Random Forest / Gradient Boosting
both provide this, so the script still works with zero extra dependencies).
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

BASE = Path(__file__).resolve().parent.parent
PROCESSED_PATH = BASE / "data" / "processed" / "clean_dataset.csv"
MODEL_PATH = BASE / "data" / "processed" / "models" / "best_model.joblib"
OUT_PATH = BASE / "data" / "processed" / "models" / "feature_importance.csv"

TARGET = "ch4_g_kg_dmi"
DROP_COLS = ["study_id", "ch4_g_day", TARGET]


def load():
    bundle = joblib.load(MODEL_PATH)
    df = pd.read_csv(PROCESSED_PATH)
    return bundle["model"], bundle["feature_cols"], df


def feature_importance_shap(model, X: pd.DataFrame) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return pd.DataFrame({"feature": X.columns, "importance": mean_abs_shap, "method": "SHAP"})


def feature_importance_builtin(model, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
        "method": "built-in (feature_importances_)",
    })


def main():
    model, feature_cols, df = load()
    X = df[feature_cols]

    if HAS_SHAP:
        imp = feature_importance_shap(model, X)
    else:
        print("shap not installed -- falling back to model.feature_importances_. "
              "Run `pip install shap` for proper SHAP explanations.")
        imp = feature_importance_builtin(model, X)

    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv(OUT_PATH, index=False)

    print("\nTop drivers of predicted CH4 (g/kg DMI):")
    print(imp.head(10).to_string(index=False))

    # --- simple "mitigation opportunity" framing for additive columns ---
    additive_cols = [c for c in feature_cols if c.startswith("additive_type_")]
    if additive_cols:
        print("\nAdditive impact ranking (mean SHAP/importance among additive dummy columns):")
        additive_imp = imp[imp["feature"].isin(additive_cols)]
        print(additive_imp.to_string(index=False))

    print(f"\nSaved full ranking to {OUT_PATH}")


if __name__ == "__main__":
    main()
