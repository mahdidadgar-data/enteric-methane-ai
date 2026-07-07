"""
Explainability layer for the enteric methane AI project.

This script explains which diet, animal, and additive variables drive methane
predictions from the trained model.

It supports three levels of interpretation:

    1. Global feature importance
       - SHAP mean absolute value if SHAP is installed
       - permutation importance fallback if SHAP is unavailable
       - built-in tree feature importance as a last fallback

    2. Local explanation example
       - explains one representative row from the cleaned dataset

    3. Mitigation-opportunity sensitivity table
       - estimates how model predictions change under controlled feature
         changes such as lower forage percentage, higher dietary fat, or
         additive category changes.

Inputs:
    data/processed/clean_dataset.csv
    models/best_model.joblib

Outputs:
    outputs/explainability/feature_importance.csv
    outputs/explainability/local_explanation_example.csv
    outputs/explainability/mitigation_opportunities.csv
    outputs/explainability/explanation_report.txt

Run:
    python src/explain.py

Important:
    This is an interpretability and portfolio-demonstration layer. The
    mitigation sensitivity table is not a validated nutrition recommendation.
    It shows how the trained model responds to controlled input changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ---------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_PATH = BASE_DIR / "data" / "processed" / "clean_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
OUTPUT_DIR = BASE_DIR / "outputs" / "explainability"

FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
LOCAL_EXPLANATION_PATH = OUTPUT_DIR / "local_explanation_example.csv"
MITIGATION_OPPORTUNITY_PATH = OUTPUT_DIR / "mitigation_opportunities.csv"
EXPLANATION_REPORT_PATH = OUTPUT_DIR / "explanation_report.txt"

TARGET = "ch4_g_kg_dmi"

DROP_COLS = [
    "study_id",
    "ch4_g_day",
    "ch4_g_day_benchmark",
    "ch4_g_kg_dmi_benchmark",
    TARGET,
]

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_model_bundle(path: Path = MODEL_PATH) -> Dict[str, object]:
    """
    Load the trained model bundle saved by train_models.py.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {path}\n"
            "Run `python src/train_models.py` first."
        )

    return joblib.load(path)


def load_clean_data(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """
    Load the cleaned dataset.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found at: {path}\n"
            "Run `python src/data_prep.py` first."
        )

    return pd.read_csv(path)


def prepare_explanation_data(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X and y for explanation.
    """

    missing_features = [col for col in feature_cols if col not in df.columns]

    if missing_features:
        raise ValueError(
            "The cleaned dataset is missing features expected by the model: "
            f"{missing_features}"
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found in cleaned dataset."
        )

    X = df[feature_cols].copy()
    y = pd.to_numeric(df[TARGET], errors="coerce")

    valid_mask = y.notna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    return X, y


# ---------------------------------------------------------------------
# Feature importance methods
# ---------------------------------------------------------------------

def feature_importance_shap(model: object, X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate SHAP mean absolute feature importance.
    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Some model types return a list. Regression usually returns an array.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    return pd.DataFrame(
        {
            "feature": X.columns,
            "importance": mean_abs_shap,
            "method": "SHAP mean absolute value",
        }
    )


def feature_importance_permutation(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """
    Calculate permutation importance using negative RMSE as scoring.
    """

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
        n_jobs=None,
    )

    return pd.DataFrame(
        {
            "feature": X.columns,
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
            "method": "permutation importance",
        }
    )


def feature_importance_builtin(model: object, X: pd.DataFrame) -> pd.DataFrame:
    """
    Use built-in tree feature importance if available.
    """

    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            "The model does not expose feature_importances_. "
            "Install SHAP or use a model compatible with permutation importance."
        )

    return pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
            "method": "built-in feature_importances_",
        }
    )


def calculate_feature_importance(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """
    Calculate feature importance using the best available method.
    """

    if HAS_SHAP:
        importance_df = feature_importance_shap(model, X)
    else:
        try:
            print(
                "SHAP is not installed. Falling back to permutation importance."
            )
            importance_df = feature_importance_permutation(model, X, y)
        except Exception as exc:
            print(
                "Permutation importance failed. Falling back to built-in "
                f"feature_importances_. Reason: {exc}"
            )
            importance_df = feature_importance_builtin(model, X)

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    importance_df["rank"] = np.arange(1, len(importance_df) + 1)

    total_importance = importance_df["importance"].sum()
    if total_importance > 0:
        importance_df["relative_importance_pct"] = (
            importance_df["importance"] / total_importance * 100
        ).round(2)
    else:
        importance_df["relative_importance_pct"] = 0.0

    return importance_df


# ---------------------------------------------------------------------
# Local explanation
# ---------------------------------------------------------------------

def local_explanation_builtin(
    model: object,
    X: pd.DataFrame,
    row_index: int,
) -> pd.DataFrame:
    """
    Approximate a local explanation using feature values weighted by global
    tree feature importance.

    This is only used when SHAP is unavailable.
    """

    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            "Local fallback requires a model with feature_importances_."
        )

    row = X.iloc[row_index]
    scaled_values = (row - X.mean()) / X.std(ddof=0).replace(0, np.nan)
    scaled_values = scaled_values.fillna(0)

    contribution_proxy = scaled_values * model.feature_importances_

    return pd.DataFrame(
        {
            "feature": X.columns,
            "feature_value": row.values,
            "local_contribution_proxy": contribution_proxy.values,
            "method": "standardized value x global importance",
        }
    ).sort_values(
        "local_contribution_proxy",
        key=lambda s: s.abs(),
        ascending=False,
    )


def calculate_local_explanation(
    model: object,
    X: pd.DataFrame,
    row_index: int = 0,
) -> pd.DataFrame:
    """
    Explain one representative prediction.
    """

    row_index = min(max(row_index, 0), len(X) - 1)

    if HAS_SHAP:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X.iloc[[row_index]])

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        local_df = pd.DataFrame(
            {
                "feature": X.columns,
                "feature_value": X.iloc[row_index].values,
                "local_contribution": shap_values[0],
                "method": "SHAP local value",
            }
        ).sort_values(
            "local_contribution",
            key=lambda s: s.abs(),
            ascending=False,
        )

        return local_df

    return local_explanation_builtin(model, X, row_index)


# ---------------------------------------------------------------------
# Mitigation opportunity sensitivity analysis
# ---------------------------------------------------------------------

def predict_mean(model: object, X: pd.DataFrame) -> float:
    """
    Return average model prediction for a dataframe.
    """

    return float(np.mean(model.predict(X)))


def apply_continuous_intervention(
    X: pd.DataFrame,
    feature: str,
    change: float,
    lower: float | None = None,
    upper: float | None = None,
) -> pd.DataFrame:
    """
    Apply an additive change to a continuous feature.
    """

    X_new = X.copy()

    if feature not in X_new.columns:
        return X_new

    X_new[feature] = X_new[feature] + change

    if lower is not None or upper is not None:
        X_new[feature] = X_new[feature].clip(lower=lower, upper=upper)

    return X_new


def apply_additive_category_intervention(
    X: pd.DataFrame,
    additive_col: str,
) -> pd.DataFrame:
    """
    Set one additive dummy column to 1 and all other additive_type dummy
    columns to 0.
    """

    X_new = X.copy()

    additive_cols = [
        col for col in X_new.columns if col.startswith("additive_type_")
    ]

    if additive_col not in additive_cols:
        return X_new

    for col in additive_cols:
        X_new[col] = 0

    X_new[additive_col] = 1

    # Use a moderate positive dose for additive scenarios if dose exists and
    # the intervention is not the "none" category.
    if "additive_dose" in X_new.columns:
        if additive_col == "additive_type_none":
            X_new["additive_dose"] = 0.0
        else:
            X_new["additive_dose"] = np.maximum(X_new["additive_dose"], 1.0)

    return X_new


def calculate_mitigation_opportunities(
    model: object,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate average prediction changes under controlled feature interventions.

    Negative delta values mean the model predicts lower methane yield after
    the intervention.
    """

    baseline_mean = predict_mean(model, X)
    scenarios = []

    # Continuous nutrition scenarios.
    continuous_scenarios = [
        {
            "scenario": "Reduce forage proportion by 10 percentage points",
            "feature_changed": "forage_pct",
            "change_tested": "-10",
            "function": lambda frame: apply_continuous_intervention(
                frame,
                feature="forage_pct",
                change=-10,
                lower=0,
                upper=100,
            ),
        },
        {
            "scenario": "Increase concentrate proportion by 10 percentage points",
            "feature_changed": "concentrate_pct",
            "change_tested": "+10",
            "function": lambda frame: apply_continuous_intervention(
                frame,
                feature="concentrate_pct",
                change=10,
                lower=0,
                upper=100,
            ),
        },
        {
            "scenario": "Increase dietary fat/ether extract by 10 g/kg DM",
            "feature_changed": "ee_g_kg",
            "change_tested": "+10",
            "function": lambda frame: apply_continuous_intervention(
                frame,
                feature="ee_g_kg",
                change=10,
                lower=0,
                upper=120,
            ),
        },
        {
            "scenario": "Reduce NDF by 50 g/kg DM",
            "feature_changed": "ndf_g_kg",
            "change_tested": "-50",
            "function": lambda frame: apply_continuous_intervention(
                frame,
                feature="ndf_g_kg",
                change=-50,
                lower=50,
                upper=850,
            ),
        },
        {
            "scenario": "Increase starch by 50 g/kg DM",
            "feature_changed": "starch_g_kg",
            "change_tested": "+50",
            "function": lambda frame: apply_continuous_intervention(
                frame,
                feature="starch_g_kg",
                change=50,
                lower=0,
                upper=550,
            ),
        },
    ]

    for scenario in continuous_scenarios:
        feature = scenario["feature_changed"]

        if feature not in X.columns:
            continue

        X_intervention = scenario["function"](X)
        intervention_mean = predict_mean(model, X_intervention)
        delta = intervention_mean - baseline_mean

        scenarios.append(
            {
                "scenario": scenario["scenario"],
                "feature_changed": feature,
                "change_tested": scenario["change_tested"],
                "baseline_mean_prediction": round(baseline_mean, 4),
                "intervention_mean_prediction": round(intervention_mean, 4),
                "estimated_delta_g_ch4_kg_dmi": round(delta, 4),
                "interpretation": (
                    "Potential methane reduction signal"
                    if delta < 0
                    else "No reduction signal in trained model"
                ),
            }
        )

    # Additive category scenarios.
    additive_cols = [
        col for col in X.columns if col.startswith("additive_type_")
    ]

    for additive_col in additive_cols:
        if additive_col == "additive_type_none":
            continue

        label = additive_col.replace("additive_type_", "")

        X_intervention = apply_additive_category_intervention(X, additive_col)
        intervention_mean = predict_mean(model, X_intervention)
        delta = intervention_mean - baseline_mean

        scenarios.append(
            {
                "scenario": f"Set additive category to {label}",
                "feature_changed": additive_col,
                "change_tested": "category set to 1",
                "baseline_mean_prediction": round(baseline_mean, 4),
                "intervention_mean_prediction": round(intervention_mean, 4),
                "estimated_delta_g_ch4_kg_dmi": round(delta, 4),
                "interpretation": (
                    "Potential methane reduction signal"
                    if delta < 0
                    else "No reduction signal in trained model"
                ),
            }
        )

    if not scenarios:
        return pd.DataFrame(
            columns=[
                "scenario",
                "feature_changed",
                "change_tested",
                "baseline_mean_prediction",
                "intervention_mean_prediction",
                "estimated_delta_g_ch4_kg_dmi",
                "interpretation",
            ]
        )

    opportunity_df = pd.DataFrame(scenarios)
    opportunity_df = opportunity_df.sort_values(
        "estimated_delta_g_ch4_kg_dmi"
    ).reset_index(drop=True)

    opportunity_df["rank"] = np.arange(1, len(opportunity_df) + 1)

    return opportunity_df


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def evaluate_model_on_full_data(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate fitted model on full cleaned data for explanation context.

    This is not a substitute for test-set performance reporting.
    """

    preds = model.predict(X)

    return {
        "MAE": round(mean_absolute_error(y, preds), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y, preds)), 4),
        "R2": round(r2_score(y, preds), 4),
    }


def build_explanation_report(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    metrics: Dict[str, float],
    importance_df: pd.DataFrame,
    opportunity_df: pd.DataFrame,
) -> str:
    """
    Build a plain-text explanation report.
    """

    top_features = importance_df.head(10)[
        ["rank", "feature", "importance", "method"]
    ]

    if opportunity_df.empty:
        top_opportunities = "No mitigation-opportunity scenarios were generated."
    else:
        top_opportunities = opportunity_df.head(10).to_string(index=False)

    lines = [
        "Enteric Methane AI - Explainability Report",
        "=" * 52,
        "",
        "Model context",
        "-------------",
        f"Model: {model_name}",
        f"Rows explained: {len(X)}",
        f"Features explained: {X.shape[1]}",
        "",
        "Full-data fit metrics",
        "---------------------",
        "These metrics are reported only for explanation context. "
        "Use train_models.py outputs for proper test-set and cross-validation results.",
        f"MAE: {metrics['MAE']}",
        f"RMSE: {metrics['RMSE']}",
        f"R2: {metrics['R2']}",
        "",
        "Top global feature drivers",
        "--------------------------",
        top_features.to_string(index=False),
        "",
        "Top mitigation-opportunity sensitivity scenarios",
        "------------------------------------------------",
        top_opportunities,
        "",
        "Responsible interpretation notes",
        "-------------------------------",
        "- Feature importance shows association learned by the model, not causal proof.",
        "- Sensitivity scenarios are controlled model probes, not validated ration advice.",
        "- Additive scenarios depend on the quality and coverage of the training data.",
        "- Any mitigation recommendation must consider animal performance, health, cost, "
        "feed availability, and scientific evidence.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run the explainability workflow.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = load_model_bundle()
    model = bundle["model"]
    model_name = bundle.get("model_name", type(model).__name__)
    feature_cols = bundle["feature_cols"]

    df = load_clean_data()
    X, y = prepare_explanation_data(df, feature_cols)

    print(f"Loaded model: {model_name}")
    print(f"Explaining {len(X)} rows and {X.shape[1]} features")
    print(f"SHAP available: {HAS_SHAP}")

    importance_df = calculate_feature_importance(model, X, y)
    local_df = calculate_local_explanation(model, X, row_index=0)
    opportunity_df = calculate_mitigation_opportunities(model, X)
    metrics = evaluate_model_on_full_data(model, X, y)

    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    local_df.to_csv(LOCAL_EXPLANATION_PATH, index=False)
    opportunity_df.to_csv(MITIGATION_OPPORTUNITY_PATH, index=False)

    report = build_explanation_report(
        model_name=str(model_name),
        X=X,
        y=y,
        metrics=metrics,
        importance_df=importance_df,
        opportunity_df=opportunity_df,
    )

    EXPLANATION_REPORT_PATH.write_text(report, encoding="utf-8")

    print("\nTop global feature drivers")
    print("--------------------------")
    print(importance_df.head(10).to_string(index=False))

    print("\nTop mitigation-opportunity scenarios")
    print("------------------------------------")
    if opportunity_df.empty:
        print("No mitigation-opportunity scenarios were generated.")
    else:
        print(opportunity_df.head(10).to_string(index=False))

    print(f"\nSaved feature importance to: {FEATURE_IMPORTANCE_PATH}")
    print(f"Saved local explanation to: {LOCAL_EXPLANATION_PATH}")
    print(f"Saved mitigation opportunities to: {MITIGATION_OPPORTUNITY_PATH}")
    print(f"Saved explanation report to: {EXPLANATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
