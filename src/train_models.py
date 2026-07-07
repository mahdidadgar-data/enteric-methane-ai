"""
Train and evaluate machine learning models for enteric methane prediction.

This script compares machine learning models against two transparent baselines:

    1. Naive mean baseline
    2. Empirical/IPCC-style methane benchmark from benchmark_equation.py

The modelling target is:
    ch4_g_kg_dmi

Inputs:
    data/processed/clean_dataset.csv

Outputs:
    models/best_model.joblib
    outputs/modeling/model_comparison.csv
    outputs/modeling/cross_validation_results.csv
    outputs/modeling/test_predictions.csv
    outputs/modeling/feature_importance.csv
    outputs/modeling/training_report.txt

Run:
    python src/train_models.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from benchmark_equation import predict_ch4_g_kg_dmi


# ---------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_PATH = BASE_DIR / "data" / "processed" / "clean_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs" / "modeling"

BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"
MODEL_COMPARISON_PATH = OUTPUT_DIR / "model_comparison.csv"
CV_RESULTS_PATH = OUTPUT_DIR / "cross_validation_results.csv"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
TRAINING_REPORT_PATH = OUTPUT_DIR / "training_report.txt"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "ch4_g_kg_dmi"

# Columns excluded from model predictors.
# ch4_g_day is excluded because it is mathematically derived from the target
# and DMI, so keeping it would create target leakage.
DROP_COLS = [
    "study_id",
    "ch4_g_day",
    "ch4_g_day_benchmark",
    "ch4_g_kg_dmi_benchmark",
    TARGET,
]


# ---------------------------------------------------------------------
# Loading and feature preparation
# ---------------------------------------------------------------------

def load_data(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """
    Load the cleaned modelling dataset.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found at: {path}\n"
            "Run `python src/generate_synthetic_data.py` and then "
            "`python src/data_prep.py` first."
        )

    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Prepare feature matrix X and target vector y.

    Returns
    -------
    X:
        Numeric feature matrix.
    y:
        Target vector.
    feature_cols:
        Names of numeric model features.
    excluded_non_numeric:
        Columns excluded because they are non-numeric.
    """

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found in the dataset. "
            f"Available columns: {list(df.columns)}"
        )

    candidate_cols = [col for col in df.columns if col not in DROP_COLS]

    excluded_non_numeric = [
        col for col in candidate_cols if not pd.api.types.is_numeric_dtype(df[col])
    ]

    feature_cols = [
        col for col in candidate_cols if pd.api.types.is_numeric_dtype(df[col])
    ]

    if not feature_cols:
        raise ValueError("No numeric feature columns available for model training.")

    X = df[feature_cols].copy()
    y = pd.to_numeric(df[TARGET], errors="coerce")

    valid_target_mask = y.notna()
    X = X.loc[valid_target_mask].reset_index(drop=True)
    y = y.loc[valid_target_mask].reset_index(drop=True)

    return X, y, feature_cols, excluded_non_numeric


# ---------------------------------------------------------------------
# Models and metrics
# ---------------------------------------------------------------------

def build_models() -> Dict[str, object]:
    """
    Define candidate regression models.
    """

    models: Dict[str, object] = {
        "Naive mean baseline": DummyRegressor(strategy="mean"),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        models["Gradient Boosting"] = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        )

    return models


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str,
) -> Dict[str, float | str]:
    """
    Calculate regression metrics.
    """

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }


def run_cross_validation(
    model_name: str,
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, float | str]:
    """
    Run 5-fold cross-validation for a candidate model.
    """

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=None,
        error_score="raise",
    )

    return {
        "model": model_name,
        "CV_MAE_mean": round(-scores["test_mae"].mean(), 4),
        "CV_MAE_std": round(scores["test_mae"].std(), 4),
        "CV_RMSE_mean": round(-scores["test_rmse"].mean(), 4),
        "CV_RMSE_std": round(scores["test_rmse"].std(), 4),
        "CV_R2_mean": round(scores["test_r2"].mean(), 4),
        "CV_R2_std": round(scores["test_r2"].std(), 4),
    }


def calculate_empirical_benchmark(
    original_df: pd.DataFrame,
    test_indices: np.ndarray,
) -> pd.Series | None:
    """
    Calculate empirical benchmark predictions for the test set.

    Returns None if required columns are unavailable.
    """

    required_cols = ["dmi_kg", "forage_pct"]
    missing_cols = [col for col in required_cols if col not in original_df.columns]

    if missing_cols:
        print(
            "Skipping empirical benchmark because required columns are missing: "
            f"{missing_cols}"
        )
        return None

    benchmark_pred = predict_ch4_g_kg_dmi(
        dmi_kg=original_df.loc[test_indices, "dmi_kg"],
        forage_pct=original_df.loc[test_indices, "forage_pct"],
    )

    return benchmark_pred


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------

def save_feature_importance(best_model: object, feature_cols: List[str]) -> None:
    """
    Save feature importance for tree-based models if available.
    """

    if not hasattr(best_model, "feature_importances_"):
        return

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": best_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)


def build_training_report(
    df: pd.DataFrame,
    feature_cols: List[str],
    excluded_non_numeric: List[str],
    comparison_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    best_model_name: str,
) -> str:
    """
    Build a plain-text training report.
    """

    lines = [
        "Enteric Methane AI - Model Training Report",
        "=" * 52,
        "",
        "Dataset",
        "-------",
        f"Rows: {len(df)}",
        f"Columns: {df.shape[1]}",
        f"Target: {TARGET}",
        f"Test size: {TEST_SIZE}",
        f"Random state: {RANDOM_STATE}",
        "",
        "Feature set",
        "-----------",
        f"Number of model features: {len(feature_cols)}",
    ]

    if excluded_non_numeric:
        lines.append(
            "Excluded non-numeric columns: " + ", ".join(excluded_non_numeric)
        )
    else:
        lines.append("Excluded non-numeric columns: none")

    lines.extend(
        [
            "",
            "Test-set model comparison",
            "-------------------------",
            comparison_df.to_string(index=False),
            "",
            "Cross-validation results",
            "------------------------",
            cv_df.to_string(index=False),
            "",
            "Selected model",
            "--------------",
            best_model_name,
            "",
            "Notes",
            "-----",
            "- ch4_g_day is excluded from predictors to avoid target leakage.",
            "- The empirical equation is used as a transparent scientific benchmark.",
            "- Synthetic data is for software testing and portfolio demonstration only.",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run the full modelling workflow.
    """

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X, y, feature_cols, excluded_non_numeric = prepare_features(df)

    print(f"Loaded cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Training with {len(feature_cols)} numeric features")
    print(f"Target: {TARGET}\n")

    indices = np.arange(len(X))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    models = build_models()
    test_results: List[Dict[str, float | str]] = []
    cv_results: List[Dict[str, float | str]] = []
    fitted_models: Dict[str, object] = {}

    # Empirical benchmark on the test set only.
    benchmark_pred = calculate_empirical_benchmark(df, idx_test)
    test_predictions = pd.DataFrame(
        {
            "row_index": idx_test,
            "y_true": y_test.to_numpy(),
        }
    )

    if benchmark_pred is not None:
        benchmark_pred = benchmark_pred.reset_index(drop=True)
        test_results.append(
            evaluate_predictions(
                y_true=y_test,
                y_pred=benchmark_pred,
                model_name="Empirical equation",
            )
        )
        test_predictions["Empirical equation"] = benchmark_pred.to_numpy()

    # ML models.
    for model_name, model in models.items():
        print(f"Training: {model_name}")

        cv_result = run_cross_validation(model_name, model, X_train, y_train)
        cv_results.append(cv_result)

        model.fit(X_train, y_train)
        fitted_models[model_name] = model

        y_pred = model.predict(X_test)
        test_results.append(
            evaluate_predictions(
                y_true=y_test,
                y_pred=y_pred,
                model_name=model_name,
            )
        )
        test_predictions[model_name] = y_pred

    comparison_df = pd.DataFrame(test_results).sort_values("RMSE")
    cv_df = pd.DataFrame(cv_results).sort_values("CV_RMSE_mean")

    # Select the best trained ML model by test RMSE.
    ml_comparison_df = comparison_df[
        ~comparison_df["model"].isin(["Empirical equation", "Naive mean baseline"])
    ]

    best_model_name = ml_comparison_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    model_bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_cols": feature_cols,
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }

    joblib.dump(model_bundle, BEST_MODEL_PATH)

    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    cv_df.to_csv(CV_RESULTS_PATH, index=False)
    test_predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)
    save_feature_importance(best_model, feature_cols)

    training_report = build_training_report(
        df=df,
        feature_cols=feature_cols,
        excluded_non_numeric=excluded_non_numeric,
        comparison_df=comparison_df,
        cv_df=cv_df,
        best_model_name=str(best_model_name),
    )

    TRAINING_REPORT_PATH.write_text(training_report, encoding="utf-8")

    print("\nTest-set model comparison")
    print("-------------------------")
    print(comparison_df.to_string(index=False))

    print("\nCross-validation results")
    print("------------------------")
    print(cv_df.to_string(index=False))

    print(f"\nBest ML model: {best_model_name}")
    print(f"Saved best model to: {BEST_MODEL_PATH}")
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
