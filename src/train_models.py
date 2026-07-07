"""
Train and evaluate ML models for enteric methane prediction, and compare
them against the published empirical equation in benchmark_equation.py.

Usage:
    python src/train_models.py
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from benchmark_equation import predict_ch4_g_kg_dmi

BASE = Path(__file__).resolve().parent.parent
PROCESSED_PATH = BASE / "data" / "processed" / "clean_dataset.csv"
MODEL_DIR = BASE / "data" / "processed" / "models"

TARGET = "ch4_g_kg_dmi"
DROP_COLS = ["study_id", "ch4_g_day", TARGET]


def load_data() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_PATH} not found. Run `python src/data_prep.py` first."
        )
    return pd.read_csv(PROCESSED_PATH)


def evaluate(y_true, y_pred, label: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {label:22s}  MAE={mae:6.3f}  RMSE={rmse:6.3f}  R2={r2:6.3f}")
    return {"model": label, "MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    df = load_data()
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42
    )

    results = []

    # --- Benchmark: published empirical / IPCC-style equation ---
    if "dmi_kg" in df.columns and "forage_pct" in df.columns:
        forage_col = "forage_pct" if "forage_pct" in df.columns else None
        bench_pred = predict_ch4_g_kg_dmi(
            df.loc[idx_test, "dmi_kg"], df.loc[idx_test, forage_col]
        )
        results.append(evaluate(y_test, bench_pred, "Empirical equation"))

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results.append(evaluate(y_test, rf.predict(X_test), "Random Forest"))

    # --- Gradient Boosting (XGBoost if available, else sklearn GBM) ---
    if HAS_XGB:
        gbm = XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        gbm_label = "XGBoost"
    else:
        gbm = GradientBoostingRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05, random_state=42
        )
        gbm_label = "GradientBoosting (sklearn)"
    gbm.fit(X_train, y_train)
    results.append(evaluate(y_test, gbm.predict(X_test), gbm_label))

    # --- Save best model (by RMSE, excluding the benchmark equation) ---
    model_results = [r for r in results if r["model"] != "Empirical equation"]
    best = min(model_results, key=lambda r: r["RMSE"])
    best_model = rf if best["model"] == "Random Forest" else gbm

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "feature_cols": feature_cols}, MODEL_DIR / "best_model.joblib")
    pd.DataFrame(results).to_csv(MODEL_DIR / "model_comparison.csv", index=False)

    print(f"\nBest model: {best['model']}  (saved to {MODEL_DIR / 'best_model.joblib'})")
    print(f"Comparison table saved to {MODEL_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
