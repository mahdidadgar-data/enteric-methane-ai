"""
Clean and feature-engineer the enteric methane dataset.

Works with either:
  - the real dataset (place at data/raw/mitigation_dataset.xlsx or .csv, and
    update RAW_PATH below), or
  - the synthetic placeholder produced by generate_synthetic_data.py

Output: data/processed/clean_dataset.csv, ready for train_models.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent.parent
RAW_DIR = BASE / "data" / "raw"
PROCESSED_DIR = BASE / "data" / "processed"

# Point this at the real dataset once you have it, e.g.:
# RAW_PATH = RAW_DIR / "mitigation_dataset.xlsx"
RAW_PATH = RAW_DIR / "synthetic_mitigation_dataset.csv"

NUMERIC_COLS = [
    "dmi_kg", "cp_g_kg", "ndf_g_kg", "adf_g_kg", "ee_g_kg",
    "forage_pct", "body_weight_kg", "milk_yield_kg",
    "additive_dose", "ch4_g_kg_dmi", "ch4_g_day",
]
CATEGORICAL_COLS = ["breed", "additive_type"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No data found at {path}.\n"
            f"Run `python src/generate_synthetic_data.py` first, or download the "
            f"real dataset and update RAW_PATH in data_prep.py (see data/README.md)."
        )
    if path.suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # keep only columns we know how to work with (real dataset has ~162 cols;
    # this keeps the pipeline focused on the core nutrition/CH4 variables)
    keep_cols = [c for c in NUMERIC_COLS + CATEGORICAL_COLS + ["study_id"] if c in df.columns]
    df = df[keep_cols]

    # drop rows with no target at all
    df = df.dropna(subset=["ch4_g_kg_dmi"], how="all")

    # median-impute missing numeric predictors, flag which rows were imputed
    for col in NUMERIC_COLS:
        if col in df.columns and col not in ("ch4_g_kg_dmi", "ch4_g_day"):
            was_missing = df[col].isna()
            if was_missing.any():
                df[f"{col}_was_missing"] = was_missing.astype(int)
                df[col] = df[col].fillna(df[col].median())

    # fill categorical NAs with "unknown" rather than dropping
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # basic sanity filtering (physiologically implausible values are almost
    # certainly data entry errors, not real biology)
    if "ch4_g_kg_dmi" in df.columns:
        df = df[df["ch4_g_kg_dmi"].between(5, 40)]
    if "dmi_kg" in df.columns:
        df = df[df["dmi_kg"].between(3, 35)]

    # simple engineered feature: concentrate proportion
    if "forage_pct" in df.columns:
        df["concentrate_pct"] = 100 - df["forage_pct"]

    # one-hot encode categoricals for modeling
    df = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns], drop_first=False)

    return df.reset_index(drop=True)


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    print(f"Loaded {len(raw)} raw records from {RAW_PATH.name}")
    cleaned = clean(raw)
    out_path = PROCESSED_DIR / "clean_dataset.csv"
    cleaned.to_csv(out_path, index=False)
    print(f"Wrote {len(cleaned)} cleaned records ({cleaned.shape[1]} columns) to {out_path}")
