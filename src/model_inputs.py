"""Shared input encoding for the live app and saved-model demo outputs."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


def build_input_row(
    feature_cols: List[str],
    inputs: Dict[str, float | int | str],
) -> pd.DataFrame:
    """
    Build a one-row dataframe matching the exact feature columns used by
    the trained model.
    """

    forage_pct = float(inputs["forage_pct"])
    dmi_kg = float(inputs["dmi_kg"])
    milk_yield_kg = float(inputs["milk_yield_kg"])

    row: Dict[str, float | int] = {
        "dmi_kg": dmi_kg,
        "cp_g_kg": float(inputs["cp_g_kg"]),
        "ndf_g_kg": float(inputs["ndf_g_kg"]),
        "adf_g_kg": float(inputs["adf_g_kg"]),
        "ee_g_kg": float(inputs["ee_g_kg"]),
        "starch_g_kg": float(inputs["starch_g_kg"]),
        "forage_pct": forage_pct,
        "concentrate_pct": 100 - forage_pct,
        "body_weight_kg": float(inputs["body_weight_kg"]),
        "milk_yield_kg": milk_yield_kg,
        "milk_per_kg_dmi": milk_yield_kg / dmi_kg if dmi_kg > 0 else 0,
        "lactating": int(inputs["lactating"]),
        "additive_dose": float(inputs["additive_dose"]),
    }

    breed = str(inputs["breed"])
    additive_type = str(inputs["additive_type"])

    for col in feature_cols:
        if col.startswith("breed_"):
            row[col] = 1 if col == f"breed_{breed}" else 0

        elif col.startswith("additive_type_"):
            row[col] = 1 if col == f"additive_type_{additive_type}" else 0

        elif col.endswith("_was_missing"):
            row[col] = 0

        elif col not in row:
            row[col] = 0

    return pd.DataFrame([row])[feature_cols]

