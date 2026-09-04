"""Build saved-model demo outputs without training data or retraining.

Run: python src/build_demo_outputs.py
The reference grid is synthetic, not a sample of real farms or training rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost

from model_inputs import build_input_row

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
MODEL_SHA256 = "c3b3080de60f0f0e587385c2d275936ae604c143280ec53ebe5787ed4956e355"
OUTPUT_DIR = ROOT / "outputs" / "explainability"

BASE_PROFILE = {
    "dmi_kg": 20.0, "cp_g_kg": 160.0, "ndf_g_kg": 400.0,
    "adf_g_kg": 220.0, "ee_g_kg": 35.0, "starch_g_kg": 220.0,
    "forage_pct": 55.0, "body_weight_kg": 575.0,
    "milk_yield_kg": 28.0, "lactating": 1, "breed": "Holstein",
    "additive_type": "none", "additive_dose": 0.0,
}
GRID = {
    "dmi_kg": [16.0, 20.0, 24.0],
    "forage_pct": [40.0, 55.0, 70.0],
    "milk_yield_kg": [20.0, 28.0, 36.0],
}


def load_saved_bundle():
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("Saved demo model changed; review provenance before regenerating outputs.")
    bundle = joblib.load(MODEL_PATH)
    if list(bundle["model"].feature_names_in_) != bundle["feature_cols"]:
        raise ValueError("Model and bundle feature order differ.")
    return bundle


def reference_profiles():
    return [
        dict(BASE_PROFILE, **dict(zip(GRID, values)))
        for values in product(*GRID.values())
    ]


def encode_profiles(profiles, feature_cols):
    return pd.concat(
        [build_input_row(feature_cols, profile) for profile in profiles],
        ignore_index=True,
    )


def feature_importance(model, feature_cols):
    # Average split gain, normalized across all features (unused features = 0).
    scores = model.get_booster().get_score(importance_type="gain")
    gain = np.array([scores.get(name, 0.0) for name in feature_cols], dtype=float)
    if not np.isfinite(gain).all() or (gain < 0).any() or gain.sum() <= 0:
        raise ValueError("Invalid built-in feature importance.")
    result = pd.DataFrame({"feature": feature_cols, "importance": gain / gain.sum()})
    result = result.sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    result["relative_importance_pct"] = result["importance"] * 100.0
    result["method"] = "XGBoost normalized split gain (not SHAP)"
    return result[["rank", "feature", "importance", "relative_importance_pct", "method"]]


def scenario_profiles(profiles):
    scenarios = []
    for label, field, delta in [
        ("Forage -10 percentage points; concentrate +10", "forage_pct", -10.0),
        ("Dietary fat +10 g/kg DM", "ee_g_kg", 10.0),
        ("NDF -50 g/kg DM", "ndf_g_kg", -50.0),
        ("Starch +50 g/kg DM", "starch_g_kg", 50.0),
    ]:
        changed = [dict(profile, **{field: profile[field] + delta}) for profile in profiles]
        scenarios.append((label, changed))
    for additive in ["3-NOP", "seaweed", "tannin", "nitrate", "lipid", "other"]:
        changed = [dict(profile, additive_type=additive, additive_dose=1.0) for profile in profiles]
        scenarios.append((f"{additive}: category change; normalized dose 1", changed))
    return scenarios


def mitigation_sensitivity(model, feature_cols):
    profiles = reference_profiles()
    baseline = float(np.mean(model.predict(encode_profiles(profiles, feature_cols))))
    rows = []
    for label, changed in scenario_profiles(profiles):
        scenario_mean = float(np.mean(model.predict(encode_profiles(changed, feature_cols))))
        delta = scenario_mean - baseline
        if not np.isfinite([baseline, scenario_mean, delta]).all():
            raise ValueError("Non-finite scenario prediction.")
        rows.append({
            "scenario": label,
            "baseline_mean_prediction": baseline,
            "intervention_mean_prediction": scenario_mean,
            "estimated_delta_g_ch4_kg_dmi": delta,
            "reference_profiles": len(profiles),
            "reference_kind": "Synthetic demonstration grid; not real farm data",
            "interpretation": "Lower model prediction" if delta < 0 else "No lower model prediction",
        })
    result = pd.DataFrame(rows).sort_values(
        ["estimated_delta_g_ch4_kg_dmi", "scenario"]
    ).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def build_outputs():
    bundle = load_saved_bundle()
    model, columns = bundle["model"], bundle["feature_cols"]
    importance = feature_importance(model, columns)
    sensitivity = mitigation_sensitivity(model, columns)
    metadata = {
        "schema_version": 1,
        "model_sha256": MODEL_SHA256,
        "xgboost_version": xgboost.__version__,
        "feature_importance_method": "Normalized average split gain; not SHAP or causal effect",
        "sensitivity_reference": "Synthetic demonstration grid, not training rows or observed farms",
        "reference_profiles": len(reference_profiles()),
        "base_profile": BASE_PROFILE,
        "grid": GRID,
        "delta_definition": "scenario mean minus baseline mean, g CH4/kg DMI",
        "rank_definition": "Sort order of model deltas; not an efficacy ranking",
        "limitations": [
            "Reference profiles and input probes are illustrative, not fully balanced rations.",
            "Concentrate and milk-per-DMI features are recomputed with the same encoder as the app.",
            "Normalized additive dose 1 has no common physical dose across additives.",
            "No outcome labels or scientific performance metrics are generated.",
            "Training-data provenance is not embedded in the saved model.",
            "Outputs are precomputed and do not change with the prediction form.",
        ],
    }
    return importance, sensitivity, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    importance, sensitivity, metadata = build_outputs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(args.output_dir / "feature_importance.csv", index=False, float_format="%.10g")
    sensitivity.to_csv(args.output_dir / "mitigation_opportunities.csv", index=False, float_format="%.10g")
    (args.output_dir / "demo_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(importance)} feature rows and {len(sensitivity)} scenarios for {metadata['reference_profiles']} synthetic profiles.")


if __name__ == "__main__":
    main()
