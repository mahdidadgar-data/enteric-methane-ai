"""
Generate a synthetic placeholder dataset for the enteric methane AI pipeline.

This script creates synthetic data with a schema inspired by published enteric
methane mitigation datasets in dairy cattle. It is NOT real experimental data.

Purpose:
    - Allow the full project pipeline to run end-to-end before the real dataset
      is downloaded.
    - Preserve realistic column names and biological value ranges.
    - Simulate plausible directional relationships between diet composition,
      animal traits, mitigation additives, and methane output.

Output:
    data/raw/synthetic_mitigation_dataset.csv

Important:
    Synthetic data is only for software testing, demonstration, and portfolio
    reproducibility. It should not be interpreted as scientific evidence.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_SEED = 42
N_RECORDS = 800

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BASE_DIR / "data" / "raw" / "synthetic_mitigation_dataset.csv"

RNG = np.random.default_rng(RANDOM_SEED)

ADDITIVES = [
    "none",
    "3-NOP",
    "seaweed",
    "tannin",
    "nitrate",
    "lipid",
    "other",
]

# Approximate synthetic reduction factors.
# These values create learnable signal in the synthetic data and are not
# intended to represent exact biological efficacy.
ADDITIVE_REDUCTION = {
    "none": 0.00,
    "3-NOP": 0.22,
    "seaweed": 0.30,
    "tannin": 0.08,
    "nitrate": 0.10,
    "lipid": 0.07,
    "other": 0.04,
}

BREEDS = [
    "Holstein",
    "Jersey",
    "Crossbred",
    "Other",
]


# ---------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------

def generate_synthetic_data(n_records: int = N_RECORDS) -> pd.DataFrame:
    """
    Generate a synthetic enteric methane mitigation dataset.

    Parameters
    ----------
    n_records:
        Number of synthetic records to generate.

    Returns
    -------
    pandas.DataFrame
        Synthetic dataset containing diet, animal, additive, and methane variables.
    """

    n = n_records

    # Diet composition variables
    cp_g_kg = RNG.uniform(42, 276, n)
    ndf_g_kg = RNG.uniform(93, 776, n)
    adf_g_kg = ndf_g_kg * RNG.uniform(0.45, 0.65, n)
    ee_g_kg = RNG.uniform(20, 60, n)
    starch_g_kg = RNG.uniform(40, 420, n)
    forage_pct = RNG.uniform(20, 90, n)

    # Animal traits
    body_weight_kg = RNG.normal(575, 60, n).clip(400, 750)
    dmi_kg = (0.025 * body_weight_kg + RNG.normal(0, 1.5, n)).clip(8, 28)

    lactating = RNG.choice([0, 1], size=n, p=[0.15, 0.85])
    milk_yield_kg = np.where(
        lactating == 1,
        RNG.uniform(8, 45, n),
        0.0,
    )

    # Additives and breed
    additive_type = RNG.choice(
        ADDITIVES,
        size=n,
        p=[0.35, 0.12, 0.12, 0.12, 0.12, 0.12, 0.05],
    )

    additive_dose = np.where(
        additive_type == "none",
        0.0,
        RNG.uniform(0.1, 2.5, n),
    )

    breed = RNG.choice(
        BREEDS,
        size=n,
        p=[0.55, 0.15, 0.20, 0.10],
    )

    # -----------------------------------------------------------------
    # Synthetic methane-generation process
    # -----------------------------------------------------------------
    # Directional assumptions:
    #   - Higher NDF and forage proportion increase methane yield.
    #   - Higher starch and ether extract reduce methane yield.
    #   - Higher DMI increases methane per day.
    #   - Mitigation additives reduce methane yield.
    #
    # This is intentionally simplified and should not be used as a
    # biological model.
    # -----------------------------------------------------------------

    baseline_ch4_yield = (
        20
        + 0.012 * (ndf_g_kg - 400)
        + 0.030 * (forage_pct - 50)
        - 0.006 * (starch_g_kg - 200)
        - 0.040 * (ee_g_kg - 35)
    )

    reduction_fraction = np.array(
        [ADDITIVE_REDUCTION[item] for item in additive_type]
    )

    # Dose scaling keeps stronger additives visible but prevents unrealistic collapse.
    dose_scaler = np.clip(additive_dose / 2.5, 0, 1)
    effective_reduction = reduction_fraction * dose_scaler

    ch4_g_kg_dmi = baseline_ch4_yield * (1 - effective_reduction)
    ch4_g_kg_dmi = ch4_g_kg_dmi + RNG.normal(0, 1.4, n)
    ch4_g_kg_dmi = ch4_g_kg_dmi.clip(8, 35)

    ch4_g_day = ch4_g_kg_dmi * dmi_kg

    df = pd.DataFrame(
        {
            "study_id": [f"S{100 + i}" for i in range(n)],
            "breed": breed,
            "lactating": lactating,
            "dmi_kg": dmi_kg.round(2),
            "cp_g_kg": cp_g_kg.round(1),
            "ndf_g_kg": ndf_g_kg.round(1),
            "adf_g_kg": adf_g_kg.round(1),
            "ee_g_kg": ee_g_kg.round(1),
            "starch_g_kg": starch_g_kg.round(1),
            "forage_pct": forage_pct.round(1),
            "body_weight_kg": body_weight_kg.round(1),
            "milk_yield_kg": milk_yield_kg.round(1),
            "additive_type": additive_type,
            "additive_dose": additive_dose.round(2),
            "ch4_g_kg_dmi": ch4_g_kg_dmi.round(2),
            "ch4_g_day": ch4_g_day.round(1),
        }
    )

    df = inject_missingness(df)

    return df


def inject_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject realistic missingness into selected columns.

    Literature-style animal nutrition datasets often have incomplete values
    for some diet fractions or production variables.
    """

    df = df.copy()
    missingness_plan = {
        "adf_g_kg": 0.15,
        "ee_g_kg": 0.25,
        "starch_g_kg": 0.20,
        "milk_yield_kg": 0.05,
    }

    for col, frac in missingness_plan.items():
        mask = RNG.random(len(df)) < frac
        df.loc[mask, col] = np.nan

    return df


def main() -> None:
    """Generate and save the synthetic dataset."""

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_data()
    df.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(df)} synthetic records to: {OUT_PATH}")
    print(f"Columns: {df.shape[1]}")
    print("\nPreview:")
    print(df.head())

    print("\nMissing values:")
    print(df.isna().sum()[df.isna().sum() > 0])


if __name__ == "__main__":
    main()