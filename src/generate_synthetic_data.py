"""
Generate a synthetic placeholder dataset with the same schema as the real
'global dataset of enteric methane mitigation experiments' (dairy cows,
1963-2022, 797 records / 162 variables — see data/README.md).

This is NOT real experimental data. It exists only so the rest of the
pipeline (data_prep -> train_models -> explain -> app) can be built and
tested end-to-end before the real dataset is downloaded.

Value ranges are drawn from the published literature ranges documented in
data/README.md (e.g. CP 42-276 g/kg DM, NDF 93-776 g/kg DM).
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N = 800  # roughly matches the real dataset's 797 records

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "synthetic_mitigation_dataset.csv"

ADDITIVES = ["none", "3-NOP", "seaweed", "tannin", "nitrate", "lipid", "other"]
ADDITIVE_EFFECT = {  # approximate directional CH4 effect, purely illustrative
    "none": 0.0,
    "3-NOP": -0.30,
    "seaweed": -0.45,
    "tannin": -0.12,
    "nitrate": -0.15,
    "lipid": -0.10,
    "other": -0.05,
}
BREEDS = ["Holstein", "Jersey", "Crossbred", "Other"]


def generate() -> pd.DataFrame:
    n = N
    cp = RNG.uniform(42, 276, n)
    ndf = RNG.uniform(93, 776, n)
    adf = ndf * RNG.uniform(0.45, 0.65, n)
    ee = RNG.uniform(20, 60, n)
    forage_pct = RNG.uniform(0, 100, n)
    body_weight = RNG.normal(575, 60, n).clip(400, 750)
    dmi = (0.025 * body_weight + RNG.normal(0, 1.5, n)).clip(8, 28)
    milk_yield = RNG.uniform(0, 45, n) * RNG.choice([0, 1], size=n, p=[0.15, 0.85])
    additive_type = RNG.choice(ADDITIVES, size=n, p=[0.35, 0.12, 0.12, 0.12, 0.12, 0.12, 0.05])
    additive_dose = np.where(
        additive_type == "none", 0.0, RNG.uniform(0.1, 2.5, n)
    )
    breed = RNG.choice(BREEDS, size=n, p=[0.55, 0.15, 0.2, 0.1])

    # crude synthetic CH4 generating process, loosely inspired by known
    # directional relationships (higher NDF/forage -> more CH4 per kg DMI;
    # additives reduce it; more concentrate/starch -> less CH4 per kg DMI)
    base_yield = 20 + 0.01 * (ndf - 400) - 0.015 * (100 - forage_pct)
    additive_effect = np.array([ADDITIVE_EFFECT[a] for a in additive_type]) * additive_dose
    ch4_g_kg_dmi = (base_yield + additive_effect + RNG.normal(0, 1.2, n)).clip(10, 32)
    ch4_g_day = ch4_g_kg_dmi * dmi

    df = pd.DataFrame({
        "study_id": [f"S{100+i}" for i in range(n)],
        "breed": breed,
        "dmi_kg": dmi.round(2),
        "cp_g_kg": cp.round(1),
        "ndf_g_kg": ndf.round(1),
        "adf_g_kg": adf.round(1),
        "ee_g_kg": ee.round(1),
        "forage_pct": forage_pct.round(1),
        "body_weight_kg": body_weight.round(1),
        "milk_yield_kg": milk_yield.round(1),
        "additive_type": additive_type,
        "additive_dose": additive_dose.round(2),
        "ch4_g_kg_dmi": ch4_g_kg_dmi.round(2),
        "ch4_g_day": ch4_g_day.round(1),
    })

    # inject realistic missingness (real literature datasets are messy)
    for col, frac in [("adf_g_kg", 0.15), ("ee_g_kg", 0.25), ("milk_yield_kg", 0.05)]:
        mask = RNG.random(n) < frac
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} synthetic records to {OUT_PATH}")
    print(df.head())
