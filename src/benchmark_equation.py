"""
Published empirical / IPCC Tier 2 style equation for enteric methane, used as
the benchmark that your ML models need to beat (or at least understand where
they beat it and where they don't -- this comparison is the strongest
scientific-contribution angle of the whole project).

IPCC (2006) Tier 2 approach:
    CH4 (MJ/day) = GEI (MJ/day) * Ym / 55.65
    CH4 (g/day)  = CH4 (MJ/day) / 0.05565   [55.65 MJ per kg CH4 -> 1 g CH4 = 0.05565 MJ]

Where:
    GEI = gross energy intake = DMI (kg/day) * GE content of diet (MJ/kg DM)
    Ym  = methane conversion factor, i.e. % of GEI converted to CH4
          (IPCC default ~6.5% for dairy cattle on mixed rations; varies 5.5-7.5%
           depending on diet quality/forage proportion in refined versions)

This is intentionally simple -- it's the equation regulators and national GHG
inventories actually use, which is exactly why it's the right baseline to beat.
"""

import numpy as np
import pandas as pd

MJ_PER_KG_CH4 = 55.65
DEFAULT_GE_MJ_PER_KG_DM = 18.4  # typical mixed ruminant diet, MJ/kg DM


def methane_conversion_factor(forage_pct: pd.Series) -> pd.Series:
    """
    Simple linear adjustment of Ym by forage proportion, reflecting the
    well-documented finding that higher-forage diets produce more CH4 per
    unit of energy than higher-concentrate/starch diets.
    Ranges roughly 5.5% (low forage) to 7.5% (high forage), centered on the
    IPCC default of 6.5%.
    """
    return 0.055 + 0.002 * (forage_pct / 100.0) * 10  # -> 0.055 to 0.075


def predict_ch4_g_day(dmi_kg: pd.Series, forage_pct: pd.Series,
                       ge_mj_per_kg: float = DEFAULT_GE_MJ_PER_KG_DM) -> pd.Series:
    gei_mj_day = dmi_kg * ge_mj_per_kg
    ym = methane_conversion_factor(forage_pct)
    ch4_mj_day = gei_mj_day * ym
    ch4_g_day = ch4_mj_day / MJ_PER_KG_CH4 * 1000
    return ch4_g_day


def predict_ch4_g_kg_dmi(dmi_kg: pd.Series, forage_pct: pd.Series,
                          ge_mj_per_kg: float = DEFAULT_GE_MJ_PER_KG_DM) -> pd.Series:
    return predict_ch4_g_day(dmi_kg, forage_pct, ge_mj_per_kg) / dmi_kg


if __name__ == "__main__":
    # quick sanity check against typical literature values (dairy cows,
    # ~18-25 g CH4/kg DMI is the commonly reported range)
    demo = pd.DataFrame({
        "dmi_kg": [18, 22, 25],
        "forage_pct": [40, 60, 80],
    })
    demo["ch4_g_kg_dmi_benchmark"] = predict_ch4_g_kg_dmi(demo.dmi_kg, demo.forage_pct)
    demo["ch4_g_day_benchmark"] = predict_ch4_g_day(demo.dmi_kg, demo.forage_pct)
    print(demo)
