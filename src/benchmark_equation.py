"""
Empirical benchmark equation for enteric methane prediction.

This module implements a transparent IPCC Tier-2-style benchmark that can be
used as a scientific baseline for the machine learning models.

The benchmark is intentionally simple. It estimates methane production from:

    - dry matter intake (DMI, kg/day)
    - diet gross energy concentration (MJ/kg DM)
    - methane conversion factor (Ym; fraction of gross energy converted to CH4)

Conceptual calculation:
    GEI (MJ/day)        = DMI (kg/day) * GE concentration (MJ/kg DM)
    CH4 energy (MJ/day) = GEI * Ym
    CH4 mass (kg/day)   = CH4 energy / 55.65
    CH4 mass (g/day)    = CH4 mass * 1000

where:
    55.65 MJ/kg CH4 is the energy content of methane.

In this project, Ym is adjusted with forage percentage to create a simple,
interpretable benchmark:
    - lower-forage diets use a lower Ym
    - higher-forage diets use a higher Ym

This benchmark should not be presented as a full mechanistic methane model.
Its role is to provide a transparent comparator for the ML pipeline.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


ArrayLike = Union[float, int, pd.Series, np.ndarray]

MJ_PER_KG_CH4 = 55.65
DEFAULT_GE_MJ_PER_KG_DM = 18.4

# Approximate Ym range used for the project benchmark.
# Values are expressed as fractions, not percentages.
LOW_FORAGE_YM = 0.055
HIGH_FORAGE_YM = 0.075


def _to_series(values: ArrayLike, name: str) -> pd.Series:
    """
    Convert scalar, NumPy array, or pandas Series input to a pandas Series.
    """

    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")

    if isinstance(values, np.ndarray):
        return pd.Series(values, name=name, dtype="float64")

    return pd.Series([values], name=name, dtype="float64")


def methane_conversion_factor(forage_pct: ArrayLike) -> pd.Series:
    """
    Estimate methane conversion factor, Ym, from forage percentage.

    Parameters
    ----------
    forage_pct:
        Forage proportion of the diet, expressed as percentage of diet DM
        from 0 to 100.

    Returns
    -------
    pandas.Series
        Methane conversion factor as a fraction of gross energy intake.
        Values are clipped between LOW_FORAGE_YM and HIGH_FORAGE_YM.
    """

    forage = _to_series(forage_pct, name="forage_pct")
    forage = forage.clip(lower=0, upper=100)

    ym = LOW_FORAGE_YM + (
        (HIGH_FORAGE_YM - LOW_FORAGE_YM) * (forage / 100)
    )

    return ym


def predict_ch4_g_day(
    dmi_kg: ArrayLike,
    forage_pct: ArrayLike,
    ge_mj_per_kg: float = DEFAULT_GE_MJ_PER_KG_DM,
) -> pd.Series:
    """
    Predict methane production in grams per day.

    Parameters
    ----------
    dmi_kg:
        Dry matter intake in kg/day.
    forage_pct:
        Forage proportion of the diet, expressed as percentage of diet DM.
    ge_mj_per_kg:
        Gross energy concentration of the diet in MJ/kg DM.

    Returns
    -------
    pandas.Series
        Predicted methane production in g/day.
    """

    dmi = _to_series(dmi_kg, name="dmi_kg")
    forage = _to_series(forage_pct, name="forage_pct")

    if len(dmi) != len(forage):
        if len(dmi) == 1:
            dmi = pd.Series(np.repeat(dmi.iloc[0], len(forage)), name="dmi_kg")
        elif len(forage) == 1:
            forage = pd.Series(np.repeat(forage.iloc[0], len(dmi)), name="forage_pct")
        else:
            raise ValueError(
                "dmi_kg and forage_pct must have the same length, "
                "or one of them must be a scalar."
            )

    if ge_mj_per_kg <= 0:
        raise ValueError("ge_mj_per_kg must be greater than zero.")

    dmi = dmi.clip(lower=0)
    gei_mj_day = dmi * ge_mj_per_kg

    ym = methane_conversion_factor(forage)
    ch4_mj_day = gei_mj_day * ym
    ch4_g_day = (ch4_mj_day / MJ_PER_KG_CH4) * 1000

    return ch4_g_day


def predict_ch4_g_kg_dmi(
    dmi_kg: ArrayLike,
    forage_pct: ArrayLike,
    ge_mj_per_kg: float = DEFAULT_GE_MJ_PER_KG_DM,
) -> pd.Series:
    """
    Predict methane yield in grams per kg dry matter intake.

    Parameters
    ----------
    dmi_kg:
        Dry matter intake in kg/day.
    forage_pct:
        Forage proportion of the diet, expressed as percentage of diet DM.
    ge_mj_per_kg:
        Gross energy concentration of the diet in MJ/kg DM.

    Returns
    -------
    pandas.Series
        Predicted methane yield in g CH4/kg DMI.
    """

    dmi = _to_series(dmi_kg, name="dmi_kg").clip(lower=0)
    ch4_g_day = predict_ch4_g_day(
        dmi_kg=dmi,
        forage_pct=forage_pct,
        ge_mj_per_kg=ge_mj_per_kg,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        ch4_g_kg_dmi = ch4_g_day / dmi.replace(0, np.nan)

    return ch4_g_kg_dmi


def add_benchmark_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add benchmark methane predictions to a modelling dataframe.

    Required input columns:
        - dmi_kg
        - forage_pct

    Added columns:
        - ch4_g_day_benchmark
        - ch4_g_kg_dmi_benchmark
    """

    required_cols = ["dmi_kg", "forage_pct"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Cannot calculate benchmark predictions. Missing columns: {missing_cols}"
        )

    result = df.copy()

    result["ch4_g_day_benchmark"] = predict_ch4_g_day(
        dmi_kg=result["dmi_kg"],
        forage_pct=result["forage_pct"],
    )

    result["ch4_g_kg_dmi_benchmark"] = predict_ch4_g_kg_dmi(
        dmi_kg=result["dmi_kg"],
        forage_pct=result["forage_pct"],
    )

    return result


def main() -> None:
    """
    Run a small sanity-check demo.
    """

    demo = pd.DataFrame(
        {
            "dmi_kg": [18, 22, 25],
            "forage_pct": [40, 60, 80],
        }
    )

    demo = add_benchmark_predictions(demo)

    print("Benchmark equation sanity check")
    print("--------------------------------")
    print(demo.round(2))


if __name__ == "__main__":
    main()
