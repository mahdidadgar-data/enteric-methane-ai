"""
Exploratory data analysis for the enteric methane AI project.

This script provides a lightweight but professional EDA layer for the cleaned
methane modelling dataset.

It is written as a plain Python file with VS Code / PyCharm cell markers
(`# %%`) so it can be run either as:

    python notebooks/01_eda.py

or explored interactively as notebook-style cells.

Required previous steps:
    python src/generate_synthetic_data.py
    python src/data_prep.py

Input:
    data/processed/clean_dataset.csv

Outputs:
    outputs/eda/summary_statistics.csv
    outputs/eda/missing_values.csv
    outputs/eda/correlation_with_target.csv
    outputs/eda/additive_group_summary.csv
    outputs/eda/ch4_distribution.png
    outputs/eda/ch4_vs_ndf.png
    outputs/eda/ch4_vs_forage.png
    outputs/eda/ch4_by_additive_type.png
    outputs/eda/top_correlations.png

Important:
    If the project is run with synthetic data, EDA findings are useful only for
    validating the pipeline and demonstrating workflow. They should not be
    interpreted as scientific conclusions.
"""

# %%

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "clean_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "eda"

TARGET = "ch4_g_kg_dmi"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def load_clean_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the cleaned modelling dataset.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found at: {path}\n"
            "Run `python src/generate_synthetic_data.py` and "
            "`python src/data_prep.py` first."
        )

    return pd.read_csv(path)


def save_missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Save missing-value counts and percentages.
    """

    missing = df.isna().sum()
    missing_pct = (missing / len(df)) * 100

    report = (
        pd.DataFrame(
            {
                "missing_count": missing,
                "missing_pct": missing_pct.round(2),
            }
        )
        .query("missing_count > 0")
        .sort_values("missing_count", ascending=False)
    )

    report.to_csv(OUTPUT_DIR / "missing_values.csv")

    return report


def save_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Save descriptive statistics for numeric columns.
    """

    summary = df.describe().T
    summary.to_csv(OUTPUT_DIR / "summary_statistics.csv")

    return summary


def save_target_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Save numeric correlations with the methane target variable.
    """

    numeric_df = df.select_dtypes(include="number")

    if TARGET not in numeric_df.columns:
        raise ValueError(f"Target column '{TARGET}' is not numeric or not present.")

    correlations = (
        numeric_df.corr(numeric_only=True)[TARGET]
        .drop(labels=[TARGET])
        .sort_values(key=lambda series: series.abs(), ascending=False)
        .reset_index()
    )

    correlations.columns = ["feature", "correlation_with_target"]
    correlations.to_csv(OUTPUT_DIR / "correlation_with_target.csv", index=False)

    return correlations


def create_additive_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize methane yield by one-hot encoded additive type columns.
    """

    additive_cols = [col for col in df.columns if col.startswith("additive_type_")]

    if not additive_cols:
        empty = pd.DataFrame(
            columns=[
                "additive_type",
                "records",
                "mean_ch4_g_kg_dmi",
                "median_ch4_g_kg_dmi",
                "std_ch4_g_kg_dmi",
            ]
        )
        empty.to_csv(OUTPUT_DIR / "additive_group_summary.csv", index=False)
        return empty

    rows = []

    for col in additive_cols:
        subset = df.loc[df[col] == 1, TARGET]

        if subset.empty:
            continue

        rows.append(
            {
                "additive_type": col.replace("additive_type_", ""),
                "records": int(subset.shape[0]),
                "mean_ch4_g_kg_dmi": round(subset.mean(), 3),
                "median_ch4_g_kg_dmi": round(subset.median(), 3),
                "std_ch4_g_kg_dmi": round(subset.std(), 3),
            }
        )

    summary = pd.DataFrame(rows).sort_values("mean_ch4_g_kg_dmi")
    summary.to_csv(OUTPUT_DIR / "additive_group_summary.csv", index=False)

    return summary


def save_target_distribution_plot(df: pd.DataFrame) -> None:
    """
    Save histogram of the methane target variable.
    """

    plt.figure(figsize=(7, 4))
    df[TARGET].hist(bins=30)
    plt.xlabel("Methane yield (g CH4/kg DMI)")
    plt.ylabel("Number of records")
    plt.title("Distribution of methane yield")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ch4_distribution.png", dpi=150)
    plt.close()


def save_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    output_name: str,
    title: str,
) -> None:
    """
    Save a scatter plot against the methane target.
    """

    if x_col not in df.columns:
        return

    plt.figure(figsize=(7, 4))
    plt.scatter(df[x_col], df[TARGET], alpha=0.45, s=18)
    plt.xlabel(x_label)
    plt.ylabel("Methane yield (g CH4/kg DMI)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=150)
    plt.close()


def save_additive_plot(additive_summary: pd.DataFrame) -> None:
    """
    Save bar chart of mean methane yield by additive category.
    """

    if additive_summary.empty:
        return

    plot_df = additive_summary.sort_values("mean_ch4_g_kg_dmi")

    plt.figure(figsize=(8, 4))
    plt.bar(plot_df["additive_type"], plot_df["mean_ch4_g_kg_dmi"])
    plt.xlabel("Additive type")
    plt.ylabel("Mean methane yield (g CH4/kg DMI)")
    plt.title("Mean methane yield by additive category")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ch4_by_additive_type.png", dpi=150)
    plt.close()


def save_top_correlation_plot(correlations: pd.DataFrame, top_n: int = 10) -> None:
    """
    Save horizontal bar chart of strongest absolute correlations.
    """

    if correlations.empty:
        return

    plot_df = correlations.head(top_n).copy()
    plot_df = plot_df.sort_values("correlation_with_target")

    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["feature"], plot_df["correlation_with_target"])
    plt.xlabel("Correlation with methane yield")
    plt.ylabel("Feature")
    plt.title("Top numeric correlations with methane yield")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_correlations.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Main EDA workflow
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run exploratory analysis and save outputs.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean_data()

    print("Enteric Methane AI - EDA")
    print("=" * 40)
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Target: {TARGET}")
    print("")

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset.")

    missing_report = save_missing_value_report(df)
    summary_stats = save_summary_statistics(df)
    correlations = save_target_correlations(df)
    additive_summary = create_additive_group_summary(df)

    save_target_distribution_plot(df)
    save_scatter_plot(
        df=df,
        x_col="ndf_g_kg",
        x_label="NDF (g/kg DM)",
        output_name="ch4_vs_ndf.png",
        title="Methane yield vs dietary fiber (NDF)",
    )
    save_scatter_plot(
        df=df,
        x_col="forage_pct",
        x_label="Forage proportion (% of diet DM)",
        output_name="ch4_vs_forage.png",
        title="Methane yield vs forage proportion",
    )
    save_additive_plot(additive_summary)
    save_top_correlation_plot(correlations)

    print("Target summary")
    print("--------------")
    print(df[TARGET].describe().round(3).to_string())
    print("")

    print("Missing values")
    print("--------------")
    if missing_report.empty:
        print("No missing values remain in the cleaned dataset.")
    else:
        print(missing_report.to_string())
    print("")

    print("Top correlations with methane yield")
    print("-----------------------------------")
    print(correlations.head(10).to_string(index=False))
    print("")

    print("Mean methane yield by additive category")
    print("---------------------------------------")
    if additive_summary.empty:
        print("No additive one-hot columns found.")
    else:
        print(additive_summary.to_string(index=False))
    print("")

    print(f"EDA outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
