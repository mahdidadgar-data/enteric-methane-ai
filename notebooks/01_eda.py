"""
Exploratory data analysis for the cleaned methane dataset.

Run as a plain script (`python notebooks/01_eda.py`) or paste cells into a
Jupyter notebook -- kept as a .py file so it runs anywhere without a Jupyter
dependency, and cells are marked with `# %%` for editors (VS Code, PyCharm,
Spyder) that render them as notebook-style cells.
"""

# %%
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE / "data" / "processed" / "clean_dataset.csv")
print(df.shape)
df.describe()

# %% Distribution of the target variable
plt.figure(figsize=(6, 4))
df["ch4_g_kg_dmi"].hist(bins=30)
plt.xlabel("CH4 (g/kg DMI)")
plt.ylabel("Count")
plt.title("Distribution of methane yield")
plt.tight_layout()
plt.savefig(BASE / "notebooks" / "ch4_distribution.png", dpi=120)
plt.close()

# %% CH4 vs NDF (fiber) -- expect a positive relationship based on rumen fermentation biology
plt.figure(figsize=(6, 4))
plt.scatter(df["ndf_g_kg"], df["ch4_g_kg_dmi"], alpha=0.4, s=12)
plt.xlabel("NDF (g/kg DM)")
plt.ylabel("CH4 (g/kg DMI)")
plt.title("Methane yield vs dietary fiber (NDF)")
plt.tight_layout()
plt.savefig(BASE / "notebooks" / "ch4_vs_ndf.png", dpi=120)
plt.close()

# %% CH4 by additive type -- boxplot-style comparison
additive_cols = [c for c in df.columns if c.startswith("additive_type_")]
if additive_cols:
    means = {c.replace("additive_type_", ""): df.loc[df[c] == 1, "ch4_g_kg_dmi"].mean()
              for c in additive_cols}
    print("Mean CH4 (g/kg DMI) by additive type:")
    for name, val in sorted(means.items(), key=lambda kv: kv[1]):
        print(f"  {name:10s} {val:.2f}")

print("\nEDA complete. Plots saved to notebooks/*.png")
