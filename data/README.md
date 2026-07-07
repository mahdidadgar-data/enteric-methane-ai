# Data dictionary & provenance

## Real dataset (recommended)

**Source:** "A global dataset of enteric methane mitigation experiments with lactating and
non-lactating dairy cows conducted from 1963 to 2022" — a data article published alongside
a review by Arndt et al. and later authors, hosted on ScienceDirect / Mendeley Data.

- 797 records (rows), 162 variables (columns)
- Compiled from 213 peer-reviewed publications
- Includes: publication info, experimental design, animal description, methane measurement
  method, diet nutrient composition (NDF, ADF, CP, EE, DM), feed additive type/dose, milk
  yield & components, nitrogen excretion, rumen fermentation parameters, CH4/CO2/H2 emissions

Download it, place the raw file at `data/raw/mitigation_dataset.xlsx`, and point
`src/data_prep.py` at it (there's a `RAW_PATH` constant at the top of the file).

## Core columns this pipeline expects (minimum viable subset)

| Column           | Meaning                                  | Typical range (from literature) |
|------------------|-------------------------------------------|----------------------------------|
| `dmi_kg`         | Dry matter intake (kg/day)                | 10 – 25 kg/day (dairy cows)      |
| `cp_g_kg`        | Crude protein (g/kg DM)                   | 42 – 276 g/kg DM                 |
| `ndf_g_kg`       | Neutral detergent fiber (g/kg DM)         | 93 – 776 g/kg DM                 |
| `adf_g_kg`       | Acid detergent fiber (g/kg DM)            | ~150 – 450 g/kg DM               |
| `ee_g_kg`        | Ether extract / fat (g/kg DM)             | 20 – 60 g/kg DM                  |
| `forage_pct`     | Forage proportion of diet (%)             | 0 – 100%                         |
| `body_weight_kg` | Animal body weight (kg)                   | 400 – 700 kg (dairy cows)        |
| `milk_yield_kg`  | Milk yield (kg/day), 0 if non-lactating   | 0 – 45 kg/day                    |
| `additive_type`  | Feed additive category (categorical)      | none, 3-NOP, seaweed, tannin, nitrate, lipid, other |
| `additive_dose`  | Additive dose (g/day or % DM, normalized) | varies                            |
| `breed`          | Breed (categorical)                       | Holstein, Jersey, crossbred, etc.|
| `ch4_g_day`      | **Target**: methane output (g/day)        | 200 – 600 g/day (dairy cows)     |
| `ch4_g_kg_dmi`   | Alternative target: CH4 yield (g/kg DMI)  | 15 – 25 g/kg DMI                 |

Not every study reports every column — expect substantial missingness; `data_prep.py`
handles this explicitly rather than silently dropping rows.

## Synthetic placeholder data

`src/generate_synthetic_data.py` generates a CSV with this exact schema, using realistic
distributions so you can build, test, and demo the whole pipeline before the real dataset
is in hand. It is clearly **not** real experimental data — do not use it to draw scientific
conclusions, only to validate that the code runs correctly.
