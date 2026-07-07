# Data Dictionary & Provenance

This folder documents the data layer for the `enteric-methane-ai` project.

The project is designed to work with either:

1. a real external enteric methane mitigation dataset, or  
2. a synthetic placeholder dataset generated inside this repository for testing and demonstration.

The raw real dataset is **not included** in this repository.

## Folder Structure

```text
data/
├── raw/
│   ├── .gitkeep
│   └── README.md              # optional notes for raw data placement
├── processed/
│   ├── .gitkeep
│   └── README.md              # optional notes for generated clean data
└── README.md                  # this file
```

Recommended `.gitignore` behavior:

```gitignore
data/raw/*
!data/raw/README.md
!data/raw/.gitkeep

data/processed/*
!data/processed/README.md
!data/processed/.gitkeep
```

This keeps the folder structure visible on GitHub while preventing raw data and generated outputs from being committed accidentally.

## Real Dataset

The project is structured around the following published dataset:

> **A global dataset of enteric methane mitigation experiments with lactating and non-lactating dairy cows conducted from 1963 to 2022**

The dataset is associated with a scientific data article and is available through external publisher/data-repository sources such as ScienceDirect or Mendeley Data.

Reported dataset characteristics:

- **797 records**
- **162 variables**
- Compiled from **213 peer-reviewed publications**
- Includes lactating and non-lactating dairy cow methane mitigation experiments
- Covers diet composition, animal traits, methane measurement, additives, production variables, and experimental metadata

### Why the Raw Dataset Is Not Included

The raw dataset is not redistributed in this repository because it should be obtained directly from the original publisher or data repository. This ensures that users access the correct version and follow the source provider's usage terms.

After downloading the dataset, place it in:

```text
data/raw/
```

Then update `RAW_PATH` in:

```text
src/data_prep.py
```

For example:

```python
RAW_PATH = RAW_DIR / "mitigation_dataset.xlsx"
```

or:

```python
RAW_PATH = RAW_DIR / "mitigation_dataset.csv"
```

## Synthetic Placeholder Dataset

To make the repository runnable without redistributing external data, this project includes a synthetic data generator:

```text
src/generate_synthetic_data.py
```

It creates:

```text
data/raw/synthetic_mitigation_dataset.csv
```

The synthetic dataset uses realistic column names and plausible biological ranges, but it is **not real experimental data**.

### Intended Use of Synthetic Data

Synthetic data is used only for:

- testing the pipeline
- demonstrating project structure
- validating that scripts run end-to-end
- supporting a reproducible portfolio demo

It must **not** be used to draw scientific conclusions about methane mitigation.

## Core Columns Used by the Pipeline

The current modelling pipeline focuses on a minimum viable subset of diet, animal, additive, and methane variables.

| Column | Type | Meaning | Typical / Allowed Range |
|---|---:|---|---|
| `study_id` | ID | Study or record identifier | text |
| `breed` | categorical | Breed group | Holstein, Jersey, Crossbred, Other |
| `lactating` | numeric/binary | Lactation status | 1 = lactating, 0 = dry/non-lactating |
| `dmi_kg` | numeric | Dry matter intake | 3–35 kg/day in cleaning filter |
| `cp_g_kg` | numeric | Crude protein | 20–350 g/kg DM in cleaning filter |
| `ndf_g_kg` | numeric | Neutral detergent fiber | 50–850 g/kg DM in cleaning filter |
| `adf_g_kg` | numeric | Acid detergent fiber | 20–600 g/kg DM in cleaning filter |
| `ee_g_kg` | numeric | Ether extract / dietary fat | 0–120 g/kg DM in cleaning filter |
| `starch_g_kg` | numeric | Starch concentration | 0–550 g/kg DM in cleaning filter |
| `forage_pct` | numeric | Forage proportion of diet | 0–100% |
| `body_weight_kg` | numeric | Animal body weight | 250–900 kg in cleaning filter |
| `milk_yield_kg` | numeric | Milk yield | 0–70 kg/day in cleaning filter |
| `additive_type` | categorical | Mitigation additive category | none, 3-NOP, seaweed, tannin, nitrate, lipid, other |
| `additive_dose` | numeric | Normalized additive dose | 0–20 in cleaning filter |
| `ch4_g_kg_dmi` | numeric | Main modelling target: methane yield | 5–45 g CH4/kg DMI in cleaning filter |
| `ch4_g_day` | numeric | Methane production per day | 20–1000 g/day in cleaning filter |

## Engineered Features

`src/data_prep.py` creates additional modelling features:

| Feature | Meaning |
|---|---|
| `concentrate_pct` | `100 - forage_pct` |
| `milk_per_kg_dmi` | milk yield divided by dry matter intake |
| `*_was_missing` | missingness indicator columns for imputed numeric predictors |
| one-hot encoded `breed_*` columns | encoded breed categories |
| one-hot encoded `additive_type_*` columns | encoded additive categories |

## Main Target Variable

The main prediction target is:

```text
ch4_g_kg_dmi
```

This target expresses methane yield per kg dry matter intake and allows comparisons across animals with different intake levels.

The variable:

```text
ch4_g_day
```

is not used as a model predictor because it is mathematically related to methane yield and dry matter intake. Including it as a predictor would create target leakage.

## Data Preparation Workflow

Run the following steps from the project root:

```bash
python src/generate_synthetic_data.py
python src/data_prep.py
```

The first script generates synthetic raw data. The second script cleans and prepares the model-ready dataset.

Generated outputs:

```text
data/processed/clean_dataset.csv
data/processed/data_quality_report.txt
```

These generated files are normally excluded from GitHub by `.gitignore`.

## Missing Values

The data preparation pipeline handles missingness explicitly:

- numeric predictors are median-imputed
- missingness indicator columns are created
- categorical missing values are filled as `unknown`
- target values are not imputed

This is important because animal nutrition datasets compiled from literature often have incomplete reporting across studies.

## Sanity Filtering

The cleaning script applies conservative sanity filters to remove clearly implausible values. These filters are not strict biological laws; they are practical safeguards against placeholder values, parsing errors, or obvious data-entry issues.

Examples:

- `forage_pct` must be between 0 and 100
- `dmi_kg` must be within a plausible animal intake range
- `ch4_g_kg_dmi` must be within a broad methane-yield range

A data-quality report is generated to document missingness and filtering decisions.

## Responsible Use

This data layer supports a research and portfolio prototype.

Important limitations:

- Synthetic data is not scientific evidence.
- Real dataset results depend on study coverage, measurement method, diet type, animal category, and experimental design.
- Methane mitigation recommendations require full scientific review and expert interpretation.
- The project should report uncertainty, limitations, and insufficient evidence clearly.

## Reproducibility Notes

For reproducible demo runs, use the synthetic dataset generated by:

```bash
python src/generate_synthetic_data.py
```

For scientific extension, replace the synthetic file with the real external dataset and update the cleaning logic as needed to match the exact column names and units in the downloaded data.
