"""
Streamlit front end for the Scientific AI Assistant for Sustainable Ruminant
Nutrition: enter a diet/animal profile, get a CH4 prediction, see which
factors drove it, and (optionally, if ANTHROPIC_API_KEY is set) pull current
literature evidence on the top mitigation lever.

Run with:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from benchmark_equation import predict_ch4_g_kg_dmi  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "data" / "processed" / "models" / "best_model.joblib"

st.set_page_config(page_title="Ruminant Methane AI Assistant", layout="centered")
st.title("🐄 Enteric Methane Prediction & Evidence Assistant")
st.caption(
    "Research / portfolio tool — not a validated on-farm decision system. "
    "Predictions are only as reliable as the training data's coverage of your scenario."
)

if not MODEL_PATH.exists():
    st.error(
        "No trained model found. Run `python src/generate_synthetic_data.py`, "
        "`python src/data_prep.py`, and `python src/train_models.py` first."
    )
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_cols = bundle["feature_cols"]

st.header("1. Diet & animal profile")
col1, col2 = st.columns(2)
with col1:
    dmi = st.number_input("Dry matter intake (kg/day)", 3.0, 35.0, 20.0, 0.5)
    cp = st.number_input("Crude protein (g/kg DM)", 42.0, 276.0, 160.0, 1.0)
    ndf = st.number_input("NDF (g/kg DM)", 93.0, 776.0, 400.0, 1.0)
    adf = st.number_input("ADF (g/kg DM)", 50.0, 500.0, 220.0, 1.0)
with col2:
    ee = st.number_input("Ether extract / fat (g/kg DM)", 10.0, 80.0, 35.0, 1.0)
    forage_pct = st.slider("Forage % of diet", 0, 100, 55)
    body_weight = st.number_input("Body weight (kg)", 300.0, 800.0, 575.0, 5.0)
    milk_yield = st.number_input("Milk yield (kg/day, 0 if dry/non-lactating)", 0.0, 55.0, 28.0, 1.0)

additive_type = st.selectbox(
    "Feed additive", ["none", "3-NOP", "seaweed", "tannin", "nitrate", "lipid", "other"]
)
additive_dose = st.number_input("Additive dose (normalized units)", 0.0, 3.0, 0.0, 0.1)
breed = st.selectbox("Breed", ["Holstein", "Jersey", "Crossbred", "Other"])

if st.button("Predict CH4 output", type="primary"):
    row = {
        "dmi_kg": dmi, "cp_g_kg": cp, "ndf_g_kg": ndf, "adf_g_kg": adf,
        "ee_g_kg": ee, "forage_pct": forage_pct, "concentrate_pct": 100 - forage_pct,
        "body_weight_kg": body_weight, "milk_yield_kg": milk_yield,
        "additive_dose": additive_dose,
    }
    for col in feature_cols:
        if col.startswith("breed_"):
            row[col] = 1 if col == f"breed_{breed}" else 0
        elif col.startswith("additive_type_"):
            row[col] = 1 if col == f"additive_type_{additive_type}" else 0
        elif col.endswith("_was_missing"):
            row[col] = 0
        elif col not in row:
            row[col] = 0

    X = pd.DataFrame([row])[feature_cols]
    pred_ml = model.predict(X)[0]
    pred_benchmark = predict_ch4_g_kg_dmi(pd.Series([dmi]), pd.Series([forage_pct])).iloc[0]

    st.header("2. Prediction")
    c1, c2 = st.columns(2)
    c1.metric("ML model prediction", f"{pred_ml:.1f} g CH4 / kg DMI")
    c2.metric("Empirical equation (IPCC-style)", f"{pred_benchmark:.1f} g CH4 / kg DMI")
    st.metric("Estimated total CH4 output", f"{pred_ml * dmi:.0f} g/day (ML model)")

    st.header("3. What's driving this")
    importance_path = BASE / "data" / "processed" / "models" / "feature_importance.csv"
    if importance_path.exists():
        imp = pd.read_csv(importance_path).head(8)
        st.bar_chart(imp.set_index("feature")["importance"])
        st.caption("Higher = more influence on the model's predictions overall (not specific to this exact input).")
    else:
        st.info("Run `python src/explain.py` to populate this section.")

    st.header("4. Current literature evidence (optional)")
    st.caption(
        "Run `python src/rag_pubmed.py --topic \"...\"` from the command line "
        "(needs internet + ANTHROPIC_API_KEY) to pull an evidence summary for "
        "whichever additive or strategy you're considering, with citations."
    )
