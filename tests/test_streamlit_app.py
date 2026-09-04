"""Runtime smoke tests; these do not measure scientific model validity."""

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
APP_PATH = ROOT / "app" / "app.py"


@pytest.fixture(scope="module")
def bundle():
    # Load only the known, repository-owned demonstration artifact.
    assert hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() == (
        "c3b3080de60f0f0e587385c2d275936ae604c143280ec53ebe5787ed4956e355"
    )
    return joblib.load(MODEL_PATH)


def launch():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    assert not app.error
    return app


def input_widget(app, label):
    return next(widget for widget in app.number_input if widget.label == label)


def test_saved_model_schema(bundle):
    assert bundle["model_name"] == "XGBoost"
    assert bundle["target"] == "ch4_g_kg_dmi"
    assert len(bundle["feature_cols"]) == 28
    assert len(set(bundle["feature_cols"])) == 28
    assert list(bundle["model"].feature_names_in_) == bundle["feature_cols"]
    assert not {"ch4_g_kg_dmi", "ch4_g_day"}.intersection(bundle["feature_cols"])


def test_startup_with_saved_demo_outputs():
    app = launch()
    assert [tab.label for tab in app.tabs] == [
        "Prediction", "Explainability", "Evidence briefs", "About"
    ]
    assert any(item.value == "Trained model loaded" for item in app.success)


def test_default_prediction_and_repeatability():
    app = launch()
    app.button[0].click().run()
    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    prediction = float(metrics["ML prediction"].split()[0])
    assert np.isfinite(prediction) and prediction > 0
    assert "Empirical benchmark" in metrics
    assert "Estimated total CH₄" in metrics
    app.button[0].click().run()
    assert not app.exception
    assert {item.label: item.value for item in app.metric} == metrics


@pytest.mark.parametrize("dmi,forage", [(3.0, 0), (35.0, 100)])
def test_ui_boundary_inputs_do_not_crash(dmi, forage):
    app = launch()
    input_widget(app, "Dry matter intake (kg/day)").set_value(dmi)
    app.slider[0].set_value(forage)
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    prediction = next(item.value for item in app.metric if item.label == "ML prediction")
    assert np.isfinite(float(prediction.split()[0]))
    assert any("edge" in item.value for item in app.warning)


def test_inconsistent_additive_input_displays_warning():
    app = launch()
    input_widget(app, "Additive dose (normalized units)").set_value(1.0).run()
    assert not app.exception
    assert any("inconsistent" in item.value for item in app.warning)


def test_explainability_tables_are_populated_and_labelled():
    app = launch()
    tables = [table.value for table in app.dataframe]
    assert any(len(frame) == 28 and "feature" in frame for frame in tables)
    assert any(len(frame) == 10 and "scenario" in frame for frame in tables)
    assert any("not SHAP" in caption.value for caption in app.caption)
    assert any("27 synthetic profiles" in item.value for item in app.info)
    assert not any("not found" in item.value.lower() for item in app.info)


def test_all_three_evidence_briefs_render():
    app = launch()
    selector = next(item for item in app.selectbox if item.label == "Select saved evidence brief")
    assert len(selector.options) == 3
    files = sorted((ROOT / "outputs" / "rag").glob("evidence_brief_*.md"))
    for path in files:
        selector = next(item for item in app.selectbox if item.label == "Select saved evidence brief")
        selector.set_value(path).run()
        assert not app.exception
        assert any(path.read_text(encoding="utf-8").strip() == text.value.strip() for text in app.markdown)


def test_missing_optional_outputs_still_allow_prediction(monkeypatch):
    original_exists = Path.exists
    absent = {
        ROOT / "outputs" / "explainability" / "feature_importance.csv",
        ROOT / "outputs" / "explainability" / "mitigation_opportunities.csv",
        ROOT / "outputs" / "rag",
    }
    monkeypatch.setattr(Path, "exists", lambda path: False if path in absent else original_exists(path))
    app = launch()
    app.button[0].click().run()
    assert not app.exception
    assert any(item.label == "ML prediction" for item in app.metric)
