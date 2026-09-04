"""Checks for reproducibility, provenance, units and consistent model inputs."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_demo_outputs import (  # noqa: E402
    MODEL_SHA256, build_outputs, encode_profiles, load_saved_bundle,
    reference_profiles, scenario_profiles,
)


@pytest.fixture(scope="module")
def generated():
    return build_outputs()


def test_reference_grid_and_scenarios_preserve_feature_relationships():
    columns = load_saved_bundle()["feature_cols"]
    profiles = reference_profiles()
    assert len(profiles) == 27
    for _, rows in [("baseline", profiles)] + scenario_profiles(profiles):
        frame = encode_profiles(rows, columns)
        np.testing.assert_allclose(frame["forage_pct"] + frame["concentrate_pct"], 100)
        np.testing.assert_allclose(frame["milk_per_kg_dmi"], frame["milk_yield_kg"] / frame["dmi_kg"])
        assert (frame["adf_g_kg"] <= frame["ndf_g_kg"]).all()
        assert (frame.filter(like="breed_").sum(axis=1) == 1).all()
        assert (frame.filter(like="additive_type_").sum(axis=1) == 1).all()
        assert np.isfinite(frame.to_numpy()).all()
        assert list(frame.columns) == columns


def test_feature_importance_is_normalized_and_complete(generated):
    importance, _, _ = generated
    assert len(importance) == 28 and importance["feature"].is_unique
    assert (importance["importance"] >= 0).all()
    assert importance["importance"].is_monotonic_decreasing
    assert importance["importance"].sum() == pytest.approx(1.0)
    assert importance["relative_importance_pct"].sum() == pytest.approx(100.0)
    assert importance["method"].str.contains("not SHAP").all()


def test_sensitivity_delta_definition_and_provenance(generated):
    _, sensitivity, metadata = generated
    assert len(sensitivity) == 10
    assert sensitivity["scenario"].is_unique
    assert sensitivity["baseline_mean_prediction"].nunique() == 1
    np.testing.assert_allclose(
        sensitivity["estimated_delta_g_ch4_kg_dmi"],
        sensitivity["intervention_mean_prediction"] - sensitivity["baseline_mean_prediction"],
    )
    assert (sensitivity["reference_profiles"] == 27).all()
    assert sensitivity["reference_kind"].str.contains("Synthetic").all()
    assert metadata["model_sha256"] == MODEL_SHA256


def test_committed_outputs_match_generator(generated):
    importance, sensitivity, metadata = generated
    directory = ROOT / "outputs" / "explainability"
    for name, expected in [
        ("feature_importance.csv", importance),
        ("mitigation_opportunities.csv", sensitivity),
    ]:
        actual = pd.read_csv(directory / name)
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-8, atol=1e-8)
    assert json.loads((directory / "demo_metadata.json").read_text()) == metadata


def test_evidence_briefs_have_expected_source_links_and_scope():
    sources = {
        "3_nop": {"26229078", "33131815"},
        "asparagopsis": {"33730064", "33516546"},
        "nitrate": {"21787938", "27236758"},
    }
    for slug, expected in sources.items():
        text = (ROOT / "outputs" / "rag" / f"evidence_brief_{slug}.md").read_text()
        assert set(re.findall(r"https://pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/", text)) == expected
        assert "4 September 2026" in text
        assert "not a systematic review" in text
        assert "not been validated" in text
