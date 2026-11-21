"""Check the published three-fit uncertainty against the archived measurements."""

import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
review = runpy.run_path(str(ROOT / "scripts/review_checkpoint_d.py"))


def test_three_fit_inference_handles_ties_and_direction():
    infer = review["three_fit_inference"]
    balanced = infer([-1.0, 0.0, 1.0])
    assert balanced["paired_t_p"] == pytest.approx(1)
    assert balanced["paired_mean_ci95"] == pytest.approx([-2.48413771, 2.48413771])
    assert balanced["sign_test_nonzero_pairs"] == 2
    assert balanced["sign_test_p"] == 1
    constant = infer([2.0, 2.0, 2.0])
    assert constant["paired_mean_ci95"] is None and constant["paired_t_p"] is None
    assert constant["sign_test_p"] == 0.25
    assert infer([0.0, 0.0, 0.0])["sign_test_p"] is None


def test_checkpoint_d_published_spread_and_paired_comparisons():
    archive = json.loads((ROOT / "experiments/checkpoint_d.json").read_text(encoding="utf-8"))
    results = archive["visit_networks"]
    summary = review["summarize_fits"](results["held_out"])
    for row, saved in zip(summary, results["paired_fits"], strict=True):
        assert (row["model"], row["selection_rule"], row["fits"]) == (
            saved["model"], saved["selection_rule"], saved["fits"])
        for name, metric in row["metrics"].items():
            published = saved["metrics"][name]
            assert published.keys() == metric.keys()
            for field, value in metric.items():
                assert published[field] == pytest.approx(value)
    tests = [metric for row in summary for metric in row["metrics"].values() if "paired_t_p" in metric]
    assert len(tests) == 60
    value = next(row for row in summary if row["model"] == "value_mlp" and row["selection_rule"] == "vnm_terminal")
    wins = value["metrics"]["win"]
    assert wins["sample_sd"] == pytest.approx(0.0267, abs=0.00005)
    assert wins["paired_mean_ci95"] == pytest.approx([0.03715024, 0.05296839])
    assert wins["paired_t_p"] == pytest.approx(0.00166007354)
    assert wins["holm_p"] == pytest.approx(0.09794433913)
    assert min(metric["holm_p"] for metric in tests) == pytest.approx(0.03784200081)
