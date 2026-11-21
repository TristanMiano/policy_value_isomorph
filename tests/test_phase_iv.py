"""Check the independent-block estimand, structural controls and exact matchup calculations."""

import gzip
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments" / "phase_iv"


def load_rows(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        return [json.loads(line) for line in stream]


def test_primary_comparisons_are_paired_on_weights_and_use_32_independent_units():
    primary = json.loads((DATA / "primary.json").read_text())
    cells = {(r["network"], r["metric"]): [] for r in primary}
    for block in range(32):
        folder = DATA / f"block_{block:02}"
        cert = json.loads((folder / "source.json").read_text())
        assert cert["boards"] == cert["optimal_moves"] == cert["own_teacher_matches"] == 4520
        assert cert["worst_payoff_forgone"] == 0
        rows = load_rows(folder / "visits.jsonl.gz")
        for (family, metric), diffs in cells.items():
            paired = {
                (r["split"], r["fit"], r["method"]): r
                for r in rows
                if r["network"] == family and r["source"] == "fixed"
            }
            diffs.append(
                mean(
                    paired[k, i, "vnm_terminal_copies"][metric] - paired[k, i, "direct"][metric]
                    for k in range(2)
                    for i in range(2)
                )
            )
    for row in primary:
        x = cells[row["network"], row["metric"]]
        assert row["block_differences"] == pytest.approx(x)
        assert row["difference"]["n"] == 32 and row["difference"]["sd"] == pytest.approx(stdev(x))
        assert row["difference"]["mean"] == pytest.approx(mean(x))


def test_full_choice_construction_and_randomized_partitions():
    rows = load_rows(DATA / "finite_bounds.jsonl.gz")
    assert len(rows) == 32
    assert all(r["rank_matches"] == r["boards"] == 4520 and r["edges"] == 16167 for r in rows)
    assert all(
        r["minimal_frequency_mse"] > 0 and all(n == 4520 for n in r["observed_boards"]) for r in rows
    )
    assert all(r["common_test_boards"] < min(v[2] for v in r["partition_board_counts"]) for r in rows)


def test_full_game_enumeration_and_changed_menu_labels():
    pytest.importorskip("scipy")
    from policy_value_isomorph.phase_iv_game import game
    from policy_value_isomorph.phase_iv_source import actors
    from policy_value_isomorph.phase_iv_metrics import Evaluation
    from policy_value_isomorph.pilot_exact import play_match

    b = game()
    folder = DATA / "block_00"
    raw, sources, opponents = actors(0, folder)
    ev = Evaluation(sources["randomized"], opponents)

    def law(p):
        return {b.states[i]: {a: float(p[i, a]) for a in np.flatnonzero(p[i])} for i in b.dec}

    player = law(sources["randomized"])
    wdl = ev.outcomes(sources["randomized"])
    for j, name in enumerate(("uniform", "priority", "perfect")):
        opp = law(opponents[name])
        assert wdl[0, j] == pytest.approx(play_match(b.graph, player, opp), abs=1e-12)
        assert wdl[1, j] == pytest.approx(play_match(b.graph, opp, player)[::-1], abs=1e-12)
    with np.load(folder / "intervention_cases.npz") as data:
        prefix = "fixed_0_0_remove_favorite_"
        idx = data[prefix + "board"]
        allowed = np.unpackbits(data[prefix + "mask"], axis=1)[:, :9].astype(bool)
        assert not np.any(allowed[np.arange(len(idx)), sources["fixed"][idx].argmax(1)])
        assert np.array_equal(np.where(allowed, raw[idx], -np.inf).argmax(1), data[prefix + "answer"])


def test_randomized_temperature_uses_only_observed_validation_counts():
    from policy_value_isomorph.phase_iv_game import game, soft

    b = game()
    folder = DATA / "block_00"
    with np.load(folder / "baseline.npz") as observations, np.load(folder / "visits.npz") as learned:
        counts = b.dense(observations["randomized_0_counts"])
        val = (observations["partition_0"] == 1) & ~b.ends
        target = counts[val] / counts[val].sum(1, keepdims=True)
        scores = b.dense(learned["randomized_0_0_Q_frequency"].clip(0, 1))
    rows = sorted(
        (
            r
            for r in load_rows(folder / "soft.jsonl.gz")
            if r["source"] == "randomized" and r["split"] == r["fit"] == 0 and r["method"] == "Q_direct"
        ),
        key=lambda r: r["temperature"],
    )
    assert len(rows) == 4
    errors = [np.abs(soft(b, scores, r["temperature"])[val] - target).sum(1).mean() / 2 for r in rows]
    assert [r["validation_tv"] for r in rows] == pytest.approx(errors)
    assert [r["selected"] for r in rows] == [j == np.argmin(errors) for j in range(4)]
