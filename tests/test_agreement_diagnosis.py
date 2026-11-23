"""Exact regression checks for the whole-play utility and the published interventions."""

import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from policy_value_isomorph.trajectory_utility import trajectory_scores

ROOT = Path(__file__).resolve().parents[1] / "experiments"
DATA = ROOT / "agreement_diagnosis"


def test_centered_path_uses_each_movers_best_and_ignores_padding():
    # Two draws after forcing edge 0. On the first draw the opponent forgoes 0.4
    # and the first player then forgoes 0.1. The second draw ends a turn earlier.
    paths = [np.array([[0, 0]]), np.array([[1, 3]]), np.array([[2, -1]])]
    z = np.array([0.25, -0.2, 0.7, -0.6])
    best = np.array([0.75, -0.6, 0.8, -0.6])
    result = trajectory_scores(paths, z, x_local_best=best)
    assert result["vnm_centered"][0] == pytest.approx([(-0.2 - 0.5) / 2])
    assert result["vnm_terminal"][0] == pytest.approx([(0.7 - 0.6) / 2])
    assert "vnm_centered" not in trajectory_scores(paths, z)


def test_benchmark_reference_and_fit_pairing():
    def records(path):
        with gzip.open(path, "rt") as f:
            return list(map(json.loads, f))

    summary = {(r["game"], r["method"], r["metric"]): r for r in records(DATA / "results.jsonl.gz")}
    for game, n in (("tictactoe", 32), ("connect4", 24), ("reversi4", 24)):
        for net in ("V", "Q"):
            for m in ("match", "win", "draw", "loss", "points"):
                direct = summary[game, net + "_direct", m]["blocks"]
                assert len(direct) == n and summary[game, net + "_centered", m]["blocks"] == pytest.approx(
                    direct
                )
            old = np.array(summary[game, net + "_direct", "match"]["blocks"])
            new = np.array(summary[game, net + "_terminal", "match"]["blocks"])
            lost = np.array(summary[game, net + "_switch", "lost_match"]["blocks"])
            gained = np.array(summary[game, net + "_switch", "gained_match"]["blocks"])
            assert new - old == pytest.approx(gained - lost)
            primary = json.loads((ROOT / "phase_v" / "primary.json").read_text())
            published = next(
                r
                for r in primary
                if r["game"] == game
                and r["method"] == net + "_VNM_terminal_copies_max"
                and r["metric"] == "agreement"
            )
            assert new - old == pytest.approx(published["block_differences"])


@pytest.mark.parametrize("game", ["connect4", "reversi4"])
def test_added_observations_respect_the_original_holdout(game):
    for k in (0, 1):
        with (
            np.load(ROOT / "phase_v" / game / "block_00" / f"data_{k}_0.npz") as orig,
            np.load(DATA / game / "block_00" / f"extra_{k}.npz") as new,
        ):
            n = len(orig["states"])
            reserved = orig["partition"] != 0
            assert np.array_equal(new["states"][:n], orig["states"])
            assert np.array_equal(new["counts"][:n][reserved], orig["counts"][reserved])
            assert np.all(new["training_partition"][n:] == 0)
            assert new["counts"].sum() - orig["counts"].sum() == new["extra_observations"]


@pytest.mark.skipif(sys.platform == "win32", reason="native simulation uses Linux/WSL")
def test_native_centered_utilities_against_explicit_sequences():
    from policy_value_isomorph.phase_v import actions, centered_vnm, install_source, moved, native, spec
    from policy_value_isomorph.phase_v_study import scoring, setup_fit

    for rev, game in ((False, "connect4"), (True, "reversi4")):
        e = native().Engine(rev)
        folder = ROOT / "phase_v" / game / "block_00"
        install_source(e, folder)
        setup_fit(e, folder, 0, 0, 0)
        with np.load(folder / "data_0_0.npz") as d:
            boards = d["states"][d["test"][:5]].tolist()
        for model in (1, 2):
            for who in (False, True):
                actual = scoring(e, boards, centered_vnm(model, who))
                policy = spec(0 if who else model)
                for i, s in enumerate(boards):
                    for a in actions(e, s):
                        path = [s] + e.trajectory(e.next(s, a), policy, policy, 0)
                        z = scoring(e, path[:-1], spec(model))
                        expected = sum(
                            (1 if h[2] == s[2] else -1) * (row[moved(h, c, rev)] - row.max())
                            for row, h, c in zip(z, path, path[1:])
                        )
                        assert actual[i, a] == pytest.approx(expected)


@pytest.mark.parametrize("game", ["tictactoe", "connect4", "reversi4"])
def test_perfect_control_uses_fitting_counts_and_freezes_before_selection(game):
    folder = DATA / game / "block_00"
    for k in (0, 1):
        with np.load(folder / f"control_visits_{k}.npz") as data:
            counts = data["counts"]
            y = counts / counts.sum(1, keepdims=True)
            if game != "tictactoe":
                assert np.all(data["continuation_counts"][:, 4] > 0)
        for f in (0, 1):
            with (
                np.load(folder / f"control_cases_{k}_{f}.npz") as cases,
                np.load(folder / f"control_sums_{k}_{f}.npz") as sums,
            ):
                direct = cases["Q_paths_direct"]
                legal = np.isfinite(direct)
                assert np.array_equal(direct[legal], y[legal])
                assert cases["Q_paths_covered"][legal].all()
                assert np.array_equal(y.argmax(1), sums["Q_paths_decay"])
                terminal = cases["Q_paths_terminal"]
                assert set(np.unique(terminal[legal])) <= {-1, 0, 1}
                assert np.any(terminal.argmax(1) != y.argmax(1))
                if game == "connect4" and k == f == 0:
                    with np.load(folder / "control_visits_0.npz") as d:
                        j = np.flatnonzero((d["states"] == [1, 0, 1]).all(1))[0]
                    assert counts[j].tolist() == [0, 0, 0, 3, 0, 0, 0]
                    assert terminal[j].tolist() == [1, 1, 1, -1, 1, 1, -1]
