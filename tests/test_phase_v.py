"""Independent checks of published pairing, forced moves, and complete-play score signs."""

import gzip
import json
from pathlib import Path
from statistics import mean, stdev
import sys

import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[1] / "experiments" / "phase_v"


def rows(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        return [json.loads(line) for line in stream]


@pytest.mark.parametrize("game", ["connect4", "reversi4"])
def test_primary_uses_same_boards_seats_and_independent_sources(game):
    primary = json.loads((DATA / "primary.json").read_text())
    deltas = {(family, m): [] for family in ("V", "Q") for m in ("agreement", "win", "points")}
    for b in range(24):
        folder = DATA / game / f"block_{b:02}"
        original_games = {
            (r["split"], r["fit"]): np.asarray(r["games"])
            for r in rows(folder / "baseline.jsonl.gz")
            if not r["mixed"]
        }
        diffs = {key: [] for key in deltas}
        for k in range(2):
            with np.load(folder / f"data_{k}_0.npz") as data:
                held = data["equal"]
                assert np.all(data["partition"][data["test"]] == 2)
                assert data["counts"].sum() == len(data["path_ids"]) - (len(data["path_offsets"]) - 1)
            for f in range(2):
                with np.load(folder / f"evaluation_{k}_0_{f}.npz") as stored:
                    assert np.array_equal(stored["games_source_max"][:, :2], original_games[k, f][:, :2])
                    source = stored["p_source_max"]
                    for net in ("V", "Q"):
                        estimates = []
                        for mode in ("direct", "VNM_terminal_copies"):
                            key = f"{net}_{mode}_max"
                            pm = stored["p_" + key]
                            outcomes = stored["games_" + key][:, 0]
                            assert len(outcomes) == 64 and np.allclose(pm.sum(1), 1)
                            estimates.append(
                                [
                                    np.average((source * pm).sum(1), weights=held),
                                    np.mean(outcomes == 1),
                                    np.mean(outcomes),
                                ]
                            )
                        for m, d in zip(("agreement", "win", "points"), np.subtract(*estimates[::-1])):
                            diffs[net, m].append(d)
        for key, data in diffs.items():
            deltas[key].append(mean(data))
    for (net, m), delta in deltas.items():
        row = next(
            r
            for r in primary
            if r["game"] == game and r["method"].startswith(net + "_") and r["metric"] == m
        )
        assert row["block_differences"] == pytest.approx(delta)
        assert row["paired"]["mean"] == pytest.approx(mean(delta))
        assert row["paired"]["sd"] == pytest.approx(stdev(delta))


def test_three_game_weights_and_reversi_pass_population():
    primary = json.loads((DATA / "primary.json").read_text())
    assert len(primary) == 24
    for r in primary:
        if r["game"] != "equal_game_average":
            continue
        within = [
            q["paired"]["mean"]
            for q in primary
            if q["game"] != "equal_game_average"
            and (q["method"], q["metric"]) == (r["method"], r["metric"])
        ]
        assert len(within) == 3 and r["paired"]["mean"] == pytest.approx(mean(within))
    bounds = json.loads((DATA / "bounds.json").read_text())
    g = bounds["decision_groups"]
    assert sum(g) == bounds["reversi4"]["boards"] and g[2] == 8988
    assert bounds["reversi4"]["constructed_matches"] == sum(g)


@pytest.mark.skipif(sys.platform == "win32", reason="optional native simulation is run on Linux/WSL")
def test_learned_complete_play_scores_match_independent_path_sums():
    pytest.importorskip("scipy")
    from policy_value_isomorph.phase_v import actions, install_source, moved, native, spec
    from policy_value_isomorph.phase_v_study import setup_fit, scoring

    for game in ("connect4", "reversi4"):
        rev = game == "reversi4"
        folder = DATA / game / "block_00"
        e = native().Engine(rev)
        install_source(e, folder)
        setup_fit(e, folder, 0, 0, 0)
        with np.load(folder / "data_0_0.npz") as data:
            states = data["states"][data["test"][:7]].tolist()
        for net in (1, 2):
            for who in (0, 1):
                got = [scoring(e, states, spec(net, r, who)) for r in (1, 2, 3)]
                policy = spec(0 if who else net)
                for i, s in enumerate(states):
                    for a in actions(e, s):
                        path = [s] + e.trajectory(e.next(s, a), policy, policy, 917)
                        v = scoring(e, path[:-1], spec(net))
                        signed = [
                            (1 - 2 * h[2]) * row[moved(h, t, rev)] for row, h, t in zip(v, path, path[1:])
                        ]
                        expected = (1 - 2 * s[2]) * np.array(
                            [signed[-1], sum(signed), sum(x * 0.9**j for j, x in enumerate(signed))]
                        )
                        assert [z[i, a] for z in got] == pytest.approx(expected)
