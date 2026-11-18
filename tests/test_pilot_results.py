import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
from math import fsum, isfinite
from pathlib import Path

import pytest

from policy_value_isomorph.pilot_protocol import fingerprint
from policy_value_isomorph.policy_mlp import TinyMLPPolicy, policy_mlp_action
from policy_value_isomorph.q_mlp import TinyMLPQ, q_mlp_predict
from policy_value_isomorph.tictactoe import state_from_rows
from policy_value_isomorph.value_mlp import TinyMLPValue, value_mlp_predict

PILOT = Path(__file__).resolve().parents[1] / "experiments/tictactoe_pilot"
RESULTS = PILOT / "results"


def test_published_population_and_checkpoint_checksums() -> None:
    manifest = json.loads((PILOT / "manifest.json").read_text())
    run = json.loads((RESULTS / "run.json").read_text())
    assert run["protocol_sha256"] == fingerprint(manifest)
    for name, digest in run["sha256"].items():
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == digest
    counts = Counter()
    with gzip.open(RESULTS / "boards.csv.gz", "rt") as stream:
        for row in csv.DictReader(stream):
            counts[row["source"], row["split"]] += 1
            assert row["source_probs"] == row["behavior_exact_probs"]
            assert sum(Fraction(row[k]) for k in ("fixed_match_mass", "other_tie_mass", "strict_loss_mass")) == 1
    assert counts == {(s, split): n for s in manifest["sources"] for split, n in run["split_decision_counts"].items()}


def test_stored_weights_reproduce_an_oriented_test_decision() -> None:
    with gzip.open(RESULTS / "boards.csv.gz", "rt") as stream:
        row = next(r for r in csv.DictReader(stream) if r["source"] == "minimax" and r["board"] == "..OO.XX.X")
    state = state_from_rows(["..O", "O.X", "X.X"], to_move=-1)
    weights = json.loads((RESULTS / "recovery_weights.json").read_text())["minimax"]
    v = TinyMLPValue(**weights["value"])
    q = TinyMLPQ(**weights["q"])
    legal = state.legal_moves()
    assert row["split"] == "test" and row["return_move"] == "7"
    assert [value_mlp_predict(state.apply_move(a), v) for a in legal] == pytest.approx(json.loads(row["successor_v_hat_x"]))
    assert [q_mlp_predict(state, a, q) for a in legal] == pytest.approx(json.loads(row["q_hat_x"]))
    action = policy_mlp_action(state, TinyMLPPolicy(**weights["policy"]))
    assert json.loads(row["imitation_probs"])[legal.index(action)] == 1


def test_move_comparison_table_reports_defined_numeric_measures() -> None:
    manifest = json.loads((PILOT / "manifest.json").read_text(encoding="utf-8"))
    sources = manifest["sources"]
    methods = manifest["recovery"]["decoders"]
    assert set(manifest["source_definitions"]) == set(sources)
    assert set(manifest["opponent_definitions"]) == set(manifest["opponent_families"])
    assert set(manifest["method_definitions"]) == set(methods)

    measures = manifest["measure_definitions"]
    groups = [*manifest["split_decision_counts"], "all"]
    expected = set(product(sources, methods, groups, ("uniform", "visitation")))
    with (RESULTS / "fidelity.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert set(reader.fieldnames) == manifest["population_definitions"].keys() | measures.keys()
        for row in reader:
            expected.remove((row["source"], row["method"], row["split"], row["weighting"]))
            for measure in measures:
                inapplicable = measure == "agreement" and row["source"] == "stochastic_heuristic"
                if inapplicable or float(row["weight_sum"]) == 0:
                    assert row[measure] == ""
                else:
                    assert isfinite(float(row[measure]))
    assert not expected


def test_payoff_gaps_reproduce_from_saved_moves_and_payoffs() -> None:
    # Independently use the mover's exact payoffs and the stored move distributions.
    boards = defaultdict(list)
    with gzip.open(RESULTS / "boards.csv.gz", "rt") as stream:
        for row in csv.DictReader(stream):
            legal = json.loads(row["legal_moves"])
            sign = 1 if row["to_move"] == "X" else -1
            returns = [sign * Fraction(value) for value in json.loads(row["q_exact_x"])]
            best = max(returns)
            gaps = [float(best - value) for value in returns]
            gap_by_move = dict(zip(legal, gaps))
            player_gaps = {
                method: gap_by_move[int(row[column])]
                for method, column in (("return_exact", "return_move"), ("value_mlp", "v_move"), ("q_mlp", "q_move"))
            }
            for method, column in (("source", "source_probs"), ("behavior_exact", "behavior_exact_probs"),
                                   ("policy_mlp", "imitation_probs")):
                probs = json.loads(row[column])
                player_gaps[method] = fsum(float(Fraction(p)) * gap for p, gap in zip(probs, gaps))
            board = (float(Fraction(row["visitation"])), player_gaps)
            for group in (row["split"], "all"):
                boards[row["source"], group].append(board)

    with (RESULTS / "fidelity.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            population = boards[row["source"], row["split"]]
            weighted = [(1.0 if row["weighting"] == "uniform" else visits, gaps) for visits, gaps in population]
            denominator = fsum(weight for weight, _ in weighted)
            assert float(row["weight_sum"]) == pytest.approx(denominator)
            for field, player in (("source_mean_payoff_gap", "source"), ("method_mean_payoff_gap", row["method"])):
                if denominator:
                    estimate = fsum(weight * gaps[player] for weight, gaps in weighted) / denominator
                    assert float(row[field]) == pytest.approx(estimate)
                else:
                    assert row[field] == ""


def test_balanced_matchups_and_paired_panel_differences() -> None:
    manifest = json.loads((PILOT / "manifest.json").read_text())
    with (RESULTS / "matchups.csv").open() as stream:
        rows = {(r["source"], r["method"], r["opponent"], r["seat"]): r for r in csv.DictReader(stream)}
    assert len(rows) == 900
    for source in manifest["sources"]:
        for method in manifest["recovery"]["decoders"]:
            panel = rows[source, method, "panel", "balanced"]
            for field in ("win", "draw", "loss", "score"):
                opponent_mean = 0
                for family, weight in manifest["opponent_families"].items():
                    seats = [float(r[field]) for (s, m, _, seat), r in rows.items()
                             if (s, m) == (source, method) and seat != "balanced" and r["family"] == family]
                    opponent_mean += weight * sum(seats) / len(seats)
                assert float(panel[field]) == pytest.approx(opponent_mean)
                source_value = float(rows[source, "source", "panel", "balanced"][field])
                assert float(panel["delta_" + field]) == pytest.approx(opponent_mean - source_value)
