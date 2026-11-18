import csv
import gzip
import hashlib
import json
from collections import Counter
from fractions import Fraction
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
