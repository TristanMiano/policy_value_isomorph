"""Replay the frozen Task 61 networks and compare with their archived Python outputs."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from itertools import groupby
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_value_isomorph.pilot_metrics import matchup_rows
from policy_value_isomorph.pilot_protocol import GameGraph, board_id, fingerprint, frozen_policies
from policy_value_isomorph.pilot_recovery import Recovery, predict
from policy_value_isomorph.policy_mlp import TinyMLPPolicy, encode_state
from policy_value_isomorph.q_mlp import TinyMLPQ
from policy_value_isomorph.torch_mlp import TorchMLP
from policy_value_isomorph.value_mlp import TinyMLPValue

PILOT = ROOT / "experiments/tictactoe_pilot"
RESULTS = PILOT / "results"


def original_policy_scores(weights: TinyMLPPolicy, inputs: list[float]) -> list[float]:
    """The archive saved move probabilities, so reconstruct raw scores in its addition order."""
    hidden = weights.b1.copy()
    for x, row in zip(inputs, weights.w1):
        for j, w in enumerate(row):
            hidden[j] += x * w
    scores = weights.b2.copy()
    for h, row in zip(hidden, weights.w2):
        activation = math.tanh(h)
        for j, w in enumerate(row):
            scores[j] += activation * w
    return scores


def compare_choices(stats: Counter, before: list[float], after: list[float], direction: int) -> None:
    """Compare all legal maximizers for X-root scores (minimizers when O moves)."""
    best = max if direction == 1 else min
    old_best, new_best = best(before), best(after)
    old_moves = [i for i, v in enumerate(before) if v == old_best]
    new_moves = [i for i, v in enumerate(after) if v == new_best]
    stats["boards"] += 1
    stats["old_tied_boards"] += len(old_moves) > 1
    stats["new_tied_boards"] += len(new_moves) > 1
    stats["different_best_sets"] += old_moves != new_moves
    stats["different_top_moves"] += old_moves[0] != new_moves[0]
    stats["legal_scores"] += len(before)
    change = max(abs(a - b) for a, b in zip(before, after))
    stats["max_abs_score_change"] = max(stats["max_abs_score_change"], change)


def replay() -> dict:
    manifest = json.loads((PILOT / "manifest.json").read_text())
    run = json.loads((RESULTS / "run.json").read_text())
    if fingerprint(manifest) != run["protocol_sha256"]:
        raise ValueError("pilot manifest differs from the archived run")
    for name in ("recovery_weights.json", "boards.csv.gz", "matchups.csv"):
        if hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() != run["sha256"][name]:
            raise ValueError(f"archived input changed: {name}")
    graph = GameGraph.build(manifest["split"])
    positions = {board_id(s): s for s in graph.states if graph.children[s]}
    weights = json.loads((RESULTS / "recovery_weights.json").read_text())
    player_path = PILOT / "weights.json"
    players = frozen_policies(manifest, player_path)
    opponents = {name: {s: policy(s) for s in positions.values()} for name, policy in players.items()}
    frozen_stats = Counter()
    for name, params in json.loads(player_path.read_text())["models"].items():
        saved = TinyMLPPolicy(**params)
        model = TorchMLP(saved)
        for state in positions.values():
            legal = list(graph.children[state])
            inputs = encode_state(state)
            before = original_policy_scores(saved, inputs)
            after = model.predict([inputs])[0].tolist()
            compare_choices(frozen_stats, [before[a] for a in legal], [after[a] for a in legal], 1)
            old_choice = max(legal, key=lambda a: (before[a], -a))
            frozen_stats["different_public_choices"] += opponents[name][state] != {old_choice: 1}
    with (RESULTS / "matchups.csv").open(newline="") as stream:
        archived = {(r["source"], r["method"], r["opponent"], r["seat"]): r for r in csv.DictReader(stream)}
    stats = {name: Counter() for name in ("policy", "value", "q")}
    outcomes, balanced = Counter(), []
    with gzip.open(RESULTS / "boards.csv.gz", "rt", newline="") as stream:
        for source, rows in groupby(csv.DictReader(stream), key=lambda r: r["source"]):
            w = weights[source]
            source_law = opponents[source]
            models = Recovery(TinyMLPPolicy(**w["policy"]), TinyMLPValue(**w["value"]), TinyMLPQ(**w["q"]),
                              any(len(law) > 1 for law in source_law.values()), {})
            pred = predict(graph, models)
            policy = TorchMLP(models.policy)
            returns = {}
            for row in rows:
                s = positions[row["board"]]
                moves = json.loads(row["legal_moves"])
                if moves != list(graph.children[s]) or s in returns:
                    raise ValueError("duplicate board or different legal moves")
                returns[s] = {int(row["return_move"]): 1}
                inputs = encode_state(s)
                old = original_policy_scores(models.policy, inputs)
                new = policy.predict([inputs])[0].tolist()
                compare_choices(stats["policy"], [old[m] for m in moves], [new[m] for m in moves], 1)
                probabilities = [pred.imitation[s].get(m, 0.0) for m in moves]
                stored_probs = json.loads(row["imitation_probs"])
                policy_stats = stats["policy"]
                prob_change = max(abs(a - b) for a, b in zip(probabilities, stored_probs))
                policy_stats["max_abs_probability_change"] = max(policy_stats["max_abs_probability_change"], prob_change)
                policy_stats["different_archived_top_moves"] += (
                    probabilities.index(max(probabilities)) != stored_probs.index(max(stored_probs))
                )
                state_change = abs(pred.v[s] - float(row["v_hat_x"]))
                stats["value"]["max_abs_current_board_change"] = max(
                    stats["value"]["max_abs_current_board_change"], state_change
                )
                for name, column, scores, move, stored_move in (
                    ("value", "successor_v_hat_x", [pred.v[c] for c in graph.children[s].values()],
                     pred.v_move[s], row["v_move"]),
                    ("q", "q_hat_x", [pred.q[s][m] for m in moves], pred.q_move[s], row["q_move"]),
                ):
                    before = json.loads(row[column])
                    compare_choices(stats[name], before, scores, s.to_move)
                    stats[name]["different_archived_top_moves"] += move != int(stored_move)
                    stats[name]["different_signs"] += sum((a > 0) - (a < 0) != (b > 0) - (b < 0)
                                                          for a, b in zip(before, scores))
            if len(returns) != len(positions):
                raise ValueError(f"incomplete board coverage for {source}")
            methods = dict(source=source_law, behavior_exact=source_law, return_exact=returns,
                           policy_mlp=pred.imitation, value_mlp={s: {m: 1} for s, m in pred.v_move.items()},
                           q_mlp={s: {m: 1} for s, m in pred.q_move.items()})
            for row in matchup_rows(source, graph, methods, opponents, manifest):
                key = tuple(row[k] for k in ("source", "method", "opponent", "seat"))
                old_row = archived.pop(key)
                outcomes["rows"] += 1
                for field in ("win", "draw", "loss", "score"):
                    difference = abs(row[field] - float(old_row[field]))
                    outcomes[field] = max(outcomes[field], difference)
                if row["opponent"] == "panel" and row["seat"] == "balanced":
                    balanced.append({k: row[k] for k in ("source", "method", "win", "draw", "loss", "score")})
    if archived:
        raise ValueError("unreplayed matchup rows")
    return dict(
        torch_version=str(torch.__version__), device="cpu", dtype="float64",
        protocol_sha256=run["protocol_sha256"],
        definitions={
            "policy": "Network trained to copy legal moves. Scores are before normalization. Top moves maximize the score.",
            "value": "Network scoring the board after each move in expected game points for X.",
            "q": "Network scoring the current board and candidate move in expected game points for X.",
            "boards": "Number of positions, counting each original player once per position.",
            "legal_scores": "Number of legal position/move scores compared across all original players.",
            "old_tied_boards / new_tied_boards": "Positions with multiple equally best legal scores in Python / PyTorch.",
            "different_best_sets / different_top_moves": "Positions with different sets of best moves / different first best moves in ascending cell order.",
            "different_archived_top_moves": "Positions whose replayed choice or most probable legal move differs from the archived choice, ties by lowest cell.",
            "max_abs_score_change / max_abs_probability_change": "Largest absolute difference per legal move in network score units / probability from 0 to 1.",
            "max_abs_current_board_change": "Largest absolute change in predicted X points on the decision boards themselves.",
            "different_signs": "Legal scores changing category among negative, exactly zero and positive.",
            "frozen_players": "Scores and choices of the original trained player and five untrained opponents, compared on every decision board. different_public_choices counts changes in their legal move rule.",
            "outcomes": "Count of matchup rows and largest absolute differences from the archive in win, draw and loss probability and in expected points (win minus loss).",
            "balanced": "Absolute replay probabilities and expected points from the opening, equal weight per seat and per opponent family, equal weight within family. Each row enumerates all continuations.",
        },
        sources=manifest["source_definitions"], opponents=manifest["opponent_definitions"],
        methods=manifest["method_definitions"], frozen_players=frozen_stats, moves=stats,
        outcomes=outcomes, balanced=balanced,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write comparison data as JSON (default: print).")
    args = parser.parse_args()
    report = json.dumps(replay(), indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(report, end="")
    else:
        args.output.write_text(report, encoding="utf-8")
