"""Compare direct decoding and sampled trajectory utility after fitting observed move frequencies."""

import argparse
import csv
import gzip
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from policy_value_isomorph.coverage_sampling import enumerated_counts
from policy_value_isomorph.neural_game_control import fit_values, train_source
from policy_value_isomorph.pilot_exact import evaluate_source
from policy_value_isomorph.pilot_metrics import fidelity_rows, matchup_rows
from policy_value_isomorph.pilot_protocol import GameGraph, State, board_id, frozen_policies
from policy_value_isomorph.policy_mlp import encode_state
from policy_value_isomorph.q_mlp import TinyMLPQ, q_mlp_predict
from policy_value_isomorph.trajectory_utility import MoveSampler, trajectory_scores
from policy_value_isomorph.value_mlp import TinyMLPValue, value_mlp_predict

PILOT = ROOT / "experiments/tictactoe_pilot"
DEFAULT_OUTPUT = ROOT / "experiments/neural_tictactoe_control"


def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def collect_counts(sampler: MoveSampler, graph: GameGraph):
    """Use independent games for the three board groups. Count only parents belonging to that group."""
    empty = np.array([graph.states.index(State.initial())])
    opening_counts = sampler.visits(empty, 256, 8100)
    counts, _ = enumerated_counts(sampler, graph, 32)
    return opening_counts, counts


def study(directory: Path) -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((PILOT / "manifest.json").read_text(encoding="utf-8"))
    graph = GameGraph.build(manifest["split"])
    decisions = [s for s in graph.states if graph.children[s]]
    policy, optimal, certificate = train_source(graph, decisions, 7101)
    print(f"The trained player preserved the best guaranteed outcome on {certificate['optimal_moves']} "
          f"of {certificate['boards']} decision boards. Its table name is optimal_net.", flush=True)
    checkpoints = {"optimal_source": policy.state_dict()}
    policies = frozen_policies(manifest, PILOT / "weights.json")
    opponents = {name: {s: player(s) for s in decisions} for name, player in policies.items()}
    legacy = json.loads((PILOT / "results/recovery_weights.json").read_text(encoding="utf-8"))
    sources = {"optimal_net": optimal, **{name: opponents[name] for name in manifest["sources"]}}
    for name, definition in manifest["source_definitions"].items():
        print(f"{definition} Its table name is {name}.", flush=True)

    comparisons, games, fits, coverage = [], [], {}, {}
    with (directory / "visits.csv.gz").open("wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0
    ) as zipped, io.TextIOWrapper(zipped, newline="", encoding="utf-8") as stream:
        visits_writer = csv.writer(stream, lineterminator="\n")
        visits_writer.writerow(("source","split","board","move","next_board","opening_count","count",
                               "parent_visits","frequency"))
        for source, laws in sources.items():
            print(f"Sampling and fitting {source}.", flush=True)
            sampler = MoveSampler(graph, laws)
            opening, counts = collect_counts(sampler, graph)
            scores, totals = sampler.frequencies(counts)
            coverage[source] = dict(opening_distinct_boards=len(np.unique(sampler.parent[opening > 0])),
                                    opening_visits=int(opening.sum()), stratified_visits=int(counts.sum()))
            for i, (s, a) in enumerate(sampler.edges):
                visits_writer.writerow((source, graph.split[s], board_id(s), a, board_id(graph.states[sampler.child[i]]),
                                        opening[i], counts[i], int(totals[i]), scores[i]))
            exact = evaluate_source(graph, laws)
            paths = sampler.rollout_paths(64 if sampler.randomized else 1, 8201)
            methods = {"source": laws}
            diagnostics = {}
            parent_split = np.array([graph.split[s] for s, _ in sampler.edges])
            test = parent_split == "test"
            val = parent_split == "validation"
            train = parent_split == "train"
            tensors = torch.tensor([encode_state(s) for s in graph.states])
            after = tensors[sampler.child]
            current = tensors[sampler.parent]
            moves = torch.tensor([a for _, a in sampler.edges])
            action = torch.nn.functional.one_hot(moves, 9).float()
            target = torch.from_numpy(scores).float()

            def record(name: str, utilities: np.ndarray, variance=None, prediction_error=None) -> None:
                # Utilities use X's perspective. O chooses the lowest score. Ties use the first legal cell.
                grid = np.full(sampler.legal.shape, -np.inf)
                grid[sampler.parent, moves.numpy()] = sampler.sign * utilities
                selected = grid.argmax(axis=1)
                methods[name] = {s: {int(selected[i]): 1} for i, s in enumerate(graph.states) if sampler.active[i]}
                mc_error = None if variance is None else float(np.sqrt(variance[test].mean()))
                diagnostics[name] = (prediction_error, mc_error)

            def compare(name: str, x_scores: np.ndarray, prediction_error=None) -> None:
                record(name + "/direct", x_scores, prediction_error=prediction_error)
                for rule, (means, variance) in trajectory_scores(paths, x_scores).items():
                    record(name + "/" + rule, means, variance, prediction_error)
                if name != "visit_table":
                    # Copy the fitted network's direct rule on both sides after the forced first move.
                    # This comparison can plan a whole game without querying the original policy.
                    copies = MoveSampler(graph, methods[name + "/direct"]).rollout_paths(1, 8201)
                    for rule, (means, variance) in trajectory_scores(copies, x_scores).items():
                        record(name + "/" + rule + "_copies", means, variance, prediction_error)

            # All-board count reference. It has direct source observations on the evaluation boards.
            compare("visit_table", sampler.sign * scores, 0.0)
            if source in legacy:
                v = TinyMLPValue(**legacy[source]["value"])
                q = TinyMLPQ(**legacy[source]["q"])
                v_scores = np.array([value_mlp_predict(s, v) for s in graph.states])[sampler.child]
                q_scores = np.array([q_mlp_predict(s, a, q) for s, a in sampler.edges])
                # The published weights predict an X payoff. Use the final output as U, without summing returns.
                for name, original_scores in (("pilot_value_mlp", v_scores), ("pilot_q_mlp", q_scores)):
                    record(name + "/direct", original_scores)
                    mean, variance = trajectory_scores(paths, original_scores)["vnm_terminal"]
                    record(name + "/vnm_terminal", mean, variance)

            families = (("value_mlp", after), ("transition_mlp", torch.cat((current, after), 1)),
                        ("q_mlp", torch.cat((current, action), 1)))
            for family, x in families:
                training_mask, validation_mask = train.copy(), val.copy()
                if family == "value_mlp":
                    # A resulting board can be shared by distinct current boards. Exclude overlapping inputs.
                    training_mask &= ~np.isin(sampler.child, sampler.child[val | test])
                    validation_mask &= ~np.isin(sampler.child, sampler.child[test])
                for scope in (("held_out", "full_coverage") if source == "optimal_net" else ("held_out",)):
                    full = scope == "full_coverage"
                    train_ids = torch.arange(len(x)) if full else torch.from_numpy(np.flatnonzero(training_mask))
                    val_ids = torch.from_numpy(np.flatnonzero(validation_mask))
                    seeds = (7201, 7202, 7203) if source == "optimal_net" and not full else (7201,)
                    for seed in seeds:
                        name = f"{family}/{scope}/{seed}"
                        model, fit = fit_values(x, train_ids, target[train_ids], val_ids, target, seed, full)
                        key = source + "/" + name
                        fits[key] = fit
                        checkpoints[key] = model.state_dict()
                        with torch.no_grad():
                            estimates = model(x)[:, 0].sigmoid().double().numpy()
                        error = float(np.mean((estimates[test] - scores[test])**2))
                        compare(name, sampler.sign * estimates, error)
                    if full and family == "q_mlp":
                        # Same architecture, data, initial weights and budget. Vary only the fitting loss.
                        model, fit = fit_values(x, train_ids, target[train_ids], val_ids, target, 7201, True,
                                                log_loss=True)
                        name = "q_mlp_log_loss/full_coverage/7201"
                        fits[source + "/" + name] = fit
                        checkpoints[source + "/" + name] = model.state_dict()
                        with torch.no_grad():
                            estimates = model(x)[:, 0].sigmoid().double().numpy()
                        compare(name, sampler.sign * estimates, float(np.mean((estimates[test] - scores[test])**2)))
            for row in fidelity_rows(source, graph, methods, exact):
                if row["split"] in ("all", "test"):
                    mse, mc = diagnostics.get(row["method"], (None, None))
                    comparisons.append(dict(row, test_frequency_mse=mse, test_rollout_score_rms_se=mc))
            games.extend(matchup_rows(source, graph, methods, opponents, manifest))
    torch.save(checkpoints, directory / "weights.pt")
    write_table(directory / "move_comparisons.csv", comparisons)
    write_table(directory / "games.csv", games)
    payload = dict(torch_version=str(torch.__version__), numpy_version=np.__version__, certificate=certificate,
                   coverage=coverage, fits=fits)
    (directory / "training.json").write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Directory for the complete results.")
    output = (parser.parse_args().output_dir or DEFAULT_OUTPUT).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as work:
        stage = Path(work) / "results"
        stage.mkdir()
        study(stage)
        if output.exists():
            for artifact in stage.iterdir():
                if artifact.read_bytes() != (output / artifact.name).read_bytes():
                    raise ValueError(f"Replay differs at {artifact.name}. Use a fresh --output-dir.")
        else:
            stage.rename(output)
    print(f"Verified results in {output}")


if __name__ == "__main__":
    main()
