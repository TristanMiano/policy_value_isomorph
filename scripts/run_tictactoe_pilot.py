"""Repeat the frozen tic-tac-toe experiment and write its quantitative comparisons."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, TextIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_value_isomorph.pilot_exact import ExactReturns, decode_behavior, evaluate_source
from policy_value_isomorph.pilot_metrics import calibration_rows, fidelity_rows, matchup_rows
from policy_value_isomorph.pilot_protocol import GameGraph, Law, State, board_id, fingerprint, frozen_policies
from policy_value_isomorph.pilot_recovery import Predictions, fit_models, predict
from policy_value_isomorph.symmetry import canonicalize_state

PILOT = ROOT / "experiments/tictactoe_pilot"


def write_rows(stream: TextIO, rows: Iterable[dict], *, header: bool = True) -> None:
    writer = None
    for row in rows:
        if writer is None:
            writer = csv.DictWriter(stream, fieldnames=row.keys(), lineterminator="\n")
            if header:
                writer.writeheader()
        writer.writerow(row)


def board_rows(
    source: str, graph: GameGraph, laws: dict[State, Law], decoded: dict[State, Law],
    exact: ExactReturns, pred: Predictions, orbits: dict[State, str],
) -> Iterator[dict]:
    """List-valued columns are JSON in ascending legal-move order. Exact quantities are ratios."""
    def array(items: Iterable) -> str:
        return json.dumps(list(items), separators=(",", ":"), allow_nan=False)

    for state, law in laws.items():
        children = graph.children[state]
        best = exact.best[state]
        match = law.get(best[0], 0)
        optimal = sum(law.get(a, 0) for a in best)
        yield dict(
            source=source, board=board_id(state), to_move="X" if state.to_move == 1 else "O",
            ply=9-state.board.count(0), orbit=orbits[state], split=graph.split[state],
            visitation=str(exact.visitation[state]), legal_moves=array(children),
            source_move=next(iter(law)) if len(law) == 1 else None,
            source_probs=array(str(law.get(a, 0)) for a in children),
            behavior_exact_probs=array(str(decoded[state].get(a, 0)) for a in children),
            v_exact_x=str(exact.value[state]),
            q_exact_x=array(str(q) for q in exact.action_value[state].values()),
            best_moves=array(best), return_move=best[0],
            fixed_match_mass=str(match), other_tie_mass=str(optimal-match), strict_loss_mass=str(1-optimal),
            imitation_probs=array(pred.imitation[state].get(a, 0.0) for a in children),
            v_hat_x=pred.v[state], successor_v_hat_x=array(pred.v[child] for child in children.values()),
            q_hat_x=array(pred.q[state].values()), v_move=pred.v_move[state], q_move=pred.q_move[state],
        )


def generate(directory: Path, manifest: dict) -> None:
    for name, definition in manifest["source_definitions"].items():
        print(f"{definition} Its table name is {name}.", flush=True)
    graph = GameGraph.build(manifest["split"])
    decisions = [s for s in graph.states if graph.children[s]]
    counts = Counter(graph.split[s] for s in decisions)
    if counts != manifest["split_decision_counts"]:
        raise ValueError("populations differ from the frozen split")
    orbits = {s: board_id(canonicalize_state(s)) for s in decisions}
    policies = frozen_policies(manifest, PILOT / "weights.json")
    tables = {name: {s: policy(s) for s in decisions} for name, policy in policies.items()}
    fidelity, calibration, matchups = [], [], []
    checkpoints, training_counts = {}, {}
    with (directory / "boards.csv.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as boards:
                for i, source in enumerate(manifest["sources"]):
                    print(f"Evaluating {source} ({i + 1} of {len(manifest['sources'])} players).", flush=True)
                    laws = tables[source]
                    exact = evaluate_source(graph, laws)
                    decoded = decode_behavior(laws)
                    models = fit_models(graph, laws, manifest["recovery"])
                    pred = predict(graph, models)
                    methods = {
                        "source": laws, "behavior_exact": decoded,
                        "return_exact": {s: {moves[0]: 1} for s, moves in exact.best.items()},
                        "policy_mlp": pred.imitation, "value_mlp": {s: {a: 1} for s, a in pred.v_move.items()},
                        "q_mlp": {s: {a: 1} for s, a in pred.q_move.items()},
                    }
                    if list(methods) != manifest["recovery"]["decoders"]:
                        raise ValueError("decoder list differs from registered methods")
                    write_rows(boards, board_rows(source, graph, laws, decoded, exact, pred, orbits), header=i == 0)
                    fidelity.extend(fidelity_rows(source, graph, methods, exact))
                    calibration.extend(calibration_rows(source, graph, exact, pred))
                    matchups.extend(matchup_rows(source, graph, methods, tables, manifest))
                    checkpoints[source] = {name: asdict(getattr(models, name)) for name in ("policy", "value", "q")}
                    training_counts[source] = models.counts
    for name, rows in (("fidelity", fidelity), ("calibration", calibration), ("matchups", matchups)):
        with (directory / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            write_rows(stream, rows)
    (directory / "recovery_weights.json").write_text(
        json.dumps(checkpoints, sort_keys=True, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )
    metadata = dict(
        version=manifest["version"], reporting_revision=manifest["reporting_revision"],
        protocol_sha256=fingerprint(manifest), weights_sha256=manifest["weights_sha256"],
        states=len(graph.states), decision_boards=len(decisions), edges=sum(map(len, graph.children.values())),
        decision_orbits=len(set(orbits.values())), split_decision_counts=counts, training=training_counts,
        sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())},
    )
    (directory / "run.json").write_bytes((json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tictactoe_pilot_pytorch",
                        help="Destination for the newly fitted networks and their tables.")
    args = parser.parse_args()
    manifest = json.loads((PILOT / "manifest.json").read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage a complete run before publication. A replay cannot replace a frozen result.
    with tempfile.TemporaryDirectory(dir=output.parent) as work:
        stage = Path(work) / "results"
        stage.mkdir()
        generate(stage, manifest)
        if output.exists():
            files = list(stage.iterdir())
            for path in files:
                existing = output / path.name
                if existing.exists() and existing.read_bytes() != path.read_bytes():
                    raise FileExistsError(f"Result differs at {existing}. Use another --output-dir for this run.")
            for path in sorted(files, key=lambda p: p.name == "run.json"):
                if not (output / path.name).exists():
                    path.replace(output / path.name)
        else:
            stage.rename(output)
    print(f"Verified pilot tables and saved network weights in {output}")


if __name__ == "__main__":
    main()
