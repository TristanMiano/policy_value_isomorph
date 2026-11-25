"""Compare complete-game utilities derived exclusively from fitted frequencies."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import gzip
import json
from pathlib import Path
import os
from itertools import product

import numpy as np
import torch
from compare_vnm_wins import GAMES, external_pairs
from policy_value_isomorph.phase_iv_game import game as finite_game, OUT as SMALL, seed as small_seed
from policy_value_isomorph.phase_iv_fit import designs, masks, selected
from policy_value_isomorph.phase_iv_metrics import Evaluation
from policy_value_isomorph.phase_iv_source import actors
from policy_value_isomorph.phase_v import native, SIZE, OUT, install_source, panel, seed, spec
from policy_value_isomorph.phase_v import fit_table
from policy_value_isomorph.phase_v_study import masks as larger_masks
from policy_value_isomorph.win_selection import FinitePlanner, Selection
from policy_value_isomorph.visit_selection import VisitSelection, FiniteVisitPlanner, candidates

CELLS = os.environ.get("PVI_VISIT_INPUTS", "cells") == "cells"
BASE = Path(
    os.getenv("PVI_VISIT_UTILITY_OUTPUT", Path(__file__).resolve().parents[1] / "experiments/visit_utility")
)
DEST = BASE / "cells" if CELLS else BASE
REFINE = os.environ.get("PVI_VISIT_SEARCH") == "refinement"
STREAM_OFFSET = 100 if REFINE else 0
if REFINE:
    if not CELLS:
        raise ValueError("the refinement uses raw board occupancy")
    DEST = DEST / "refinement"
HYBRID = Selection("V", "guarded_mixture", "hybrid", budget=16)
ASSISTED = Selection("V", "mixture", "last")
REFERENCE = ["original", *[Selection(n, "copies", "direct") for n in ("V", "Q")], HYBRID, ASSISTED]


def label(rule):
    if rule == "original":
        return rule
    return ("reference/" if isinstance(rule, Selection) else "") + rule.name


def rules_for_network(rules, network: str):
    """Evaluate the frozen original once, alongside the V methods."""
    return [rule for rule in rules if getattr(rule, "network", "V") == network]


def rules_for(stage):
    if stage == "discovery":
        if REFINE:
            options = product(
                ("V", "Q"),
                ("mixed_reply", "uniform_both"),
                ("signed_last_90", "signed_last_75", "signed_last_97", "sqrt_last_90", "square_last_90"),
                (8, 32, 64),
            )
            return [VisitSelection(n, c, u, paths=p) for n, c, u, p in options], None
        return candidates(), None
    lock = json.loads((DEST / "selection.json").read_text())
    winner = VisitSelection.parse(lock["winner"])
    if stage == "state_scores":
        state_rule = replace(winner, network="V")
        return [state_rule], state_rule
    if stage == "board_sums":
        state_rule = replace(winner, network="V")
        return [replace(state_rule, utility=u) for u in ("signed_90", "unsigned_90")], state_rule
    if stage == "matched":
        # Change only utility, holding the selected network, budget and projected players fixed.
        return [
            replace(winner, utility=u)
            for u in ("signed_last", "signed_90", "unsigned_90", "own_90")
            if replace(winner, utility=u).name not in lock["confirm"]
        ], winner
    rules = [VisitSelection.parse(name) for name in lock["confirm"]]
    rules += [replace(winner, flat_score=True), replace(winner, flat_score=True, flat_moves=True)]
    return list(dict.fromkeys(rules + REFERENCE)), winner


def sample_small(b, law, opponents, frequencies, seed_id):
    """Separate sampled paths for explanation. Main tic-tac-toe outcomes are exact."""
    rng = np.random.default_rng(seed_id)
    records = []
    for group in (["uniform"], [k for k in opponents if k.startswith("random")], ["priority"], ["perfect"]):
        for _ in range(8):
            opponent = opponents[rng.choice(group)]
            for seat in (1, -1):
                state = 0
                path = []
                while not b.ends[state]:
                    p = law[state] if b.side[state] == seat else opponent[state]
                    a = rng.choice(9, p=p)
                    path.append(float(frequencies[state, a]))
                    state = b.child[state, a]
                records.append([int(seat * b.terminal[state]), path, seat])
    return records


def small(block, rules, winner):
    b = finite_game()
    origin = SMALL / f"block_{block:02}"
    _, sources, opponents = actors(block + 21000 + STREAM_OFFSET, origin)
    source = sources["fixed"]
    evaluator = Evaluation(source, opponents)
    rows, training = [], []
    if CELLS:
        cells = np.column_stack([[s.board for s in b.states], b.side])
        inputs = {
            "V": cells[b.after],
            "Q": np.column_stack([cells[b.parent], cells[b.after], np.eye(9)[b.cell]]),
        }
    else:
        inputs = designs()
    with np.load(origin / "baseline.npz") as labels:
        for k in (0, 1):
            split = b.split(block, k)
            held = (split == 2) & ~b.ends
            obs = b.dense(labels[f"fixed_{k}_counts"])
            y = (obs / obs.sum(1, keepdims=True).clip(1))[b.legal]
            for f in (0, 1):
                for net in ("V", "Q"):
                    tr, va, _ = masks(split, net)
                    pred, log = selected(
                        inputs[net], y, tr, va, small_seed(block, 51 if net == "V" else 52, k, f)
                    )
                    training.append(
                        dict(
                            split=k,
                            fit=f,
                            network=net,
                            training=float(np.square(pred[tr] - y[tr]).mean()),
                            boards=int(np.unique(b.parent[tr]).size),
                            candidates=int(tr.sum()),
                            observed=int((obs.sum(1) > 0).sum()),
                            settings=log,
                        )
                    )
                    scores = b.dense(pred.clip(0, 1))
                    blind = FiniteVisitPlanner(b, scores)
                    historical = FinitePlanner(b, scores) if winner else None
                    for r in rules_for_network(rules, net):
                        if r == "original":
                            law = source
                        elif isinstance(r, Selection):
                            law = historical.law(r)
                        else:
                            law = blind.law(r)
                        seats = evaluator.outcomes(law)[:, :-1]
                        row = dict(
                            split=k,
                            fit=f,
                            method=label(r),
                            external=seats.tolist(),
                            wdl=(seats.mean(0).T @ evaluator.weights).tolist(),
                            agreement=float((law.argmax(1)[held] == source.argmax(1)[held]).mean()),
                        )
                        if winner:
                            own = b.endings(law, law)[0]
                            row["self_first"] = own.tolist()
                            row["self_wdl"] = ((own + own[::-1]) / 2).tolist()
                        if r == winner:
                            row["paths"] = sample_small(
                                b, law, opponents, scores, small_seed(block, 217 + STREAM_OFFSET, k, f)
                            )
                        rows.append(row)
    return dict(rows=rows, opponents=list(opponents), training=training)


def large(game, block, rules, winner):
    folder = OUT / game / f"block_{block:02}"
    e = native().Engine(game == "reversi4")
    install_source(e, folder)
    rng = np.random.default_rng(seed(game, block, 214 + STREAM_OFFSET))
    for opponent in (9, 10):
        e.set(opponent, True, 0, rng.normal(0, 0.1, SIZE))
    rows, cases, training = [], [], []
    for k in (0, 1):
        with np.load(folder / f"data_{k}_0.npz") as d:
            use = d["equal"] > 0
            states = d["states"][d["test"]][use].tolist()
            weights = d["equal"][use]
            truth = e.profiles(states, spec(0), False)[:, :-2].argmax(1)
            if CELLS:
                all_states = d["states"].tolist()
                counts = d["counts"]
                total = counts.sum(1, keepdims=True)
                targets = counts / total.clip(1)
                bits = np.asarray([e.legal(s) for s in all_states])
                legal = (bits[:, None] & (1 << np.arange(counts.shape[1]))) != 0
                qm, vm = larger_masks(e, all_states, d["partition"], legal, counts)
                fit_inputs = {
                    model: (e.features(all_states, model == 2, 2), vm if model == 1 else qm)
                    for model in (1, 2)
                }
        for f in (0, 1):
            pairs = external_pairs(e, game, block, k, f, stream=215 + STREAM_OFFSET)
            cases.append(dict(split=k, fit=f, pairs=pairs))
            salt = seed(game, block, 216 + STREAM_OFFSET, k, f)
            with np.load(folder / f"fit_{k}_0_{f}.npz") as fit:
                for model, net in ((1, "V"), (2, "Q")):
                    if CELLS:
                        features, (tr, va) = fit_inputs[model]
                        weights_fit, log, _ = fit_table(
                            features, targets, tr, va, seed(game, block, 40 + model, k, f)
                        )
                        training.append(
                            dict(split=k, fit=f, network=net, observed=int((total > 0).sum()), **log)
                        )
                        e.set(model, net == "Q", 2, weights_fit)
                    else:
                        e.set(model, net == "Q", 0, fit[f"m{model}"])
                    planners = {}
                    for r in rules_for_network(rules, net):
                        diag = None
                        if r == "original":
                            picks = truth
                            games = [row[0] for row in panel(e, spec(0), pairs)]
                            if winner:
                                mirror = [row[0] for row in panel(e, spec(0), pairs, against=spec(0))][::2]
                        elif isinstance(r, Selection):
                            key = ("historical", r.budget)
                            if key not in planners:
                                planners[key] = native().WinPlanner(e, model, salt, r.budget)
                            planner = planners[key]
                            args = r.native_args()
                            picks = [planner.choose(s, *args) for s in states]
                            games = [
                                planner.result(s, seat, opp, u, *args, False)
                                for s, opp, u in pairs
                                for seat in (0, 1)
                            ]
                            if winner:
                                mirror = [planner.result(s, 0, opp, u, *args, True) for s, opp, u in pairs]
                        else:
                            key = ("blind", r.flat_moves, r.paths)
                            if key not in planners:
                                planners[key] = native().VisitPlanner(e, model, salt, r.paths, r.flat_moves)
                            planner = planners[key]
                            args = r.native_args()
                            picks = [planner.choose(s, *args) for s in states]
                            paths = [
                                native().visit_match(planner, e, s, model, seat, opp, u, *args, False)
                                for s, opp, u in pairs
                                for seat in (0, 1)
                            ]
                            games = [row[0] for row in paths]
                            if r == winner:
                                diag = paths
                            if winner:
                                mirror = [
                                    native().visit_match(planner, e, s, model, 0, opp, u, *args, True)[0]
                                    for s, opp, u in pairs
                                ]
                        row = dict(
                            split=k,
                            fit=f,
                            method=label(r),
                            external=games,
                            agreement=float(np.average(picks == truth, weights=weights)),
                            wdl=[float(np.mean(np.array(games) == v)) for v in (1, 0, -1)],
                        )
                        if winner:
                            draws = float(np.mean(np.array(mirror) == 0))
                            row.update(self_first=mirror, self_wdl=[(1 - draws) / 2, draws, (1 - draws) / 2])
                        if diag is not None:
                            row["paths"] = diag
                        rows.append(row)
    return dict(rows=rows, cases=cases, training=training)


def run(job):
    stage, game, block, rules, winner = job
    path = DEST / stage / game / f"block_{block:02}.json.gz"
    if path.exists():
        return
    torch.set_num_threads(1)
    data = small(block, rules, winner) if game == "tictactoe" else large(game, block, rules, winner)
    data.update(game=game, block=block)
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(data, separators=(",", ":"), allow_nan=False).encode(), mtime=0))
    tmp.replace(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("discovery", "confirmation", "matched", "state_scores", "board_sums"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    rules, winner = rules_for(args.stage)
    native()
    jobs = [
        (args.stage, g, i, rules, winner)
        for i in range(32)
        for g in GAMES
        if (i < 8 if args.stage == "discovery" else 8 <= i < (32 if g == "tictactoe" else 24))
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(run, jobs))
    print(f"Finished {len(jobs)} originals and {len(rules)} rules.", flush=True)
