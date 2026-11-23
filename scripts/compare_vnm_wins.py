"""Evaluate frozen candidate rules on common external games and on their own copies."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import gzip
import json
import os
from pathlib import Path

import numpy as np
import torch
from policy_value_isomorph.phase_iv_fit import designs, masks, selected
from policy_value_isomorph.phase_iv_game import game as finite_game, seed as small_seed, OUT as SMALL
from policy_value_isomorph.phase_iv_metrics import Evaluation
from policy_value_isomorph.phase_iv_source import actors
from policy_value_isomorph.phase_v import OUT, SIZE, native, seed, install_source, prefixes, spec
from policy_value_isomorph.win_selection import FinitePlanner, Selection, candidates

DEST = Path(
    os.environ.get(
        "PVI_WIN_SELECTION_OUTPUT", Path(__file__).resolve().parents[1] / "experiments/win_selection"
    )
)
GAMES = ("tictactoe", "connect4", "reversi4")


def external_pairs(e, g, block, split, fit):
    rng = np.random.default_rng(seed(g, block, 410, split, fit))
    pairs = []
    for opponents in ((-1,), (9, 10), (-2,), (-3,)):
        varied = prefixes(e, 4, rng, 6)
        for j in range(8):
            opening = e.start() if j < 4 else varied[j - 4]
            pairs.append((opening, opponents[j % len(opponents)], int(rng.integers(2**32))))
    return pairs


def larger(g, block, rules, self_games):
    origin = OUT / g / f"block_{block:02}"
    engine = native().Engine(g == "reversi4")
    install_source(engine, origin)
    rng = np.random.default_rng(seed(g, block, 411))
    for idx in (9, 10):
        engine.set(idx, True, 0, rng.normal(0, 0.1, SIZE))
    records, cases = [], []
    for k in (0, 1):
        with np.load(origin / f"data_{k}_0.npz") as d:
            ids = d["test"]
            states = d["states"][ids].tolist()
            weights = d["equal"]
            truth = engine.profiles(states, spec(0), False)[:, :-2].argmax(1)
        for f in (0, 1):
            pairs = external_pairs(engine, g, block, k, f)
            cases.append(dict(split=k, fit=f, pairs=pairs))
            with np.load(origin / f"fit_{k}_0_{f}.npz") as data:
                for net, model in (("V", 1), ("Q", 2)):
                    engine.set(model, net == "Q", 0, data[f"m{model}"])
                    for budget in sorted({r.budget for r in rules if r.network == net}):
                        planner = native().WinPlanner(engine, model, seed(g, block, 412, k, f), budget)
                        for rule in (r for r in rules if r.network == net and r.budget == budget):
                            settings = rule.native_args()
                            picks = np.array([planner.choose(s, *settings) for s in states])
                            games = [
                                planner.result(s, seat, opp, u, *settings, False)
                                for s, opp, u in pairs
                                for seat in (0, 1)
                            ]
                            outcomes = np.asarray(games)
                            row = dict(
                                split=k,
                                fit=f,
                                method=rule.name,
                                agreement=float(np.average(picks == truth, weights=weights)),
                                external=games,
                                wdl=[float(np.mean(outcomes == v)) for v in (1, 0, -1)],
                            )
                            if self_games:
                                # Both seats in the exact same deterministic game. Record first-seat outcomes.
                                own = [planner.result(s, 0, opp, u, *settings, True) for s, opp, u in pairs]
                                row["self_first"] = own
                                draws = float(np.mean(np.array(own) == 0))
                                row["self_wdl"] = [(1 - draws) / 2, draws, (1 - draws) / 2]
                            records.append(row)
                        del planner
    return dict(cases=cases, rows=records)


def smaller(block, rules, self_games):
    b = finite_game()
    origin = SMALL / f"block_{block:02}"
    _, source, opponents = actors(block + 10000, origin)
    evaluator = Evaluation(source["fixed"], opponents)
    truth = source["fixed"].argmax(1)
    rows = []
    with np.load(origin / "baseline.npz") as labels, np.load(origin / "visits.npz") as archived:
        for k in (0, 1):
            split = b.split(block, k)
            held = (split == 2) & ~b.ends
            counts = b.dense(labels[f"fixed_{k}_counts"])
            y = (counts / counts.sum(1, keepdims=True).clip(1))[b.legal]
            for f in (0, 1):
                for net in ("V", "Q"):
                    choices = [r for r in rules if r.network == net]
                    if not choices:
                        continue
                    tr, va, _ = masks(split, net)
                    pred, _ = selected(
                        designs()[net], y, tr, va, small_seed(block, 51 if net == "V" else 52, k, f)
                    )
                    planner = FinitePlanner(b, b.dense(pred.clip(0, 1)), source["fixed"])
                    assert np.array_equal(
                        planner.direct[b.dec].argmax(1), archived[f"fixed_{k}_{f}_{net}_direct_choice"]
                    )
                    for rule in choices:
                        law = planner.law(rule)
                        seats = evaluator.outcomes(law)[:, :-1, :]
                        wdl = seats.mean(0).T @ evaluator.weights
                        row = dict(
                            split=k,
                            fit=f,
                            method=rule.name,
                            agreement=float((law.argmax(1)[held] == truth[held]).mean()),
                            external=seats.tolist(),
                            wdl=wdl.tolist(),
                        )
                        if self_games:
                            own = b.endings(law, law)[0]
                            row["self_first"] = own.tolist()
                            row["self_wdl"] = ((own + own[::-1]) / 2).tolist()
                        rows.append(row)
    return dict(opponents=list(opponents), rows=rows)


def study_rules(stage):
    direct = [Selection(n, "copies", "direct") for n in ("V", "Q")]
    if stage == "discovery":
        return candidates() + direct
    lock = json.loads((DEST / ("selection.json" if stage == "tuning" else "standard.json")).read_text())
    chosen = Selection.parse(lock["winner"])
    if stage == "tuning":
        return [replace(chosen, budget=b, fitted_ties=tie) for b in (4, 1, 16) for tie in (False, True)]
    rules = [
        chosen,
        Selection.parse(lock["runner"]),
        Selection.parse(lock["learned"]),
        Selection.parse(lock["hybrid"]),
        Selection.parse(lock["outcome"]),
        Selection("Q", "copies", "last"),
        *direct,
    ]
    rules += [
        replace(chosen, finish=who) for who in ("copies", "guarded", "mixture", "guarded_mixture", "original")
    ]
    rules += [
        replace(chosen, prize=u)
        for u in ("last", "sum", "discounted", "win", "points", "quarter_draw", "hybrid")
    ]
    rules += [replace(chosen, network=n) for n in ("V", "Q")]
    rules += [replace(chosen, fitted_ties=not chosen.fitted_ties), Selection.parse(lock["grid_winner"])]
    if "mixture" in chosen.finish:
        rules += [replace(chosen, budget=b) for b in (1, 4, 16)]
    rules += [Selection.parse(s) for s in lock["game_winners"].values()]
    # Re-evaluate both architectures under the selected rule, direct and the previous standard.
    rules += [Selection(n, "copies", "last") for n in ("V", "Q")]
    return list(dict.fromkeys(rules))


def run(job):
    stage, g, block, rules = job
    path = DEST / stage / g / f"block_{block:02}.json.gz"
    if path.exists():
        return g, block
    torch.set_num_threads(1)
    result = (
        smaller(block, rules, stage == "confirmation")
        if g == "tictactoe"
        else larger(g, block, rules, stage == "confirmation")
    )
    result.update(game=g, block=block)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as stream:
        json.dump(result, stream, separators=(",", ":"), allow_nan=False)
    tmp.replace(path)
    return g, block


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("discovery", "tuning", "confirmation"))
    ap.add_argument("--games", nargs="+", default=GAMES, choices=GAMES)
    ap.add_argument("--blocks", nargs="+", type=int)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    rules = study_rules(args.stage)
    jobs = []
    for g in args.games:
        ids = range(8) if args.stage != "confirmation" else range(8, 32 if g == "tictactoe" else 24)
        for i in ids:
            if args.blocks is None or i in args.blocks:
                jobs.append((args.stage, g, i, rules))
    with ProcessPoolExecutor(args.workers) as pool:
        for done in pool.map(run, jobs):
            print(*done, flush=True)
