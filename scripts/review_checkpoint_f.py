"""Replay Phase V confirmation games and compare the standard with direct on the same fit.

Replacement choices only remove an action at the current board. The complete-game
utility and the legal moves available later in the game are unchanged.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

from compare_vnm_wins import DEST, GAMES, external_pairs
from report_vnm_wins import estimate
from policy_value_isomorph.phase_iv_fit import designs, masks, selected
from policy_value_isomorph.phase_iv_game import OUT as SMALL, game as small_game, seed as small_seed
from policy_value_isomorph.phase_iv_metrics import Evaluation
from policy_value_isomorph.phase_iv_source import actors
from policy_value_isomorph.phase_iv_statistics import holm
from policy_value_isomorph.phase_v import OUT, SIZE, install_source, native, panel, seed, spec
from policy_value_isomorph.win_selection import FinitePlanner, Selection, standard_vnm

CHECKPOINT = Path(__file__).resolve().parents[1] / "experiments/checkpoint_f"
METHODS = [Selection(n, "copies", "direct") for n in ("V", "Q")] + [standard_vnm(n) for n in ("V", "Q")]


def menus(scores: np.ndarray, legal: np.ndarray):
    """Return the frozen favorite, its next choice and the offered root menu."""
    truth = np.where(legal, scores, -np.inf).argmax(1)
    changed = legal.sum(1) >= 2
    offered = legal.copy()
    offered[np.flatnonzero(changed), truth[changed]] = False
    second = np.where(offered, scores, -np.inf).argmax(1)
    return truth, second, offered


def population(states, legal, truth, second, weights):
    return dict(
        states=states,
        legal_count=legal.sum(1).tolist(),
        favorite=truth.tolist(),
        replacement=second.tolist(),
        weight=weights.tolist(),
    )


def record(games, picks, other, pop, baseline, exact_weights=None):
    """Check each externally played seat or exact seat/opponent probability against the archive."""
    external = np.asarray(games)
    if exact_weights is None:
        wdl = np.array([(external == v).mean() for v in (1, 0, -1)])
    else:
        wdl = external.mean(0).T @ exact_weights
    weights = np.asarray(pop["weight"])
    truth = np.asarray(pop["favorite"])
    agreement = float(np.average(picks == truth, weights=weights))
    if baseline is not None:
        np.testing.assert_allclose(external, baseline["external"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(wdl, baseline["wdl"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(agreement, baseline["agreement"], atol=1e-12, rtol=0)
    metrics = dict(
        win=float(wdl[0]),
        draw=float(wdl[1]),
        loss=float(wdl[2]),
        points=float(wdl[0] - wdl[2]),
        agreement=agreement,
    )
    sizes = np.asarray(pop["legal_count"])
    for name, group, success in (
        ("choice_agreement", sizes >= 2, picks == truth),
        ("replacement", sizes >= 2, other == pop["replacement"]),
        ("competitive_replacement", sizes >= 3, other == pop["replacement"]),
    ):
        metrics[name] = float(np.average(success[group], weights=weights[group]))
    result = dict(picks=picks.tolist(), after_removal=other.tolist(), metrics=metrics)
    if baseline is None:
        result["external"] = external.tolist()
    return result


def small(block, reference):
    b = small_game()
    origin = SMALL / f"block_{block:02}"
    raw, source, opponents = actors(block + 10000, origin)
    ev = Evaluation(source["fixed"], opponents)
    truth, second, offered = menus(raw, b.legal)
    source_games = ev.outcomes(source["fixed"])[:, :-1, :]
    splits = []
    with np.load(origin / "baseline.npz") as labels:
        for k in (0, 1):
            split = b.split(block, k)
            ids = np.flatnonzero((split == 2) & ~b.ends)
            pop = population(b.names[ids].tolist(), b.legal[ids], truth[ids], second[ids], np.ones(len(ids)))
            counts = b.dense(labels[f"fixed_{k}_counts"])
            targets = (counts / counts.sum(1, keepdims=True).clip(1))[b.legal]
            fits = []
            for f in (0, 1):
                rows = {"original": record(source_games, truth[ids], second[ids], pop, None, ev.weights)}
                for n in ("V", "Q"):
                    tr, va, _ = masks(split, n)
                    pred, _ = selected(
                        designs()[n], targets, tr, va, small_seed(block, 51 if n == "V" else 52, k, f)
                    )
                    planner = FinitePlanner(b, b.dense(pred.clip(0, 1)))
                    for rule in (r for r in METHODS if r.network == n):
                        law = planner.law(rule)
                        other = planner.law(rule, offered=offered)
                        rows[rule.name] = record(
                            ev.outcomes(law)[:, :-1, :],
                            law[ids].argmax(1),
                            other[ids].argmax(1),
                            pop,
                            reference[k, f, rule.name],
                            ev.weights,
                        )
                fits.append(rows)
            splits.append(dict(population=pop, fits=fits))
    return splits


def larger(game, block, reference, saved_cases):
    folder = OUT / game / f"block_{block:02}"
    e = native().Engine(game == "reversi4")
    install_source(e, folder)
    rng = np.random.default_rng(seed(game, block, 411))
    for model in (9, 10):
        e.set(model, True, 0, rng.normal(0, 0.1, SIZE))
    splits = []
    for k in (0, 1):
        with np.load(folder / f"data_{k}_0.npz") as d:
            states = d["states"][d["test"]].tolist()
            weights = d["equal"]
            assert (d["partition"][d["test"]] == 2).all()
        width = 17 if game == "reversi4" else 7
        masks_raw = np.array([e.legal(s) for s in states])
        legal = (masks_raw[:, None] & (1 << np.arange(width))) != 0
        raw = e.profiles(states, spec(0), False)[:, :width]
        truth, second, offered = menus(raw, legal)
        allowed = offered.astype(int) @ (1 << np.arange(width))
        pop = population(states, legal, truth, second, weights)
        fits = []
        for f in (0, 1):
            pairs = saved_cases[k, f]
            assert all(
                tuple(s) == tuple(t) and (opp, u) == (opp2, u2)
                for (s, opp, u), (t, opp2, u2) in zip(
                    pairs, external_pairs(e, game, block, k, f), strict=True
                )
            )
            source_games = np.asarray(panel(e, spec(0), pairs))[:, 0]
            rows = {"original": record(source_games, truth, second, pop, None)}
            with np.load(folder / f"fit_{k}_0_{f}.npz") as data:
                for n, model in (("V", 1), ("Q", 2)):
                    e.set(model, n == "Q", 0, data[f"m{model}"])
                    planner = native().WinPlanner(
                        e, model, seed(game, block, 412, k, f), standard_vnm(n).budget
                    )
                    for rule in (r for r in METHODS if r.network == n):
                        args = rule.native_args()
                        picks = np.array([planner.choose(s, *args) for s in states])
                        other = np.array(
                            [planner.choose(s, *args, int(mask)) for s, mask in zip(states, allowed)]
                        )
                        games = [
                            planner.result(s, seat, opp, u, *args, False)
                            for s, opp, u in pairs
                            for seat in (0, 1)
                        ]
                        rows[rule.name] = record(games, picks, other, pop, reference[k, f, rule.name])
            fits.append(rows)
        splits.append(dict(population=pop, fits=fits))
    return splits


def replay(job):
    game, block = job
    torch.set_num_threads(1)
    with gzip.open(DEST / "confirmation" / game / f"block_{block:02}.json.gz", "rt") as stream:
        archive = json.load(stream)
    assert (archive["game"], archive["block"]) == job
    reference = {(r["split"], r["fit"], r["method"]): r for r in archive["rows"]}
    if game == "tictactoe":
        splits = small(block, reference)
    else:
        saved = {(r["split"], r["fit"]): r["pairs"] for r in archive["cases"]}
        splits = larger(game, block, reference, saved)
    return dict(game=game, block=block, splits=splits)


def summarize(records):
    methods = ["original", *[r.name for r in METHODS]]
    metrics = records[0]["splits"][0]["fits"][0]["original"]["metrics"]
    blocks = {}
    resamples = {}
    rng = np.random.default_rng(230011)
    for game in GAMES:
        rows = [r for r in records if r["game"] == game]
        source_ids = [r["block"] for r in rows]
        n = len(rows)
        assert source_ids == list(range(8, 32 if game == "tictactoe" else 24))
        idx = np.arange(n)
        strata = [idx[::2], idx[1::2]] if game == "connect4" else [idx]
        resamples[game] = np.hstack([rng.choice(s, (5000, len(s))) for s in strata])
        blocks[game] = {
            m: {
                metric: np.array(
                    [
                        np.mean([f[m]["metrics"][metric] for split in r["splits"] for f in split["fits"]])
                        for r in rows
                    ]
                )
                for metric in metrics
            }
            for m in methods
        }
    summary = []
    contrasts = []
    for game in (*GAMES, "equal_games"):
        gs = GAMES if game == "equal_games" else (game,)
        for method in methods:
            for metric in metrics:
                vals = [blocks[g][method][metric] for g in gs]
                s = estimate(vals)
                s.pop("p")
                summary.append(dict(game=game, method=method, metric=metric, **s))
        comparisons = [
            (standard_vnm(n).name, Selection(n, "copies", "direct").name, metric)
            for n in ("V", "Q")
            for metric in ("win", "agreement", "competitive_replacement", "points")
        ]
        comparisons.append((standard_vnm().name, "original", "win"))
        for method, ref, metric in comparisons:
            ds = [blocks[g][method][metric] - blocks[g][ref][metric] for g in gs]
            sampled = np.mean([d[resamples[g]].mean(1) for g, d in zip(gs, ds)], axis=0)
            effect = dict(
                game=game,
                method=method,
                reference=ref,
                metric=metric,
                **estimate(ds),
                bootstrap_ci95=np.quantile(sampled, [0.025, 0.975]).tolist(),
                holm_p=None,
                block_differences=[d.tolist() for d in ds],
            )
            if len(gs) == 1:
                d = ds[0]
                effect["original_counts"] = {}
                directions = {"higher": d > 0, "equal": d == 0, "lower": d < 0}
                for label, direction in directions.items():
                    count = int(direction.sum())
                    interval = binomtest(count, len(d)).proportion_ci()
                    effect["original_counts"][label] = dict(count=count, share=count / len(d), ci95=list(interval))
            contrasts.append(effect)
    holm(contrasts)
    return dict(methods=methods, estimates=summary, contrasts=contrasts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CHECKPOINT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    jobs = [(g, b) for g in GAMES for b in range(8, 32 if g == "tictactoe" else 24)]
    native()
    with ProcessPoolExecutor(args.workers) as pool:
        records = list(pool.map(replay, jobs))
    report = summarize(records)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    encoded = "".join(json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n" for r in records).encode()
    (args.output / "observations.jsonl.gz").write_bytes(gzip.compress(encoded, mtime=0))
    print(f"Reviewed {len(records)} originals and {len(records) * 4} paired fits.", flush=True)
