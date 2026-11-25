"""Freeze the discovery ranking and summarize paired confirmation and whole-play observations."""

import argparse
from collections import defaultdict
from dataclasses import replace
import gzip
import json
import numpy as np

from compare_visit_utilities import DEST, GAMES, HYBRID, REFINE, label, rules_for
from report_vnm_wins import estimate
from policy_value_isomorph.phase_iv_statistics import holm
from policy_value_isomorph.visit_selection import UTILITIES, VisitSelection, path_values


def dump(name, data):
    encoded = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
    path = DEST / name
    path.write_bytes(gzip.compress(encoded, mtime=0) if name.endswith(".gz") else encoded)


def diagnostic(paths, utility):
    """Within-fit means and slopes from the evaluated seat. Require both observed outcomes for differences."""
    outcomes = np.array([row[0] for row in paths])
    lengths = np.array([len(row[1]) for row in paths])
    turns = np.array([row[2] for row in paths])
    last_sign = turns * np.where(lengths % 2, 1, -1)
    scores = np.array([path_values(row[1], row[2]) for row in paths])
    columns = {
        name: scores[:, UTILITIES.index(name)]
        for name in dict.fromkeys((utility, "signed_90", "unsigned_90", "unsigned_last", "signed_last"))
    }
    columns.update(turns=lengths, chooser_moved_last=(last_sign == 1).astype(float))
    counts = {name: int((outcomes == value).sum()) for value, name in ((1, "win"), (0, "draw"), (-1, "loss"))}
    results = {}
    for name, values in columns.items():
        for value, group in ((1, "win"), (0, "draw"), (-1, "loss")):
            if counts[group]:
                results[f"{name}/{group}"] = float(values[outcomes == value].mean())
        if counts["win"] and counts["loss"]:
            results[f"{name}/win_minus_loss"] = results[f"{name}/win"] - results[f"{name}/loss"]
        if name != "turns" and np.ptp(lengths):
            centered = lengths - lengths.mean()
            results[f"{name}/per_turn"] = float(centered @ values / (centered @ centered))
    return counts, results


def collect(stage, winner):
    rules, _ = rules_for(stage)
    expected = {label(r) for r in rules}
    blocks, diagnostics = {}, []
    for g in GAMES:
        by_method = defaultdict(list)
        for b in range(8) if stage == "discovery" else range(8, 32 if g == "tictactoe" else 24):
            with gzip.open(DEST / stage / g / f"block_{b:02}.json.gz", "rt") as file:
                data = json.load(file)
            assert (g, b) == (data["game"], data["block"])
            fits = defaultdict(list)
            for r in data["rows"]:
                w, d, l = r["wdl"]
                assert np.isclose(w + d + l, 1)
                metric = dict(win=w, draw=d, loss=l, agreement=r["agreement"])
                if "self_wdl" in r:
                    sw, sd, sl = r["self_wdl"]
                    assert np.isclose(sw, sl)
                    metric.update(self_win=sw, self_draw=sd)
                fits[r["method"]].append((r["split"], r["fit"], metric))
                if "paths" in r:
                    assert winner is not None and r["method"] == winner.name and len(r["paths"]) == 64
                    if g != "tictactoe":
                        assert [v[0] for v in r["paths"]] == r["external"]
                    counts, values = diagnostic(r["paths"], winner.utility)
                    diagnostics.append(
                        dict(game=g, block=b, split=r["split"], fit=r["fit"], counts=counts, values=values)
                    )
            assert set(fits) == expected
            for method, rows in fits.items():
                assert sorted((k, f) for k, f, _ in rows) == [(0, 0), (0, 1), (1, 0), (1, 1)]
                by_method[method].append({m: np.mean([v[m] for _, _, v in rows]) for m in rows[0][2]})
        blocks[g] = {
            method: {m: np.array([row[m] for row in group]) for m in group[0]}
            for method, group in by_method.items()
        }
    return blocks, diagnostics


def rank(blocks):
    def avg(name):
        return np.mean([blocks[g][name]["win"].mean() for g in GAMES])

    names = [label(rule) for rule in rules_for("discovery")[0]]
    order = sorted(names, key=avg, reverse=True)
    win = VisitSelection.parse(order[0])

    def best(field, value):
        return next(n for n in order if getattr(VisitSelection.parse(n), field) == value)

    leaders = {g: max(names, key=lambda n: blocks[g][n]["win"].mean()) for g in GAMES}
    keep = [
        *order[:2],
        best("network", "V"),
        best("network", "Q"),
        *(
            [
                VisitSelection("Q", "uniform_both", "signed_last_90").name,
                replace(win, utility="signed_last_90").name,
                replace(win, paths=8).name,
                best("utility", "sqrt_last_90"),
                best("utility", "square_last_90"),
            ]
            if REFINE
            else [best("utility", "signed_90"), best("utility", "unsigned_90")]
        ),
        replace(win, network="Q" if win.network == "V" else "V").name,
        *leaders.values(),
    ]
    lock = dict(
        winner=win.name, runner=order[1], best_by_game=leaders, confirm=list(dict.fromkeys(keep)), rank=order
    )
    path = DEST / "selection.json"
    if path.exists() and json.loads(path.read_text()) != lock:
        raise ValueError("discovery ranking changed relative to the frozen rule")
    dump("selection.json", lock)
    return win


def report(stage):
    winner = rules_for(stage)[1]
    blocks, diagnostics = collect(stage, winner)
    if stage == "discovery":
        winner = rank(blocks)
    elif stage == "state_scores":
        sums, _ = collect("board_sums", winner)
        for game in GAMES:
            blocks[game].update(sums[game])
    elif stage == "confirmation":
        matched, _ = collect("matched", winner)
        for game in GAMES:
            assert not (blocks[game].keys() & matched[game].keys())
            blocks[game].update(matched[game])
    estimates, contrasts = [], []
    rng = np.random.default_rng(21772)
    resamples = {}
    for g in GAMES:
        n = len(blocks[g][winner.name]["win"])
        indices = np.arange(n)
        strata = [indices[::2], indices[1::2]] if g == "connect4" else [indices]
        resamples[g] = np.hstack([rng.choice(s, (5000, len(s))) for s in strata])
    for game in (*GAMES, "equal_games"):
        group = GAMES if game == "equal_games" else [game]
        for method, metrics in blocks[GAMES[0]].items():
            for metric in metrics:
                vals = [blocks[g][method][metric] for g in group]
                est = estimate(vals)
                est.pop("p")
                estimates.append(dict(game=game, method=method, metric=metric, **est))
                if method != winner.name and metric in ("win", "agreement"):
                    diffs = [blocks[g][winner.name][metric] - v for g, v in zip(group, vals)]
                    boot = np.mean([d[resamples[g]].mean(1) for g, d in zip(group, diffs)], axis=0)
                    contrasts.append(
                        dict(
                            game=game,
                            reference=method,
                            winner=winner.name,
                            metric=metric,
                            **estimate(diffs),
                            holm_p=None,
                            bootstrap_ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
                            block_differences=[d.tolist() for d in diffs],
                        )
                    )
    holm(contrasts)
    dump(stage + "_estimates.json.gz", estimates)
    dump(stage + "_contrasts.json.gz", contrasts)
    if stage == "confirmation" and REFINE:
        # Both seats continue uniformly. With constant frequencies, V and Q therefore
        # have the very same choices for every board, path seed and opening.
        assert winner.continuation == "uniform_both"
        flat = replace(winner, flat_score=True).name
        architectures = []
        standards = []
        for net in ("V", "Q"):
            selected = replace(winner, network=net).name
            references = [f"reference/{net}/copies/direct", flat, "original"]
            if net == "V":
                references.append(label(HYBRID))
            for game in (*GAMES, "equal_games"):
                group = GAMES if game == "equal_games" else (game,)
                for ref in references:
                    comparisons = standards if ref == label(HYBRID) else architectures
                    for metric in ("win", "agreement"):
                        delta = [blocks[g][selected][metric] - blocks[g][ref][metric] for g in group]
                        comparisons.append(
                            dict(
                                game=game,
                                method=selected,
                                reference=ref,
                                metric=metric,
                                **estimate(delta),
                                holm_p=None,
                                block_differences=[d.tolist() for d in delta],
                            )
                        )
        holm(architectures)
        dump("architecture_contrasts.json.gz", architectures)
        holm(standards)
        dump("standard_comparison.json.gz", standards)
    if diagnostics:
        results = []
        for g in GAMES:
            rows = [r for r in diagnostics if r["game"] == g]
            names = sorted({name for r in rows for name in r["values"]})
            for m in names:
                within = defaultdict(list)
                for r in rows:
                    if m in r["values"]:
                        within[r["block"]].append(r["values"][m])
                means = np.array([np.mean(v) for v in within.values()])
                est = (
                    estimate([means])
                    if len(means) > 1
                    else dict(mean=float(means[0]), n=[1], sd=None, ci95=None, p=None)
                )
                if not m.endswith(("/win_minus_loss", "/per_turn")):
                    est["p"] = None
                results.append(
                    dict(
                        game=g,
                        metric=m,
                        fits=sum(map(len, within.values())),
                        **est,
                        holm_p=None,
                        original_means=means.tolist(),
                    )
                )
        holm([r for r in results if r["p"] is not None])
        filename = (
            "state_scores_path_diagnostics.json.gz" if stage == "state_scores" else "path_diagnostics.json.gz"
        )
        dump(filename, dict(fits=diagnostics, results=results))
    print(stage, winner.name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("discovery", "confirmation", "state_scores"))
    report(parser.parse_args().stage)
