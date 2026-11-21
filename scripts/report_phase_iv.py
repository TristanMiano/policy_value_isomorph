"""Publish the frozen design's primary contrasts, exploratory comparisons and rankings."""

import json
from itertools import chain
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from policy_value_isomorph.phase_iv_game import OUT, game
from policy_value_isomorph.phase_iv_metrics import read, dump
from policy_value_isomorph.phase_iv_statistics import compare, holm, records, summarize, tabulate


def ranking(rows, lower):
    order = sorted(rows, key=lambda r: r["mean"], reverse=not lower)
    result = []
    for r in order:
        result.append({k: r.get(k) for k in ("network", "method", "mean", "sd", "ci95", "n", "definition")})
    if len(order) > 1:
        lead = np.array(order[0]["block_means"])
        for item, r in zip(result, order):
            item["difference_from_first"] = compare(np.array(r["block_means"]) - lead)
        holm([item["difference_from_first"] for item in result[1:]])
    return result


def rankings():
    rows = read(OUT / "methods.jsonl.gz")
    output = {}
    for kind in ("fixed", "randomized"):
        for objective, metric in (
            ("fidelity", "agreement" if kind == "fixed" else "tv"),
            ("strength", "points"),
            ("wins", "win"),
        ):
            data = [r for r in rows if r["source"] == kind and r["metric"] == metric]
            for access in ("all", "independent_observed_moves", "visit_V_Q"):
                subset = data
                if access != "all":
                    subset = [
                        r
                        for r in data
                        if r["definition"]["deployment"]
                        in ("fitted weights", "game rules and fitted weights", "nine fitted cell counts")
                        and "16 source continuations" not in r["definition"]["collection"]
                    ]
                if access == "visit_V_Q":
                    subset = [
                        r
                        for r in subset
                        if r["definition"]["network"] in ("V", "Q")
                        and "game points" not in r["definition"]["score"]
                    ]
                output["/".join([kind, objective, access])] = ranking(subset, metric == "tv")
    for kind in ("fixed", "randomized"):
        interventions = [
            {**r, "error": r["tv"]}
            for r in records("interventions")
            if r["source"] == kind and r["event"].startswith("remove_")
        ]
        values = tabulate(interventions, ["method"], ["error"], lambda g: None)
        output[kind + "/predict_removed_move"] = ranking(values, True)
    for kind in ("shared_draw", "presentation"):
        data = [r for r in records("lotteries") if r["source"] == kind and r["event"] == "changed_odds"]
        values = tabulate(data, ["method"], ["error"], lambda g: None)
        output[kind + "/predict_changed_lottery"] = ranking(values, True)
    (OUT / "rankings.json").write_text(json.dumps(output, separators=(",", ":"), allow_nan=False) + "\n")


def opponent_results():
    panels = {
        "uniform": [0],
        "priority": [1],
        "perfect": [2],
        "untrained_networks": list(range(3, 8)),
        "original": [8],
    }
    for row in chain(records("baseline"), records("visits")):
        seats = np.asarray(row["seats"])
        for group, columns in panels.items():
            results = seats[:, columns].mean(axis=1)
            for seat, outcomes in zip(("X", "O", "both"), [*results, results.mean(axis=0)]):
                yield dict(
                    block=row["block"],
                    source=row["source"],
                    network=row.get("network", "policy"),
                    method=row["method"],
                    opponent=group,
                    seat=seat,
                    **dict(zip(("win", "draw", "loss"), outcomes)),
                    points=float(outcomes[0] - outcomes[2]),
                )


def refinements():
    b = game()
    opponents = tabulate(
        opponent_results(),
        ["source", "network", "method", "opponent", "seat"],
        ["win", "draw", "loss", "points"],
        lambda g: (g[0], g[1], "direct" if g[1] in ("V", "Q") else "source", *g[3:]),
    )
    dump(OUT / "opponents.jsonl.gz", opponents)
    details = []
    for r in records("visits"):
        if r["method"] == "direct":
            continue
        seat = np.array(r["against_direct"])
        for name, value in zip(("win", "draw", "loss"), seat.mean(0)):
            details.append(
                dict(
                    block=r["block"],
                    source=r["source"],
                    network=r["network"],
                    method=r["method"],
                    measure="head_to_head_" + name,
                    value=float(value),
                )
            )
        details.append(
            dict(
                block=r["block"],
                source=r["source"],
                network=r["network"],
                method=r["method"],
                measure="head_to_head_win_minus_loss",
                value=float(np.mean(seat[:, 0] - seat[:, 2])),
            )
        )
        if "move_pairs" in r:
            for label, n in zip(("both_match", "direct_only", "VNM_only", "neither"), r["move_pairs"]):
                details.append(
                    dict(
                        block=r["block"],
                        source=r["source"],
                        network=r["network"],
                        method=r["method"],
                        measure=label,
                        value=n / r["boards"],
                    )
                )
    for r in records("best_reply"):
        for name, v in zip(("win", "draw", "loss"), np.mean(r["seat_wdl"], axis=0)):
            details.append(
                dict(
                    block=r["block"],
                    source="best_possible_against_opponent",
                    network="table",
                    method=r["opponent"],
                    measure=name,
                    value=float(v),
                )
            )
    games = tabulate(details, ["source", "network", "method", "measure"], ["value"], lambda g: None)
    for row in games:
        if row["measure"] == "head_to_head_win_minus_loss":
            row["paired"] = compare(row["block_means"])
    holm([r["paired"] for r in games if "paired" in r])
    dump(OUT / "paired_games.jsonl.gz", games)
    # Equal probability on either side of a reported interval is used for these binomial count intervals.
    recurrence = []
    for row in json.loads((OUT / "primary.json").read_text()):
        d = row["difference"]
        for label in ("positive", "equal", "negative"):
            n = d[label]
            interval = binomtest(n, 32).proportion_ci(0.95)
            recurrence.append(
                dict(
                    network=row["network"],
                    metric=row["metric"],
                    direction=label,
                    blocks=n,
                    fraction=n / 32,
                    ci95=[interval.low, interval.high],
                )
            )
    (OUT / "recurrence.json").write_text(json.dumps(recurrence, indent=2) + "\n")
    calibration = []
    for block in range(32):
        folder = OUT / f"block_{block:02}"
        with np.load(folder / "source.npz") as original, np.load(folder / "sampling.npz") as samples:
            logits = np.zeros((b.n, 9))
            logits[b.dec] = original["scores"]
            first = b.greedy(logits)
            second = b.greedy(np.where(first > 0, -np.inf, logits))
            restricted = 0.75 * first + 0.25 * second
            restricted[b.legal.sum(1) == 1] = first[b.legal.sum(1) == 1]
            for k in range(2):
                ids = np.flatnonzero((b.split(block, k) == 2) & ~b.ends)
                legal = b.legal[ids]
                for label, source in (
                    ("mixture", 0.75 * first + 0.25 * b.uniform),
                    ("top_two", restricted),
                ):
                    for n in (1, 4, 16, 64):
                        counts = samples[f"{label}_{k}_{n}"]
                        for method, p in (
                            ("counts", counts / n),
                            ("plus_one", (counts + legal) / (n + legal.sum(1, keepdims=True))),
                        ):
                            for i in range(5):
                                mask = legal & (np.minimum((5 * p).astype(int), 4) == i)
                                if mask.any():
                                    calibration.append(
                                        dict(
                                            block=block,
                                            split=k,
                                            source=label,
                                            draws=n,
                                            method=method,
                                            lower=i / 5,
                                            upper=(i + 1) / 5,
                                            legal_moves=int(mask.sum()),
                                            prediction=float(p[mask].mean()),
                                            actual=float(source[ids][mask].mean()),
                                        )
                                    )
    dump(OUT / "probability_calibration.jsonl.gz", calibration)


if __name__ == "__main__":
    summarize()
    rankings()
    refinements()
