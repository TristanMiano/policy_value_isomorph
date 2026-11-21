"""Equal-source paired inference for repeated fits on randomized finite-game partitions."""

from collections import defaultdict

import numpy as np
from scipy.stats import t

from .phase_iv_game import OUT
from .phase_iv_metrics import read, dump


def records(name):
    for block in range(32):
        yield from read(OUT / f"block_{block:02}" / (name + ".jsonl.gz"))


def describe(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mu = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else None
    h = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else None
    return dict(mean=mu, sd=sd, ci95=[mu - h, mu + h] if h is not None else None, n=n)


def holm(rows):
    tests = sorted((r for r in rows if r["p"] is not None), key=lambda r: r["p"])
    running = 0
    for i, r in enumerate(tests):
        running = max(running, min(1.0, (len(rows) - i) * r["p"]))
        r["holm_p"] = running


def compare(x):
    x = np.asarray(x, dtype=float)
    r = describe(x)
    n = len(x)
    se = r["sd"] / np.sqrt(n) if n > 1 else 0
    r.update(
        p=float(2 * t.sf(abs(r["mean"]) / se, n - 1)) if se else None,
        holm_p=None,
        positive=int((x > 0).sum()),
        equal=int((x == 0).sum()),
        negative=int((x < 0).sum()),
    )
    return r


def block_means(rows, keys, metrics):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        group = tuple(row[k] for k in keys)
        for m in metrics:
            v = row.get(m)
            if isinstance(v, (int, float)) and np.isfinite(v):
                grouped[group][m][row["block"]].append(v)
    result = {}
    for group, ms in grouped.items():
        result[group] = {m: {b: float(np.mean(x)) for b, x in bs.items()} for m, bs in ms.items()}
    return result


def tabulate(rows, keys, metrics, reference):
    groups = block_means(rows, keys, metrics)
    result = []
    for group, ms in groups.items():
        ref = reference(group)
        for m, values in ms.items():
            blocks = sorted(values)
            row = dict(zip(keys, group), metric=m, **describe([values[b] for b in blocks]))
            row["block_means"] = [values[b] for b in blocks]
            row["block_ids"] = blocks
            if ref in groups and m in groups[ref] and ref != group:
                base = groups[ref][m]
                common = sorted(values.keys() & base.keys())
                if common:
                    row["reference"] = dict(zip(keys, ref))
                    row["paired"] = compare([values[b] - base[b] for b in common])
            result.append(row)
    holm([r["paired"] for r in result if "paired" in r])
    return result


def policies():
    for name in ("baseline", "visits", "returns", "oracles", "soft"):
        for r in records(name):
            r["network"] = r.get("network", "policy")
            if name == "soft":
                r["network"] = "scores"
                method = r["method"]
                r["method"] = (
                    method + "_T" + str(r["temperature"]) if r["temperature"] is not None else method
                )
                if r["selected"]:
                    yield {**r, "method": method + "_Tselected"}
            yield r


def definition(net, method):
    """Machine-readable method access, target, continuation and data budget."""
    result = {
        "network": net,
        "decoder": "direct maximum; lowest legal cell ties",
        "collection": "partitioned complete source games; one start per board for fixed, eight for randomized",
        "deployment": "fitted weights",
        "score": "estimated move frequency",
        "continuation": None,
    }
    if net == "scores" and "_T" in method:
        method, temp = method.rsplit("_T", 1)
        result["decoder"] = "equal mixture over maxima" if temp == "0" else "temperature sampling"
        result["temperature"] = temp
    if method.startswith(("V_", "Q_")):
        family, method = method.split("_", 1)
        result["network"] = family
    if method in ("points", "outcomes", "outcomes_score"):
        result.update(
            score="predicted game points"
            if method == "points"
            else "predicted outcome probabilities averaged with fixed game points",
            collection="16 source continuations per legal training candidate",
        )
    elif method in ("imitation", "source", "exact_log"):
        result.update(
            score={
                "imitation": "nine scores fitted to observed moves, normalized on legal cells",
                "source": "original player's decision rule",
                "exact_log": "log of exact original move probability",
            }[method],
            decoder="source law or legal sampling for a randomized player; maximum for a fixed player",
        )
        if method != "imitation":
            result.update(
                deployment="original player" if method == "source" else "complete lookup",
                collection="full source on all reachable decisions",
            )
    elif method.startswith(("exact_", "oracle_")):
        result.update(
            collection="complete source distribution on all boards",
            deployment="complete lookup",
            score="full choice ordering"
            if method == "exact_ordered_board"
            else "full-information move frequencies",
        )
    elif method.startswith("cell_frequency"):
        result.update(score="global count for each cell", deployment="nine fitted cell counts")
    name = method.lower()
    if "vnm" in name or method in ("supplied_exact",) or method.startswith("rollout_"):
        result["decoder"] = "VNM " + result["decoder"]
        result["continuation"] = (
            "fitted direct copies" if "copies" in name else "original player in both seats"
        )
        result["deployment"] = (
            "game rules and fitted weights"
            if "copies" in name
            else "game rules and original continuation law"
        )
        if "oracle" in name:
            result["deployment"] += " and complete frequency lookup"
        if "terminal" in name or "sum" in name or "decay" in name:
            result["score"] = (
                "signed fitted frequency of final move"
                if "terminal" in name
                else (
                    "sum of signed frequencies discounted by 0.9 per turn"
                    if "decay" in name
                    else "sum of signed move frequencies"
                )
            )
            result["continuations_per_move"] = (
                "enumerated"
                if "oracle" in name
                else "one for fixed, 16 for randomized original, one for fitted copies"
            )
        else:
            result["score"] = (
                "game points with fitted draw utility"
                if "compact" in name
                else (
                    "equal mixture of draw utilities -1 and +1"
                    if "random_utility" in name
                    else "supplied game points"
                )
            )
            result["continuations_per_move"] = (
                int(method.split("_")[1]) if method.startswith("rollout_") else "enumerated"
            )
        if "masked" in name:
            result["deployment"] += " and original support mask"
            result.update(
                decoder="VNM temperature sampling among original-supported moves", temperature=0.25
            )
        if "random_utility" in name:
            result["decoder"] = "uniform draw of utility, then maximize its expectation; lowest cell ties"
        if "supplied" in name or name.startswith("rollout"):
            result["collection"] = "supplied payoff and continuation rule; no utility fitted"
    return result


def headline():
    rows = [
        r
        for r in records("visits")
        if r["source"] == "fixed" and r["method"] in ("direct", "vnm_terminal_copies")
    ]
    grouped = block_means(rows, ["network", "method"], ["agreement", "win", "draw", "loss", "points"])
    output = []
    rng = np.random.default_rng(7219)
    sample = rng.integers(0, 32, size=(10000, 32))
    for family in ("V", "Q"):
        for m in ("agreement", "win", "draw", "loss", "points"):
            x, y = (
                np.array([grouped[family, rule][m][b] for b in range(32)])
                for rule in ("direct", "vnm_terminal_copies")
            )
            delta = y - x
            result = compare(delta)
            result["bootstrap_ci95"] = np.quantile(delta[sample].mean(1), [0.025, 0.975]).tolist()
            output.append(
                dict(
                    network=family,
                    metric=m,
                    direct=describe(x),
                    VNM=describe(y),
                    difference=result,
                    block_differences=delta.tolist(),
                )
            )
    holm([r["difference"] for r in output if r["metric"] in ("agreement", "win", "points")])
    return output


def summarize():
    import json

    primary = headline()
    (OUT / "primary.json").write_text(json.dumps(primary, indent=2) + "\n")
    rows = tabulate(
        policies(),
        ["source", "network", "method"],
        [
            "agreement",
            "tv",
            "win",
            "draw",
            "loss",
            "points",
            "visit_agreement",
            "visit_tv",
            "train_loss",
            "penalized_loss_per_candidate",
            "training_candidates",
            "validation_candidates",
            "bounded_train_loss",
            "training_nats",
            "test_exact_loss",
            "outside_source",
            "missing",
            "rms_score_residual",
        ],
        lambda g: (g[0], g[1], "direct") if g[1] in ("V", "Q") else (g[0], "Q", "direct"),
    )
    for row in rows:
        row["definition"] = definition(row["network"], row["method"])
    dump(OUT / "methods.jsonl.gz", rows)

    def aux(name, keys, metrics, reference, transform=None):
        rows = records(name)
        if transform:
            rows = map(transform, rows)
        result = tabulate(rows, keys, metrics, reference)
        for row in result:
            row["experiment"] = name
        return result

    diagnostics = []
    diagnostics += aux(
        "menus",
        ["contextual", "strategy", "budget"],
        [
            "pairs",
            "correct",
            "ordered",
            "unknown",
            "conflicts",
            "calls",
            "accuracy",
            "unresolved_rate",
            "full_to_pair_reversal",
        ],
        lambda g: (g[0], "random pairs", g[2]),
        lambda r: {
            **r,
            "accuracy": r["correct"] / r["pairs"],
            "unresolved_rate": r["unknown"] / r["pairs"],
        },
    )
    diagnostics += aux(
        "sample_laws",
        ["source", "method", "draws"],
        ["boards", "tv", "infinite_kl_boards", "finite_mean_kl", "missed_probability", "extra_probability"],
        lambda g: (g[0], "counts", g[2]),
        lambda r: {**r, "draws": r["n"]},
    )
    diagnostics += aux(
        "coverage",
        ["source", "method", "budget", "design", "group"],
        ["seen", "normal_visit_mass", "boards", "group_seen", "agreement", "tv", "opening_pool_moves"],
        lambda g: (g[0], "random_openings", *g[2:]),
    )
    diagnostics += aux(
        "training_steps",
        ["source", "network", "width", "step"],
        ["agreement", "points", "win", "train_loss", "test_exact_loss"],
        lambda g: (*g[:3], 1),
    )

    def curve_settings():
        for r in records("curves"):
            setting = (
                "half_training"
                if r["method"] == "half_training"
                else str(r["width"]) + "_" + str(r["penalty"])
            )
            yield {**r, "setting": setting}
            if setting != "half_training" and r["selected"]:
                yield {**r, "setting": "selected"}

    curves = tabulate(
        curve_settings(),
        ["source", "network", "setting"],
        ["train_loss", "validation_loss", "test_exact_loss", "agreement", "points", "win"],
        lambda g: (g[0], g[1], "selected"),
    )
    diagnostics.extend({**r, "experiment": "curves"} for r in curves)
    diagnostics += aux(
        "stability",
        ["source", "network"],
        ["frequency_rms", "neighbor_gap_rms", "reversed", "legal_pairs", "changed_ties"],
        lambda g: (g[0], "Q"),
    )
    diagnostics += aux(
        "bounds",
        ["source", "population"],
        [
            "points_given_up",
            "weight",
            "boards",
            "fixed_best_mass",
            "other_equal_mass",
            "strictly_worse_mass",
            "deterministic_tv_floor",
            "maximizing_mixture_tv_floor",
        ],
        lambda g: None,
    )
    diagnostics += aux(
        "interventions",
        ["source", "event", "method"],
        ["tv", "independent_match", "boards"],
        lambda g: (g[0], g[1], "imitation"),
    )
    diagnostics += aux(
        "lotteries",
        ["source", "event", "method"],
        ["error", "draw_utility"],
        lambda g: (g[0], g[1], "direct_bins"),
    )
    dump(OUT / "diagnostics.jsonl.gz", diagnostics)
