"""Publish source-level estimates and paired uncertainty for the three named games."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import math
import numpy as np
from scipy.stats import t as student, binomtest
from policy_value_isomorph.phase_iv_game import OUT as TTT
from policy_value_isomorph.phase_iv_statistics import holm
from policy_value_isomorph.phase_v import OUT, read, save
from policy_value_isomorph.phase_v_evaluate import temperature_label

GAMES = ("tictactoe", "connect4", "reversi4")
METRICS = (
    "agreement",
    "visit_agreement",
    "tv",
    "visit_tv",
    "win",
    "draw",
    "loss",
    "points",
    "source_calls",
    "simulated_moves",
    "distinct_in_eight",
    "entropy",
    "probability_missing",
    "probability_outside",
    "centered_residual_rms",
)


def reduce_block(job):
    game, b = job
    folder = OUT / game / f"block_{b:02}"
    groups = defaultdict(list)

    def add(mix, scope, name, metric, v):
        if v is not None and np.isfinite(v):
            groups[(mix, scope, name, metric)].append(v)

    def mainrow(mix, r, keys):
        for label in keys:
            for m in METRICS:
                add(mix, "performance", label, m, r.get(m))
            for g, row in enumerate(r["strata"]):
                for m in ("n", "agreement", "tv"):
                    add(mix, "positions_" + str(g), label, m, row[m])
            if r.get("move_pairs"):
                for n, v in zip(("both", "direct_only", "VNM_only", "neither"), r["move_pairs"]):
                    add(mix, "paired_decisions", label, n, v / r["boards"])

    if game == "tictactoe":
        for r in read(folder / "metrics.jsonl.gz"):
            lab = r["method"] + "_" + temperature_label(r["temperature"])
            mainrow(r["mixed"], r, [lab] + ([r["method"] + "_Tselected"] if r["selected"] else []))
        for group in ("seats", "duels"):
            for r in read(folder / (group + ".jsonl.gz")):
                label = r["method"] + "_" + temperature_label(r["temperature"])
                scope = (
                    group + "_" + str(r["seat"]) + ("_" + str(r["opponent"]) if group == "seats" else "")
                )
                for m in ("win", "draw", "loss", "points"):
                    add(r["mixed"], scope, label, m, r.get(m, r["win"] - r["loss"]))
        for r in read(TTT / f"block_{b:02}" / "baseline.jsonl.gz"):
            if r["method"] in ("source", "imitation"):
                name = (
                    "source_max"
                    if r["method"] == "source"
                    else ("imitator_max" if r["source"] == "fixed" else "imitator_T1")
                )
                for m in METRICS:
                    add(int(r["source"] == "randomized"), "performance", name, m, r.get(m))
        old = TTT / f"block_{b:02}"
        for r in read(old / "returns.jsonl.gz"):
            if r["method"] in ("points", "outcomes"):
                name = f"{r['network']}_{r['method']}_max"
                for m in METRICS:
                    add(int(r["source"] == "randomized"), "performance", name, m, r.get(m))
        for r in read(old / "soft.jsonl.gz"):
            name = r["method"]
            if name.startswith(("Q_direct", "V_direct", "Q_VNM", "V_VNM")):
                continue
            labels = [
                name if r["temperature"] is None else name + "_" + temperature_label(r["temperature"])
            ]
            if r["selected"]:
                labels.append(name + "_Tselected")
            for label in labels:
                for m in METRICS:
                    add(int(r["source"] == "randomized"), "performance", label, m, r.get(m))
        for r in read(old / "interventions.jsonl.gz"):
            name = r["method"]
            mix = int(r["source"] == "randomized")
            if name == "imitation":
                name = "imitator"
            if name == "original_scores":
                name = "source"
            if "_T" not in name:
                name += "_max"
            add(mix, r["event"], name, "tv", r["tv"])
            add(mix, r["event"], name, "agreement", r["independent_match"])
            if r["event"] in ("remove_favorite", "remove_other"):
                add(mix, "removed_move", name, "tv", r["tv"])
    else:
        for k in range(2):
            for mix in (0, 1):
                for f in range(2):
                    tag = f"{k}_{mix}_{f}"
                    with np.load(folder / f"evaluation_{tag}.npz") as cases:
                        for r in read(folder / f"metrics_{tag}.jsonl.gz"):
                            key = r["method"] + "_" + temperature_label(r["temperature"])
                            labels = [key] + ([r["method"] + "_Tselected"] if r["selected"] else [])
                            mainrow(mix, r, labels)
                            games = cases["games_" + key][:, 0].reshape(4, 8, 2)
                            for g in range(4):
                                for seat in (0, 1):
                                    z = games[g, :, seat]
                                    for metric, v in (
                                        ("win", (z == 1).mean()),
                                        ("draw", (z == 0).mean()),
                                        ("loss", (z == -1).mean()),
                                        ("points", z.mean()),
                                    ):
                                        add(mix, f"seats_{seat}_{g}", key, metric, v)
                            if "duels_" + key in cases:
                                duel = cases["duels_" + key][:, 0].reshape(-1, 2)
                                for seat in (0, 1):
                                    z = duel[:, seat]
                                    for metric, v in (
                                        ("win", (z == 1).mean()),
                                        ("draw", (z == 0).mean()),
                                        ("loss", (z == -1).mean()),
                                        ("points", z.mean()),
                                    ):
                                        add(mix, f"duels_{seat}", key, metric, v)
                    for r in read(folder / f"changes_{tag}.jsonl.gz"):
                        for m in ("tv", "agreement"):
                            add(mix, r["event"], r["method"], m, r[m])
                        if r["event"] in ("remove_favorite", "remove_other"):
                            add(mix, "removed_move", r["method"], "tv", r["tv"])
                    for r in read(folder / f"diagnostics_{tag}.jsonl.gz"):
                        scope = r["experiment"]
                        name = (
                            str(r["model"]) + "_" + r["comparison"]
                            if scope == "stability"
                            else r["strategy"] + "_" + str(r["budget"])
                            if scope == "queries"
                            else r["population"]
                        )
                        for m, v in r.items():
                            if isinstance(v, (float, int)) and m not in ("model", "budget"):
                                add(mix, scope, name, m, v)
                    for r in read(folder / f"training_{tag}.jsonl.gz"):
                        for m, v in r["selected"].items():
                            add(mix, "training", str(r["model"]), m, v)
                    for r in read(folder / f"curves_{tag}.jsonl.gz"):
                        for m in ("agreement", "points", "held_squared", "training", "validation"):
                            add(mix, "curve", f"{r['model']}_step{r['step']}", m, r[m])
                    for r in read(folder / f"lotteries_{tag}.jsonl.gz"):
                        add(mix, "changed_continuation", r["method"], "points_forgone", r["forgone"])
                    for r in read(folder / f"diversity_{tag}.jsonl.gz"):
                        name = r["method"] + "_" + temperature_label(r["temperature"])
                        for m in ("distinct_boards", "decisions", "validation_tv"):
                            add(mix, "eight_games", name, m, r[m])
    records = [
        dict(
            mixed=int(mix),
            scope=scope,
            method=method,
            metric=m,
            value=float(np.mean(v)),
            observations=len(v),
        )
        for (mix, scope, method, m), v in groups.items()
    ]
    save(folder / "block_means.jsonl.gz", records)
    return job


def ref_name(scope, name):
    if scope == "stability" and name.endswith("_other_architecture"):
        return name.split("_")[0] + "_new_initialization"
    if scope == "training" and name == "2":
        return "1"
    if scope == "coverage" and name != "all_held":
        return "all_held"
    if "_VNM_" in name and name.startswith(("V_", "Q_")):
        return name.split("_")[0] + "_direct_" + name.rsplit("_", 1)[1]
    if scope == "queries":
        return "random_" + name.split("_")[-1]
    if scope == "curve":
        return name.split("_")[0] + "_step4"
    if scope in ("remove_favorite", "remove_other", "removed_move", "swap_first_two"):
        return "Q_direct_max"
    if scope == "changed_continuation":
        return "Q_direct"
    return None


def summary(x):
    n = len(x)
    sd = float(np.std(x, ddof=1)) if n > 1 else None
    h = CRIT[n] * sd / np.sqrt(n) if n > 1 else None
    avg = float(np.mean(x))
    return dict(mean=avg, sd=sd, n=n, ci95=[avg - h, avg + h] if h is not None else None)


CRIT = {n: float(student.ppf(0.975, n - 1)) for n in range(2, 81)}


def paired(x):
    r = summary(x)
    se = r["sd"] / np.sqrt(len(x)) if len(x) > 1 else 0
    r.update(
        p=float(2 * student.sf(abs(r["mean"]) / se, len(x) - 1)) if se else None,
        holm_p=None,
        positive=int((x > 0).sum()),
        equal=int((x == 0).sum()),
        negative=int((x < 0).sum()),
    )
    return r


def analyze():
    cells = defaultdict(dict)
    for game in GAMES:
        for folder in sorted((OUT / game).glob("block_*")):
            b = int(folder.name[-2:])
            for r in read(folder / "block_means.jsonl.gz"):
                cells[(game, r["mixed"], r["scope"], r["method"], r["metric"])][b] = r["value"]
    # Applying a temperature selected on other originals gives an additional transfer comparison.
    for game in GAMES[1:]:
        for mix in (0, 1):
            for net in ("V", "Q"):
                for rule in (
                    "direct",
                    *[
                        f"VNM_{u}_{who}"
                        for u in ("terminal", "sum", "decay")
                        for who in ("copies", "source")
                    ],
                ):
                    prefix = net + "_" + rule + "_"
                    choices = [prefix + temperature_label(temp) for temp in (0, 0.05, 0.25, 1)]
                    val = [cells[game, mix, "eight_games", n, "validation_tv"] for n in choices]
                    for b in range(24):
                        selected = min(
                            range(4), key=lambda i: np.mean([x for j, x in val[i].items() if j != b])
                        )
                        for scope, metrics in (
                            ("performance", METRICS),
                            ("eight_games", ("distinct_boards", "decisions")),
                        ):
                            for m in metrics:
                                source = cells.get((game, mix, scope, choices[selected], m))
                                if source and b in source:
                                    cells[game, mix, scope, prefix + "Ttransfer", m][b] = source[b]
    tables = []
    differences = {}
    primary = []
    secondary = []

    def is_primary(k):
        return (
            k[1] == 0
            and k[2] == "performance"
            and k[3] in ("V_VNM_terminal_copies_max", "Q_VNM_terminal_copies_max")
            and k[4] in ("agreement", "win", "points")
        )

    rng = np.random.default_rng(810082)
    samples = {}
    for game in GAMES:
        n = 32 if game == "tictactoe" else 24
        samples[game] = (
            np.column_stack([2 * rng.integers(0, 12, size=(5000, 12)) + j for j in (0, 1)])
            if game == "connect4"
            else rng.integers(0, n, size=(5000, n))
        )
    for key, values in cells.items():
        game, mix, scope, name, m = key
        ids = sorted(values)
        x = np.array([values[i] for i in ids])
        row = dict(
            game=game,
            mixed=mix,
            scope=scope,
            method=name,
            metric=m,
            **summary(x),
            block_ids=ids,
            block_means=x.tolist(),
        )
        ref = ref_name(scope, name)
        base = cells.get((game, mix, scope, ref, m))
        if ref != name and base:
            common = sorted(values.keys() & base.keys())
            if common:
                d = np.array([values[i] - base[i] for i in common])
                effect = paired(d)
                row.update(reference=ref, paired=effect, paired_ids=common, block_differences=d.tolist())
                if len(d) == (32 if game == "tictactoe" else 24) and scope == "performance":
                    differences[key] = d
                    effect["bootstrap_ci95"] = np.quantile(
                        d[samples[game]].mean(1), [0.025, 0.975]
                    ).tolist()
                (primary if is_primary(key) else secondary).append(effect)
                if is_primary(key):
                    effect["positive_fraction_ci95"] = list(
                        binomtest(effect["positive"], len(d)).proportion_ci(0.95)
                    )
                    effect["recurrence_ci95"] = {
                        direction: list(binomtest(effect[direction], len(d)).proportion_ci(0.95))
                        for direction in ("positive", "equal", "negative")
                    }
        tables.append(row)
    combined = []
    for key, first in differences.items():
        if key[0] != GAMES[0]:
            continue
        ks = [(g, *key[1:]) for g in GAMES]
        if not all(k in differences for k in ks):
            continue
        groups = [differences[k] for k in ks]
        mean = float(np.mean([x.mean() for x in groups]))
        var = np.array([x.var(ddof=1) / len(x) / 9 for x in groups])
        se = math.sqrt(sum(var))
        df = sum(var) ** 2 / sum(v**2 / (len(x) - 1) for x, v in zip(groups, var)) if se else None
        h = float(student.ppf(0.975, df) * se) if se else 0.0
        boot = sum(x[samples[g]].mean(1) for g, x in zip(GAMES, groups)) / 3
        effect = dict(
            mean=mean,
            sd=math.sqrt(sum(x.var(ddof=1) for x in groups)) / 3,
            ci95=[mean - h, mean + h],
            bootstrap_ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
            p=float(2 * student.sf(abs(mean) / se, df)) if se else None,
            holm_p=None,
            source_counts=[len(x) for x in groups],
            positive_shares=[float((x > 0).mean()) for x in groups],
        )
        reference = ref_name(key[2], key[3])

        def absolute(method):
            values = [np.array(list(cells[(g, key[1], key[2], method, key[4])].values())) for g in GAMES]
            return dict(
                mean=float(np.mean([x.mean() for x in values])),
                sd=float(np.sqrt(sum(x.var(ddof=1) for x in values)) / 3),
            )

        effect["recurrence"] = {}
        for label in ("higher", "equal", "lower"):
            indicators = [
                (x > 0 if label == "higher" else x < 0 if label == "lower" else x == 0) for x in groups
            ]
            draws = sum(x[samples[g]].mean(1) for g, x in zip(GAMES, indicators)) / 3
            effect["recurrence"][label] = dict(
                fraction=float(np.mean([x.mean() for x in indicators])),
                bootstrap_ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
            )
        combined.append(
            dict(
                mixed=key[1],
                method=key[3],
                reference=reference,
                metric=key[4],
                direct=absolute(reference),
                VNM=absolute(key[3]),
                paired=effect,
            )
        )
        (primary if is_primary(key) else secondary).append(effect)
    assert len(primary) == 24
    holm(primary)
    holm(secondary)
    save(OUT / "summary.jsonl.gz", tables)
    save(OUT / "three_games.jsonl.gz", combined)
    planned = [
        r for r in tables if is_primary((r["game"], r["mixed"], r["scope"], r["method"], r["metric"]))
    ]
    planned.extend(
        dict(game="equal_game_average", **r)
        for r in combined
        if is_primary(("", r["mixed"], "performance", r["method"], r["metric"]))
    )
    (OUT / "primary.json").write_text(json.dumps(planned, indent=2) + "\n")
    # Rank independently deployable controls within each game, separating additional original-score access.
    rankings = []
    for game in GAMES:
        for mix in (0, 1):
            for scope, metric, goal in (
                ("performance", "tv" if mix else "agreement", "copy"),
                ("performance", "points", "game"),
                ("remove_favorite", "tv", "remove_favorite"),
                ("remove_other", "tv", "remove_other"),
                ("removed_move", "tv", "removed_move"),
            ):
                options = [
                    r
                    for r in tables
                    if r["game"] == game
                    and r["mixed"] == mix
                    and r["scope"] == scope
                    and r["metric"] == metric
                    and not any(
                        w in r["method"]
                        for w in ("source", "distilled", "masked", "oracle", "original", "exact")
                    )
                    and not (
                        game == "tictactoe"
                        and r["method"].startswith(("compact_VNM", "supplied_VNM", "random_utility_VNM"))
                        and "copies" not in r["method"]
                    )
                ]
                if not options:
                    continue
                options.sort(key=lambda r: r["mean"], reverse=metric in ("agreement", "points"))
                lead = options[0]
                arr = np.array(lead["block_means"])
                out = []
                for r in options:
                    if r["block_ids"] != lead["block_ids"]:
                        continue
                    diff = paired(np.array(r["block_means"]) - arr)
                    out.append(dict(method=r["method"], mean=r["mean"], sd=r["sd"], diff_from_leader=diff))
                holm([r["diff_from_leader"] for r in out[1:]])
                rankings.append(dict(game=game, mixed=mix, goal=goal, methods=out))
    (OUT / "rankings.json").write_text(json.dumps(rankings, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    with ProcessPoolExecutor(6) as pool:
        for done in pool.map(
            reduce_block, [(g, b) for g in GAMES for b in range(32 if g == "tictactoe" else 24)]
        ):
            print(*done, flush=True)
    analyze()
