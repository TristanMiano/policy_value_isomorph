"""Summarize the perfect-training-agreement control separately from held-out results."""

from collections import defaultdict
import numpy as np
from policy_value_isomorph.phase_iv_statistics import compare, describe, holm
from policy_value_isomorph.phase_v import read, save
from diagnose_agreement import DEST, GAMES


def run():
    result = []
    contrasts = []
    for game in GAMES:
        n = 32 if game == "tictactoe" else 24
        values = defaultdict(list)
        for b in range(n):
            repeats = defaultdict(list)
            folder = DEST / game / f"block_{b:02}"
            for r in read(folder / "training_control.jsonl.gz") + read(folder / "training_sums.jsonl.gz"):
                for m, v in r.items():
                    if m in ("split", "fit", "method", "rule"):
                        continue
                    repeats[r["method"], r["rule"], m].append(v)
            for key, group in repeats.items():
                values[key].append(float(np.mean(group)))
        rng = np.random.default_rng(23112026)
        indices = (
            np.column_stack([2 * rng.integers(12, size=(5000, 12)) + j for j in (0, 1)])
            if game == "connect4"
            else rng.integers(n, size=(5000, n))
        )
        for (fit, rule, m), arr in values.items():
            result.append(dict(game=game, fit=fit, rule=rule, metric=m, **describe(arr), blocks=arr))
            if rule == "direct":
                continue
            comparisons = [(fit, "direct")]
            if fit == "Q_paths" and game != "tictactoe":
                comparisons.append(("Q_observed", rule))
            for other, ref in comparisons:
                if (
                    m not in ("match", "win", "draw", "loss", "points", "visit_match")
                    or (other, ref, m) not in values
                ):
                    continue
                x = np.array(arr) - values[other, ref, m]
                contrasts.append(
                    dict(
                        game=game,
                        fit=fit,
                        rule=rule,
                        reference_fit=other,
                        reference=ref,
                        metric=m,
                        **compare(x),
                        bootstrap_ci95=np.quantile(x[indices].mean(1), [0.025, 0.975]).tolist(),
                        block_differences=x.tolist(),
                    )
                )
    holm(contrasts)
    save(DEST / "training_results.jsonl.gz", result)
    save(DEST / "training_contrasts.jsonl.gz", contrasts)
    lookup = {(r["game"], r["fit"], r["rule"], r["metric"]): r for r in result}
    diff = {
        (r["game"], r["fit"], r["rule"], r["reference_fit"], r["reference"], r["metric"]): r
        for r in contrasts
    }

    def fmt(r, scale=100):
        return f"{r['mean'] * scale:.2f} ± {r['sd'] * scale:.2f}"

    def p(val):
        return "exact equality" if val is None else f"{val:.3g}"

    labels = {
        "Q_observed": "Q, original visits",
        "Q_paths": "Q, all alternative paths fitted",
        "V_pooled": "V, all observed parents",
        "V_disjoint": "V, disjoint successors",
    }
    text = [
        "# Changing selection on the fitting positions",
        "",
        "The original is the frozen neural player's favorite legal move. Q scores a board and move. V scores only the board resulting from the move. Direct chooses the largest immediate score. Von Neumann–Morgenstern (VNM) selection compares the score of complete games from each candidate, with both seats continuing by fitted direct play. Terminal VNM counts only the final move’s frequency, signed for its mover. Centered VNM sums the played score minus the best available score at each turn, signed for the mover. Each position evaluated here contributed its actual original move to training. All original visits in the collection were pooled across the old split labels. These rows measure in-sample behavior.",
        "",
        "For Q, original visits fit the observed board-and-move frequencies. In the all-paths fit we also observe both copies of the original playing out every initial alternative before freezing the fit. A forced first alternative contributes zero source observations. On every following freely chosen move its selected candidate gets 1 and the others get 0. Tic-tac-toe already includes every decision. For V, multiple incoming candidates can share a successor. V pooled assigns each successor its mean training target. V disjoint keeps the first board in fixed order whose successor inputs are all unused, using only legal transitions to choose the set. This directly fits the visit fractions on every selected board. Unobserved inputs use independent random background scores. A covered board has every legal alternative and all subsequent direct decisions scored entirely with fitted inputs.",
        "",
        "Values are means ± sample standard deviations across independently frozen original-player blocks. Average the two collections and two initializations first. The fixed tic-tac-toe and Reversi choices can coincide between repetitions. Differences are VNM minus direct on the same frozen fit, boards and paired games. Rates are percentages and their differences are percentage points. The 95% intervals and two-sided p-values refer to the zero-mean paired-difference model. Holm adjustment covers this training-control family. A dash for a p-value denotes a zero standard error. Games use the Phase V seat-balanced opponent panel. [Full diagnosis](../../AGREEMENT_DIAGNOSIS.md)",
        "",
        "| Game | Originals | Training population | Fitted roots evaluated | All alternatives covered (%) | Direct agreement (%) | Terminal VNM agreement (%) | Change ± spread [95% interval] | Raw / adjusted p | Centered VNM agreement (%) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines = []
    for g, name in zip(GAMES, ("Tic-tac-toe", "Connect Four", "4×4 Reversi")):
        for fit, label in labels.items():

            def item(rule, m):
                return lookup[g, fit, rule, m]

            if (g, fit, "direct", "match") not in lookup:
                continue
            d = diff[g, fit, "terminal", fit, "direct", "match"]
            n = d["n"]
            text.append(
                f"| {name} | {n} | {label} | {fmt(item('direct', 'boards'), 1)} | {fmt(item('direct', 'all_paths_covered'))} | {fmt(item('direct', 'match'))} | {fmt(item('terminal', 'match'))} | {fmt(d)} [{100 * d['ci95'][0]:.2f}, {100 * d['ci95'][1]:.2f}] | {p(d['p'])} / {p(d['holm_p'])} | {fmt(item('centered', 'match'))} |"
            )
            for rule, short in (
                ("direct", "Direct"),
                ("terminal", "Terminal VNM"),
                ("sum", "All signed moves VNM"),
                ("decay", "Discounted signed moves VNM"),
                ("centered", "Centered VNM"),
            ):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            name,
                            str(n),
                            label,
                            short,
                            *[
                                fmt(item(rule, m), 1 if m == "points" else 100)
                                for m in ("win", "draw", "loss", "points")
                            ],
                        ]
                    )
                    + " |"
                )
    text += [
        "",
        "## Which whole-game utility preserves the initial fit?",
        "",
        "Sum counts every signed move frequency. Discounted sum gives successive moves weights 1, 0.9, 0.81 and so on. All columns use the same frozen fit and continuations.",
        "",
        "| Game | Originals | Training population | Direct agreement (%) | Final-move VNM (%) | All moves VNM (%) | Discounted VNM (%) | Centered VNM (%) |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for g, name in zip(GAMES, ("Tic-tac-toe", "Connect Four", "4×4 Reversi")):
        for fit, label in labels.items():
            if (g, fit, "direct", "match") not in lookup:
                continue
            scores = [
                fmt(lookup[g, fit, r, "match"]) for r in ("direct", "terminal", "sum", "decay", "centered")
            ]
            text.append(
                "| "
                + " | ".join([name, str(lookup[g, fit, "direct", "match"]["n"]), label, *scores])
                + " |"
            )
    text += [
        "",
        "## Games played from the panel openings",
        "",
        "| Game | Originals | Fit | Rule | Wins (%) | Draws (%) | Losses (%) | Points/game |",
        "|---|---:|---|---|---:|---:|---:|---:|",
        *lines,
        "",
        "Per-fit board choices, alternative-path coverage, counts and means are retained in the individual source directories. [All estimates](training_results.jsonl.gz) · [Paired changes, source-level differences, resampling intervals and raw and adjusted p-values](training_contrasts.jsonl.gz)",
    ]
    (DEST / "training_tables.md").write_text("\n".join(text) + "\n", encoding="utf8")


if __name__ == "__main__":
    run()
