"""Render reproducible comparison tables from the source-level measurements."""

import json
import numpy as np
from policy_value_isomorph.phase_v import OUT, read, save
from policy_value_isomorph.phase_iv_statistics import holm
from report_phase_v import paired

GAMES = {
    "tictactoe": "Tic-tac-toe",
    "connect4": "Connect Four",
    "reversi4": "4×4 Reversi",
    "equal_game_average": "Three games, equal weight",
}
METRICS = {
    "agreement": "Same move (%)",
    "win": "Wins (%)",
    "points": "Points/game",
    "tv": "Probability error (%)",
}


def pair(x, scale):
    digits = 2 if scale == 100 else 3
    return f"{x['mean'] * scale:.{digits}f} ± {x['sd'] * scale:.{digits}f}"


def interval(x, scale):
    digits = 2 if scale == 100 else 3
    return "[" + ", ".join(f"{v * scale:.{digits}f}" for v in x) + "]"


def probability(x):
    return "—" if x is None else f"{x:.3g}"


def cohort(g):
    return "32/24/24" if g == "equal_game_average" else 32 if g == "tictactoe" else 24


def main():
    rows = read(OUT / "summary.jsonl.gz")
    lookup = {(r["game"], r["mixed"], r["scope"], r["method"], r["metric"]): r for r in rows}
    primary = json.loads((OUT / "primary.json").read_text())
    out = []

    def row(*cells):
        out.append("| " + " | ".join(map(str, cells)) + " |")

    def head(*cols):
        out.append("")
        row(*cols)
        row(*(["---"] * len(cols)))

    def section(title, definition):
        out.extend(["", "## " + title, "", definition])

    out.extend(
        [
            "# Three-game measurements",
            "",
            "The **original** chooses the legal move with the highest output from the frozen player. **V** estimates the frequency of a move from its resulting board. **Q** also sees the preceding board and candidate move. **Direct** chooses the largest immediate fitted frequency. **Von Neumann–Morgenstern (VNM) selection** compares the expected numerical scores of complete games, viewed from the player choosing now. In the primary comparison each candidate is played out by two direct copies of the fitted network and only the final move’s fitted frequency counts, positive for the first player and negative for the second. The lowest move number breaks exact ties. No original is consulted during either primary selection.",
            "",
            "Each value is a mean ± sample standard deviation across independently trained originals, after averaging both board partitions and both fresh fits in each partition. The three-game row gives each named game one third of the weight. Its spread describes the average of one independent original per game. Agreement is the fraction of withheld positions with the same chosen move. Wins, draws and losses are percentages of games against the specified opponents, averaged equally over both seats and the four opponent groups. Points are +1 for a win, 0 for a draw and −1 for a loss. Rate differences are in percentage points.",
        ]
    )
    section(
        "Planned paired comparisons",
        "Each difference subtracts direct selection on the same fit and evaluation cases. A 95% confidence interval describes uncertainty in the mean. The resampling interval repeats the calculation on 5,000 samples of entire originals, keeping the two Connect Four target groups in their original proportions. The raw p-value is the two-sided probability of a result at least this extreme under the model of zero mean difference. Holm adjustment accounts for examining 24 planned differences together. Intervals are individual. Within a game, the last column counts originals with a higher, equal or lower VNM result. For the three-game mean it shows the equally game-weighted percentages in those categories. Full precision and intervals on recurrence are in the saved results.",
    )
    head(
        "Game",
        "Originals",
        "Network",
        "Measure",
        "Direct",
        "VNM",
        "VNM − direct",
        "95% interval",
        "Resampling 95%",
        "Raw p",
        "Holm p",
        "Higher / equal / lower",
    )
    for r in primary:
        g, m, n = r["game"], r["metric"], r["method"]
        scale = 1 if m == "points" else 100
        d = r["paired"]
        combined = g == "equal_game_average"
        direct = r["direct"] if combined else lookup[g, 0, "performance", r["reference"], m]
        counts = (
            " / ".join(f"{d['recurrence'][v]['fraction'] * 100:.1f}%" for v in ("higher", "equal", "lower"))
            if combined
            else " / ".join(str(d[v]) for v in ("positive", "equal", "negative"))
        )
        row(
            GAMES[g],
            cohort(g),
            n[0],
            METRICS[m],
            pair(direct, scale),
            pair(r["VNM"] if combined else r, scale),
            pair(d, scale),
            interval(d["ci95"], scale),
            interval(d["bootstrap_ci95"], scale),
            probability(d["p"]),
            probability(d["holm_p"]),
            counts,
        )
    section(
        "Absolute outcomes and alternative whole-game scores",
        "Terminal scores only the final move. Sum counts every move’s signed frequency. Discounted sum also multiplies each successive move’s weight by 0.9. Both players use fitted direct copies throughout these rollouts. The original row is the frozen player itself. Each rule takes its best-scoring move. Game points and agreement have different units.",
    )
    head("Game", "Originals", "Rule", "Agreement (%)", "Wins (%)", "Draws (%)", "Losses (%)", "Points/game")
    for g in list(GAMES)[:3]:
        for n, name in [("source_max", "Original")] + [
            (f"{family}_{mode}_max", family + " " + label)
            for family in ("V", "Q")
            for mode, label in (
                ("direct", "direct"),
                ("VNM_terminal_copies", "VNM terminal"),
                ("VNM_sum_copies", "VNM sum"),
                ("VNM_decay_copies", "VNM discounted sum"),
            )
        ]:
            row(
                GAMES[g],
                cohort(g),
                name,
                *[
                    pair(lookup[g, 0, "performance", n, m], 1 if m == "points" else 100)
                    for m in ("agreement", "win", "draw", "loss", "points")
                ],
            )
    section(
        "Direct opponents and changed menus",
        "For head-to-head games the VNM player uses final-move utility and fitted direct continuations, while its actual opponent always plays direct on the same weights. Each seat has equal weight. In the second table a probability error is the fraction of move probability that must be reassigned to reproduce the original answer. Smaller values indicate closer predictions. These are positions offering at least two moves before the favorite was removed. The imitator maximizes the fitted probability of observed moves. Its predictions, both direct frequency rules, and both VNM rules use observed play for training.",
    )
    head(
        "Game",
        "Originals",
        "VNM network",
        "VNM wins (%)",
        "Draws (%)",
        "Direct wins (%)",
        "VNM wins − direct wins, points [95%]",
        "Raw / Holm p",
        "Played VNM wins / draws / direct wins",
    )
    duels = []
    for g in list(GAMES)[:3]:
        for family in ("V", "Q"):
            name = family + "_VNM_terminal_copies_max"
            result = []
            estimates = []
            for m in ("win", "draw", "loss"):
                z = np.mean(
                    [lookup[g, 0, "duels_" + str(i), name, m]["block_means"] for i in range(2)], axis=0
                )
                estimates.append(z)
                result.append(pair(dict(mean=z.mean(), sd=z.std(ddof=1)), 100))
            diff = paired(estimates[0] - estimates[2])
            counts = (
                "Exact game probabilities"
                if g == "tictactoe"
                else " / ".join(str(round(x.mean() * 6144)) for x in estimates)
            )
            duels.append((g, family, result, diff, counts))
    holm([d for _, _, _, d, _ in duels])
    for g, family, result, diff, counts in duels:
        row(
            GAMES[g],
            cohort(g),
            family,
            *result,
            pair(diff, 100) + " " + interval(diff["ci95"], 100),
            probability(diff["p"]) + " / " + probability(diff["holm_p"]),
            counts,
        )
    out.extend(
        [
            "",
            "For the head-to-head difference the reference is equal win rates for both players. Holm adjustment covers the six head-to-head comparisons. The tic-tac-toe results enumerate possible games from the opening. Each larger-game row contains 6,144 actual games across 24 originals.",
            "",
        ]
    )
    head(
        "Game",
        "Originals",
        "Rule",
        "Favorite removed, error (%)",
        "Another move removed, error (%)",
        "Either removal, error (%)",
    )
    for g in list(GAMES)[:3]:
        for n in (
            "imitator_max",
            "V_direct_max",
            "Q_direct_max",
            "V_VNM_terminal_copies_max",
            "Q_VNM_terminal_copies_max",
        ):
            row(
                GAMES[g],
                cohort(g),
                n.replace("_max", "").replace("_", " "),
                *[
                    pair(lookup[g, 0, s, n, "tv"], 100)
                    for s in ("remove_favorite", "remove_other", "removed_move")
                ],
            )
    section(
        "Where the choices change",
        "For fixed originals and final-move VNM, the first table groups withheld boards by which rules reproduce the original choice. These are raw fractions in the sampled cases, counting each position once. Reversi's main estimate additionally restores the population weights. See the per-fit board records for exact counts and denominators. The second table gives the percentage of independent originals with a higher, equal or lower VNM result and individual 95% binomial intervals on those shares.",
    )
    head("Game", "Originals", "Network", "Both match (%)", "Only direct (%)", "Only VNM (%)", "Neither (%)")
    for g in list(GAMES)[:3]:
        for family in ("V", "Q"):
            n = family + "_VNM_terminal_copies_max"
            row(
                GAMES[g],
                cohort(g),
                family,
                *[
                    pair(lookup[g, 0, "paired_decisions", n, m], 100)
                    for m in ("both", "direct_only", "VNM_only", "neither")
                ],
            )
    head(
        "Game", "Originals", "Network", "Measure", "Higher (%) [95%]", "Equal (%) [95%]", "Lower (%) [95%]"
    )
    for r in primary:
        g = r["game"]
        if g == "equal_game_average":
            continue
        d = r["paired"]
        row(
            GAMES[g],
            d["n"],
            r["method"][0],
            METRICS[r["metric"]],
            *[
                f"{100 * d[v] / d['n']:.1f} " + interval(d["recurrence_ci95"][v], 100)
                for v in ("positive", "equal", "negative")
            ],
        )
    section(
        "Outcomes by opponent and seat",
        "Each cell lists direct, final-move VNM, and their paired change in that order. Each value gives the mean ± spread across originals. New-game rates use eight openings per fit, opponent group and seat. The group of untrained networks averages two opponents in the new games and five in tic-tac-toe. The third and fourth opponent groups use the rules in the study report. All adjusted p-values and paired intervals for these exploratory differences are available in the source-level results.",
    )
    head(
        "Game",
        "Originals",
        "Network",
        "Seat",
        "Opponent",
        "Wins (%), direct / VNM / change",
        "Draws (%), direct / VNM / change",
        "Losses (%), direct / VNM / change",
        "Points/game, direct / VNM / change",
    )
    for g in list(GAMES)[:3]:
        names = (
            ("Uniform", "Untrained networks", "Win/block/center/corner", "Optimal")
            if g == "tictactoe"
            else ("Uniform", "Untrained networks", "Positional", "Tactical or optimal")
        )
        for f in ("V", "Q"):
            for seat in (0, 1):
                for j, opp in enumerate(names):
                    vals = []
                    for m in ("win", "draw", "loss", "points"):
                        scope = f"seats_{seat}_{j}"
                        dr = lookup[g, 0, scope, f + "_direct_max", m]
                        vr = lookup[g, 0, scope, f + "_VNM_terminal_copies_max", m]
                        scale = 1 if m == "points" else 100
                        vals.append(" / ".join(pair(v, scale) for v in (dr, vr, vr["paired"])))
                    row(GAMES[g], cohort(g), f, "First" if seat == 0 else "Second", opp, *vals)
    section(
        "Fixed-player controls against direct Q",
        "Each entry gives the absolute mean ± spread. The paired differences below subtract Q direct on the identical cases. A move imitator learns the probability of observed actions. A cell frequency pools choices by action, a revealed ranking uses extra restricted-menu queries, and distillation fits the original numerical scores. Game-points and outcome-frequency networks learn from observed endings. Supplied-payoff VNM gives complete games their actual points, and fitted-payoff VNM learns the value of a draw. Both use two fitted direct copies wherever labeled. The source-continuation rows consult the original at future turns. A dash denotes an unavailable measurement. All intervals and p-values for these fixed-control contrasts are in controls.jsonl.gz; Holm adjustment covers this table's paired control contrasts.",
    )
    head(
        "Game",
        "Originals",
        "Method and training/deployment access",
        "Agreement (%)",
        "Wins (%)",
        "Draws (%)",
        "Losses (%)",
        "Points/game",
        "Error after removing favorite (%)",
        "Agreement change [95%]",
        "Wins change [95%]",
        "Removal error change [95%]",
    )
    controls = []
    for g in list(GAMES)[:3]:
        ttt = g == "tictactoe"
        choices = [
            ("Q_direct_max", "Q direct, visits"),
            ("imitator_max", "Move imitator, visits"),
            ("cell_frequency_max", "Pooled cell frequency, visits"),
            ("revealed_order_max", "Revealed order, extra menu queries"),
            ("distilled_logits_max", "Distillation, original scores during fitting"),
            ("Q_points_max" if ttt else "payoff_network_max", "Game-points network, observed endings"),
            (
                "Q_outcomes_max" if ttt else "outcome_network_max",
                "Outcome-frequency network, observed endings",
            ),
            (
                "supplied_VNM_copies_max" if ttt else "supplied_VNM_max",
                "Supplied-payoff VNM, fitted Q copies",
            ),
            ("compact_VNM_copies_max" if ttt else "compact_VNM_max", "Fitted-payoff VNM, fitted Q copies"),
            (
                "compact_VNM_max" if ttt else "compact_VNM_source_max",
                "Fitted-payoff VNM, original continuations",
            ),
        ]
        for n, label in choices:
            vals = []
            diffs = {}
            for scope, metric in [
                ("performance", m) for m in ("agreement", "win", "draw", "loss", "points")
            ] + [("remove_favorite", "tv")]:
                r = lookup.get((g, 0, scope, n, metric))
                if r is None:
                    vals.append("—")
                    continue
                ref = lookup[g, 0, scope, "Q_direct_max", metric]
                scale = 1 if metric == "points" else 100
                vals.append(pair(r, scale))
                if n != "Q_direct_max":
                    x = np.array(r["block_means"]) - ref["block_means"]
                    d = paired(x)
                    diffs[metric] = d
                    controls.append(
                        dict(
                            game=g,
                            method=n,
                            reference="Q_direct_max",
                            scope=scope,
                            metric=metric,
                            mean=r["mean"],
                            sd=r["sd"],
                            block_differences=x.tolist(),
                            paired=d,
                        )
                    )
            if vals[0] == "—":
                continue
            changes = [
                pair(diffs[m], 100) + " " + interval(diffs[m]["ci95"], 100) if m in diffs else "—"
                for m in ("agreement", "win", "tv")
            ]
            row(GAMES[g], cohort(g), label, *vals, *changes)
    for g in ("connect4", "reversi4"):
        name, ref = "distilled_logits_max", "imitator_max"
        r = lookup[g, 0, "remove_favorite", name, "tv"]
        x = np.array(r["block_means"]) - lookup[g, 0, "remove_favorite", ref, "tv"]["block_means"]
        controls.append(
            dict(
                game=g,
                method=name,
                reference=ref,
                scope="remove_favorite",
                metric="tv",
                mean=r["mean"],
                sd=r["sd"],
                block_differences=x.tolist(),
                paired=paired(x),
            )
        )
    holm([r["paired"] for r in controls])
    save(OUT / "controls.jsonl.gz", controls)
    section(
        "All common deterministic choices and randomized originals",
        "A randomized original takes its favorite with probability ¾ and otherwise samples uniformly. Independent-draw agreement is the probability that one draw from each player gives the same move. For a positive temperature T, each legal move gets sampling weight exp(score/T). At T = 0 exact best moves share probability equally. Here each method’s temperature for randomized originals is selected by its own validation boards. The direct comparator uses its own validation setting. Copies finish with fitted direct play. Original continuations consult the frozen player at subsequent turns. Each cell is VNM minus direct, mean ± spread with individual 95% interval, raw p and adjusted p. Adjustments use the separate exploratory family except where the fixed-choice comparison was designated primary. Differences for every individual temperature, game average, position type, seat and opponent are machine-readable.",
    )
    cross = read(OUT / "three_games.jsonl.gz")
    for mix in (0, 1):
        out.extend(["", "### " + ("Randomized originals" if mix else "Fixed-choice originals"), ""])
        metrics = ("tv", "agreement", "win", "points") if mix else ("agreement", "win", "points")
        head(
            "Game",
            "Originals",
            "Network",
            "VNM whole-game score",
            "Both seats continue with",
            *[METRICS[m] for m in metrics],
        )
        for g in GAMES:
            for family in ("V", "Q"):
                for utility in ("terminal", "sum", "decay"):
                    for who in ("copies", "source"):
                        name = f"{family}_VNM_{utility}_{who}_" + ("Tselected" if mix else "max")
                        results = []
                        for m in metrics:
                            if g == "equal_game_average":
                                rr = next(
                                    a
                                    for a in cross
                                    if (a["mixed"], a["method"], a["metric"]) == (mix, name, m)
                                )
                            else:
                                rr = lookup[g, mix, "performance", name, m]
                            d = rr["paired"]
                            scale = 1 if m == "points" else 100
                            results.append(
                                f"{pair(d, scale)} {interval(d['ci95'], scale)}, {probability(d['p'])} / {probability(d['holm_p'])}"
                            )
                        row(
                            GAMES[g],
                            cohort(g),
                            family,
                            {"terminal": "Final move", "sum": "Sum", "decay": "Discounted sum"}[utility],
                            "Original" if who == "source" else "Fitted copies",
                            *results,
                        )
    out.extend(
        [
            "",
            "[Report and interpretation](../../PHASE_V.md) · [Design](../../PHASE_V_PROTOCOL.md) · [File guide and denominators](README.md) · [All source-level results](summary.jsonl.gz) · [Equally weighted three-game comparisons](three_games.jsonl.gz) · [Method rankings and paired differences](rankings.json)",
        ]
    )
    (OUT / "tables.md").write_text("\n".join(out) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
