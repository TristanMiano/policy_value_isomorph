"""Render the visit-only study from archived per-original statistics and training measurements."""

import argparse
import gzip
import json
import statistics
from compare_visit_utilities import DEST, GAMES
from policy_value_isomorph.visit_selection import VisitSelection


def read(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        return json.load(stream)


def method(key):
    if key == "original":
        return "Frozen original"
    if key.startswith("reference/"):
        if key.endswith("/direct"):
            return "Direct " + key.split("/")[1]
        return (
            "VNM V, supplied outcomes and frequencies, guarded mixture"
            if "hybrid" in key
            else "VNM V, signed final frequency, assisted opponent mixture"
        )
    r = VisitSelection.parse(key)
    suffix = (
        ", constant scores and continuations"
        if r.flat_moves
        else (", constant scores" if r.flat_score else "")
    )
    return f"VNM {r.network}, {r.utility}, {r.continuation}, {r.paths} paths{suffix}"


def spread(row, scale=100, digits=2):
    return f"{row['mean'] * scale:.{digits}f} ± {row['sd'] * scale:.{digits}f}"


def contrast(row, scale=100, digits=2):
    if row["ci95"] is None:
        return f"{scale * row['mean']:.{digits}f} (one original)"
    low, high = row["ci95"]
    p = row["holm_p"]
    prob = "—" if p is None else f"{p:.3g}"
    return f"{row['mean'] * scale:+.{digits}f} ± {row['sd'] * scale:.{digits}f} [{low * scale:.{digits}f}, {high * scale:.{digits}f}], {prob}"


def section(lines, heading, columns):
    lines.extend(["", heading, "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)])


def table(stage):
    lock = json.loads((DEST / "selection.json").read_text())
    winner = lock["winner"]
    estimates = read(DEST / f"{stage}_estimates.json.gz")
    effects = read(DEST / f"{stage}_contrasts.json.gz")
    by_key = {(r["method"], r["game"], r["metric"]): r for r in estimates}
    paired = {(r["reference"], r["game"], r["metric"]): r for r in effects}
    order = sorted(
        {r["method"] for r in estimates}, key=lambda key: -by_key[key, "equal_games", "win"]["mean"]
    )
    lines = [
        "# " + stage.capitalize() + " results for visit-trained game scores",
        "",
        "The frozen original takes its highest-scoring legal move. A network scoring the resulting board (V) or the board and move (Q) estimates the fraction of observed visits taking a move. Direct selection takes the largest immediate fit. Von Neumann–Morgenstern (VNM) selection maximizes expected complete-game score, replanning on each actual turn. Ties prefer the lowest move index. 'copies' projects highest fitted choices for both seats. The rule 'uniform_reply' projects greedy own moves against uniform legal replies. The rule 'mixed_reply' mixes those two opponents equally once per path. The rule 'uniform_both' projects uniform legal moves for both seats. Each random component gets the listed number of full paths, with exact expectations in tic-tac-toe. The assisted comparison includes an opponent considering tactics and one using positional priorities. The supplied-outcome reference adds actual game results and projects guarded own play against four opponent types.",
        "",
        "The utility name specifies the score of a completed game from the current choosing seat. 'signed' counts own moves positively and opposing moves negatively, 'unsigned' counts both positively, and 'own' counts only own moves. 'last' counts the final fitted frequency. The rule 'sum' adds the whole path. The suffix 50, 75, 90 or 97 gives the discount per turn in percent, starting at weight one. 'sqrt_last' and 'square_last' apply the square root or square to the final frequency, signed by mover. 'last_plus_sum' adds the signed last frequency and 0.1 times the signed 0.9-discounted sum. A constant score uses 0.5 in the identical formula. Constant continuations also break all greedy choices by move index. Both constants induce the identical player when all continuing moves are uniformly random.",
        "",
        "Win, draw and loss percentages are seat-balanced against four equally weighted external opponent groups. Agreement is the weighted percentage of withheld boards on which the reconstruction chose the original's favorite. Exact-copy self-play uses two identical deployed players and counts both seats. Values give means ± sample standard deviations across originals after averaging their four fits. The equal-game spread describes the average of one independent original per named game. The selected rule is "
        + method(winner)
        + ".",
        "",
        "Independent original counts for tic-tac-toe, Connect Four and 4×4 Reversi are "
        + ("8, 8, 8." if stage == "discovery" else "24, 16, 16."),
        "For paired changes, brackets give individual 95% confidence intervals, followed by the two-sided p-value for a zero mean difference, Holm-adjusted across the table's win and agreement comparisons in all four populations. A dash means the across-original variance is exactly zero. Discovery contrasts are exploratory because the leader was chosen on those data. The compressed estimates and contrasts alongside this table contain all unrounded measurements, raw p-values, original-level differences, and intervals resampling the independent originals.",
    ]
    section(
        lines,
        "## External win rate (%)",
        ["Rule", "Tic-tac-toe", "Connect Four", "4×4 Reversi", "Equal games"],
    )
    for key in order:
        lines.append(
            "| "
            + " | ".join([method(key), *[spread(by_key[key, g, "win"]) for g in (*GAMES, "equal_games")]])
            + " |"
        )
    if stage == "confirmation":
        for g, title in zip(
            (*GAMES, "equal_games"), ("Tic-tac-toe", "Connect Four", "4×4 Reversi", "Equal games")
        ):
            section(
                lines,
                "## " + title,
                [
                    "Rule",
                    "External wins / draws / losses (%)",
                    "Move agreement (%)",
                    "Exact-copy wins / draws (%)",
                    "Selected minus row in wins, points [95%], p",
                    "Selected minus row in agreement, points [95%], p",
                ],
            )
            for key in order:

                def cell(metric):
                    return spread(by_key[key, g, metric])

                row = [
                    method(key),
                    " / ".join(map(cell, ("win", "draw", "loss"))),
                    cell("agreement"),
                    " / ".join(map(cell, ("self_win", "self_draw"))),
                ]
                row.extend(
                    [
                        contrast(paired[key, g, m]) if key != winner else "reference"
                        for m in ("win", "agreement")
                    ]
                )
                lines.append("| " + " | ".join(row) + " |")
        diag = read(DEST / "path_diagnostics.json.gz")
        section(
            lines,
            "## Scores on actual external games",
            [
                "Game (originals)",
                "Measure",
                "Outcome or association",
                "Mean ± spread [95%], adjusted p for differences and slopes",
                "Contributing fits",
            ],
        )
        lines[-3:-3] = [
            "",
            "Win minus loss subtracts each fitted player's mean utility on its lost games from the mean on its won games. Per turn is the within-fit least-squares change in utility per additional observed turn. 'unsigned_last' is the raw fitted frequency of the last played move. 'chooser_moved_last' is 1 if the evaluated seat made that move, otherwise 0. 'turns' counts the remaining turns including mandatory passes. Slopes and differences are averaged per original first. All diagnostic contrasts share a separate Holm adjustment. Missing outcomes are omitted from the corresponding fit-specific mean. Counts of observed wins, draws and losses and the fit-level values are in the path diagnostics file.",
        ]
        for r in diag["results"]:
            m, group = r["metric"].split("/")
            lines.append(f"| {r['game']} ({r['n'][0]}) | {m} | {group} | {contrast(r, 1, 4)} | {r['fits']} |")
        train = []
        for g in GAMES:
            for p in sorted((DEST / "confirmation" / g).glob("*.json.gz")):
                data = read(p)
                for net in ("V", "Q"):
                    logs = [r for r in data.get("training", []) if r["network"] == net]
                    if logs:
                        train.append(
                            dict(
                                game=g,
                                network=net,
                                original=data["block"],
                                **{
                                    k: statistics.mean([v[k] for v in logs])
                                    for k in ("training", "boards", "candidates", "observed")
                                },
                            )
                        )
        if train:
            section(
                lines,
                "## Network fitting",
                [
                    "Game (originals)",
                    "Network",
                    "Training mean squared frequency error",
                    "Fitting boards",
                    "Fitting legal candidates",
                    "Distinct observed decision boards",
                ],
            )
            for g in GAMES:
                for net in ("V", "Q"):
                    rows = [v for v in train if v["game"] == g and v["network"] == net]
                    if rows:

                        def fmt(k, d):
                            v = [r[k] for r in rows]
                            return f"{statistics.mean(v):.{d}f} ± {statistics.stdev(v):.{d}f}"

                        lines.append(
                            "| "
                            + " | ".join(
                                [
                                    f"{g} ({len(rows)})",
                                    net,
                                    fmt("training", 4),
                                    *[fmt(k, 1) for k in ("boards", "candidates", "observed")],
                                ]
                            )
                            + " |"
                        )
    (DEST / f"{stage}.md").write_text("\n".join(lines) + "\n", encoding="utf8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("discovery", "confirmation"))
    table(ap.parse_args().stage)
