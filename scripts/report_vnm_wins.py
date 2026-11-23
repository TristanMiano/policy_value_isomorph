"""Freeze the discovery choice and summarize paired external and mirror games."""

import argparse
from collections import defaultdict
import gzip
import json
import numpy as np
from scipy.stats import t
from compare_vnm_wins import DEST, GAMES
from policy_value_isomorph.phase_iv_statistics import holm
from policy_value_isomorph.win_selection import Selection, candidates


def load_blocks(stage):
    results = {}
    for game in GAMES:
        ids = range(8) if stage != "confirmation" else range(8, 32 if game == "tictactoe" else 24)
        by_method = defaultdict(list)
        for block in ids:
            path = DEST / stage / game / f"block_{block:02}.json.gz"
            with gzip.open(path, "rt") as stream:
                data = json.load(stream)
            assert (data["game"], data["block"]) == (game, block)
            runs = defaultdict(list)
            for r in data["rows"]:
                seats = np.asarray(r["external"])
                wdl = np.asarray(r["wdl"])
                if game == "tictactoe":
                    groups = np.stack(
                        [seats[:, 0, 0], seats[:, 3:, 0].mean(1), seats[:, 1, 0], seats[:, 2, 0]]
                    )
                    seat_wins = groups.mean(0)
                else:
                    wins = seats.reshape(4, 8, 2) == 1
                    groups = wins.mean(1)
                    seat_wins = groups.mean(0)
                assert np.allclose([wdl.sum(), wdl[0]], [1, seat_wins.mean()])
                metrics = dict(
                    win=wdl[0],
                    draw=wdl[1],
                    loss=wdl[2],
                    agreement=r["agreement"],
                    first_win=seat_wins[0],
                    second_win=seat_wins[1],
                )
                metrics.update(
                    zip(("uniform_win", "neural_win", "position_win", "tactic_win"), groups.mean(1))
                )
                if "self_wdl" in r:
                    same = np.array(r["self_first"])
                    first = same[0] if game == "tictactoe" else (same == 1).mean()
                    sw, sd, sl = r["self_wdl"]
                    assert np.allclose([sw, sl], [(1 - sd) / 2] * 2)
                    metrics.update(self_win=sw, self_draw=sd, self_loss=sl, self_first_win=first)
                runs[r["method"]].append((r["split"], r["fit"], metrics))
            for method, rows in runs.items():
                assert sorted((a, b) for a, b, _ in rows) == [(0, 0), (0, 1), (1, 0), (1, 1)]
                by_method[method].append({m: np.mean([r[m] for _, _, r in rows]) for m in rows[0][2]})
        results[game] = {
            method: {m: np.array([r[m] for r in rows]) for m in rows[0]} for method, rows in by_method.items()
        }
    return results


def estimate(arrays):
    n = np.asarray([len(a) for a in arrays])
    var = np.asarray([a.var(ddof=1) for a in arrays])
    mean = float(np.mean([a.mean() for a in arrays]))
    v = var / n
    se = np.sqrt(v.sum()) / len(arrays)
    df = float(v.sum() ** 2 / (v**2 / (n - 1)).sum()) if se else None
    half = float(t.ppf(0.975, df) * se) if se else 0.0
    return dict(
        mean=mean,
        sd=float(np.sqrt(var.sum()) / len(arrays)),
        n=n.tolist(),
        ci95=[mean - half, mean + half],
        p=float(2 * t.sf(abs(mean) / se, df)) if se else None,
    )


def run(stage):
    blocks = load_blocks(stage)
    methods = list(blocks[GAMES[0]])

    def score(m, game=None):
        gs = (game,) if game else GAMES
        return np.mean([blocks[g][m]["win"].mean() for g in gs])

    if stage == "discovery":
        rules = [r.name for r in candidates()]
        order = sorted(rules, key=score, reverse=True)
        winner = order[0]
        lock = dict(
            winner=winner,
            runner=order[1],
            learned=next(m for m in order if Selection.parse(m).prize in ("last", "sum", "discounted")),
            hybrid=next(m for m in order if Selection.parse(m).prize == "hybrid"),
            outcome=next(m for m in order if Selection.parse(m).prize in ("win", "points", "quarter_draw")),
            game_winners={g: max(rules, key=lambda m: score(m, g)) for g in GAMES},
            rank=order,
        )
        lock_path = DEST / "selection.json"
        body = json.dumps(lock, indent=2) + "\n"
        if lock_path.exists() and lock_path.read_text() != body:
            raise ValueError("discovery lock disagrees with archived results")
        lock_path.write_text(body)
    elif stage == "tuning":
        lock = json.loads((DEST / "selection.json").read_text())
        lock["grid_winner"] = lock["winner"]
        lock["tuning_rank"] = sorted(methods, key=score, reverse=True)
        lock["winner"] = lock["tuning_rank"][0]
        body = json.dumps(lock, indent=2) + "\n"
        lock_path = DEST / "standard.json"
        if lock_path.exists() and lock_path.read_text() != body:
            raise ValueError("tuning lock disagrees with archived results")
        lock_path.write_text(body)
        winner = lock["winner"]
    else:
        lock = json.loads((DEST / "standard.json").read_text())
        winner = lock["winner"]
    estimates = []
    contrasts = []
    for g in (*GAMES, "equal_games"):
        group = [g] if g != "equal_games" else GAMES
        for method in methods:
            for metric in blocks[GAMES[0]][method]:
                vals = [blocks[key][method][metric] for key in group]
                s = estimate(vals)
                s.pop("p")
                estimates.append(dict(game=g, method=method, metric=metric, **s))
                if method == winner or metric != "win":
                    continue
                diff = [blocks[key][winner][metric] - v for key, v in zip(group, vals)]
                r = estimate(diff)
                contrasts.append(
                    dict(
                        game=g,
                        method=method,
                        metric=metric,
                        winner=winner,
                        holm_p=None,
                        block_differences=[a.tolist() for a in diff],
                        **r,
                    )
                )
    holm(contrasts)
    with gzip.open(DEST / f"{stage}_estimates.json.gz", "wt") as f:
        json.dump(estimates, f, allow_nan=False)
    with gzip.open(DEST / f"{stage}_contrasts.json.gz", "wt") as f:
        json.dump(contrasts, f, allow_nan=False)
    table(stage, lock, estimates, contrasts, methods)
    print(stage, {k: v for k, v in lock.items() if k != "rank"})


def table(stage, lock, results, contrasts, methods):
    rows = {(r["game"], r["method"], r["metric"]): r for r in results}
    diffs = {(r["game"], r["method"]): r for r in contrasts}

    def cell(r):
        return f"{100 * r['mean']:.2f} ± {100 * r['sd']:.2f}"

    def bounds(r):
        return f"[{100 * r['ci95'][0]:.2f}, {100 * r['ci95'][1]:.2f}]"

    gs = [*GAMES, "equal_games"]
    names = sorted(methods, key=lambda m: rows["equal_games", m, "win"]["mean"], reverse=True)
    lines = [
        f"# {stage.capitalize()} win rates",
        "",
        "Each rule is named by its network, its continuing players and its whole-game utility. V sees the board after a move. Q sees the board and move. Copies take the largest immediate fitted frequency. Guarded copies first consider immediate wins and threats. A mixture draws an opponent type once per simulated path. Original uses the frozen source on both seats. Signed last, sum and discounted scores use learned visit frequencies. Win, points and quarter-draw give draws weights 0, 0.5 and 0.25. Hybrid adds a bounded learned term to quarter-draw. Fit ties prefer the higher immediate fitted score. Paths gives the sample count for the uniform-reply part in the larger games. Each VNM row compares complete games following the candidate, then chooses the highest expectation. Direct chooses the highest immediate score. [Full definitions and precision plan](DESIGN.md)",
        "",
        "Values are win percentages, mean ± sample standard deviation across originals. Each repeat averages two partitions with two fitted initializations each. The combined column weights each game by one third. Its spread describes the average of one independent original per game. Intervals are individual 95% intervals on paired differences, in percentage points. P-values use a two-sided zero-difference model and Holm correction over the win comparisons within the study stage.",
        "",
        f"The independent original counts in table order are {rows['equal_games', names[0], 'win']['n']}.",
        "",
        "| Method | Tic-tac-toe | Connect Four | 4×4 Reversi | Equal games | Selected minus row [95% interval] | Adjusted p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in names:
        diff = diffs.get(("equal_games", m))
        change = f"{100 * diff['mean']:+.2f} {bounds(diff)}" if diff else "reference"
        p = f"{diff['holm_p']:.3g}" if diff and diff["holm_p"] is not None else "—"
        lab = ("Direct " if m.endswith("/direct") else "VNM ") + m
        lines.append("| " + " | ".join([lab, *[cell(rows[g, m, "win"]) for g in gs], change, p]) + " |")
    if stage == "confirmation":
        m = lock["winner"]
        lines += [
            "",
            "## Selected VNM rule against external opponents and exact copies",
            "",
            "| Game | External wins | Draws | Losses | Same move as original on withheld boards | Mirror wins | Mirror draws | First seat wins in mirror |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for g in gs:
            lines.append(
                "| "
                + " | ".join(
                    [
                        g,
                        *[
                            cell(rows[g, m, metric])
                            for metric in (
                                "win",
                                "draw",
                                "loss",
                                "agreement",
                                "self_win",
                                "self_draw",
                                "self_first_win",
                            )
                        ],
                    ]
                )
                + " |"
            )
    (DEST / f"{stage}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("discovery", "tuning", "confirmation"))
    run(ap.parse_args().stage)
