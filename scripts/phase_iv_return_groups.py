"""Evaluate payoff predictions on common/rare boards and against exact move-score gaps."""

from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from policy_value_isomorph.phase_iv_game import OUT, game, seed
from policy_value_isomorph.phase_iv_metrics import dump
from policy_value_isomorph.phase_iv_returns import ending_samples
from policy_value_isomorph.phase_iv_statistics import tabulate


def measurements():
    b = game()
    for block in range(32):
        folder = OUT / f"block_{block:02}"
        with np.load(folder / "source.npz") as original, np.load(folder / "returns.npz") as archive:
            moves = np.zeros(b.n, dtype=int)
            moves[b.dec] = original["moves"]
            fixed = b.point(moves)
            for kind, source in (("fixed", fixed), ("randomized", 0.75 * fixed + 0.25 * b.uniform)):
                sampler = b.sampler(source)
                reach = b.visitation(source)
                exact = archive[kind + "_exact"]
                sorted_scores = np.sort(np.where(b.legal, b.dense(exact), -np.inf), axis=1)
                multiple = b.legal.sum(1) > 1
                gap = np.zeros(b.n)
                gap[multiple] = sorted_scores[multiple, -1] - sorted_scores[multiple, -2]
                for k in range(2):
                    test = (b.split(block, k) == 2) & ~b.ends
                    median = np.median(reach[test & (reach > 0)])
                    groups = {
                        "all": test,
                        "reachable": test & (reach > 0),
                        "unreachable": test & (reach == 0),
                        "common": test & (reach >= median),
                        "rare": test & (reach > 0) & (reach < median),
                        "forced": test & ~multiple,
                        "tied": test & multiple & (gap == 0),
                        "up_to_0.25": test & (gap > 0) & (gap <= 0.25),
                        "up_to_1": test & (gap > 0.25) & (gap <= 1),
                        "above_1": test & (gap > 1),
                    }
                    for fit in range(2):
                        rolls = ending_samples(b, sampler.rollout_paths(16, seed(block, 61, k, fit)))
                        scores = {"exact": exact}
                        for budget in (1, 4, 16):
                            scores[f"rollout_{budget}"] = rolls[:, :budget].mean(1) * b.side[b.parent]
                        for net in ("V", "Q"):
                            for target in ("points", "outcomes"):
                                scores[net + "_" + target] = archive[
                                    f"{kind}_{k}_{fit}_{net}_{target}_score"
                                ]
                        for method, score in scores.items():
                            # Decode each archived table at its stored precision.
                            agreement = (b.greedy(b.dense(score)) * source).sum(1)
                            sq_error = np.square(score - exact)
                            for group, mask in groups.items():
                                edges = mask[b.parent]
                                count = int(mask.sum())
                                yield dict(
                                    block=block,
                                    split=k,
                                    fit=fit,
                                    source=kind,
                                    method=method,
                                    group=group,
                                    boards=count,
                                    candidates=int(edges.sum()),
                                    agreement=float(agreement[mask].mean()) if count else None,
                                    squared_error=float(sq_error[edges].mean()) if count else None,
                                )


def main():
    runs = list(measurements())
    dump(OUT / "return_groups_by_fit.jsonl.gz", runs)
    rows = tabulate(
        runs,
        ["source", "group", "method"],
        ["agreement", "squared_error", "boards", "candidates"],
        lambda g: (g[0], g[1], "exact"),
    )
    dump(OUT / "return_groups.jsonl.gz", rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3), layout="constrained", sharey=True)
    groups = ("tied", "up_to_0.25", "up_to_1", "above_1")
    for ax, kind in zip(axes, ("fixed", "randomized")):
        for method, label in (
            ("exact", "Exact points"),
            ("V_points", "Resulting board"),
            ("Q_points", "Board and move"),
        ):
            available = {
                r["group"]: r
                for r in rows
                if r["source"] == kind and r["method"] == method and r["metric"] == "agreement"
            }
            sequence = [(i, available[g]) for i, g in enumerate(groups) if g in available]
            ax.errorbar(
                [i for i, _ in sequence],
                [r["mean"] for _, r in sequence],
                yerr=[r["sd"] or 0 for _, r in sequence],
                fmt="o",
                capsize=3,
                label=label,
            )
        ax.set_title(kind.capitalize() + " original player")
        ax.set_xticks(range(4), ["Tied", "(0, 0.25]", "(0.25, 1]", "(1, 2]"])
        ax.set_xlabel("Best minus next-best exact game points")
        ax.set_ylim(-0.02, 1.05)
    axes[0].set_ylabel("Mean probability of matching the original move")
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Withheld boards with at least two moves\nMeans and one standard deviation across independent original players"
    )
    fig.savefig(OUT / "return_gaps.png", dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    main()
