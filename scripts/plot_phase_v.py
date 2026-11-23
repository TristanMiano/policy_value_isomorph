"""Plot paired primary effects with their individual 95% intervals."""

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from policy_value_isomorph.phase_v import OUT


def main():
    rows = json.loads((OUT / "primary.json").read_text())
    games = ["tictactoe", "connect4", "reversi4", "equal_game_average"]
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.5), sharey=True, layout="constrained")
    for ax, metric, title in zip(
        axes, ("agreement", "win"), ("Copying the original move", "Winning against the opponent panel")
    ):
        for net, offset, color, label in (
            ("V", 0.12, "#217a70", "Scores resulting board"),
            ("Q", -0.12, "#a84829", "Scores board and move"),
        ):
            local = [
                next(
                    r["paired"]
                    for r in rows
                    if r["game"] == game and r["metric"] == metric and r["method"].startswith(net + "_")
                )
                for game in games
            ]
            for i, r in enumerate(local):
                avg = r["mean"] * 100
                lo, hi = [a * 100 for a in r["ci95"]]
                ax.errorbar(
                    avg,
                    3 - i + offset,
                    xerr=[[avg - lo], [hi - avg]],
                    fmt="o",
                    color=color,
                    capsize=4,
                    label=label if i == 0 else None,
                )
        ax.axvline(0, color="#555", linewidth=0.8)
        ax.grid(axis="x", alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("VNM minus direct (percentage points)")
        ax.set_yticks([3, 2, 1, 0], ["Tic-tac-toe", "Connect Four", "4×4 Reversi", "Equal game average"])
        ax.set_ylim(-0.5, 3.55)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Looking ahead using the fitted frequency of the final move\nBoth seats continue with fitted direct copies · means and individual 95% confidence intervals"
    )
    fig.savefig(OUT / "primary.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
