"""Simulate precision for two randomized holdouts and two fresh fits within each source."""

import json
import numpy as np
from scipy.stats import t
from policy_value_isomorph.phase_v import OUT as DEST


def main():
    rng = np.random.default_rng(730022)
    rows = []
    for game, base in (("connect4", (0.085, 0.12, 0.21)), ("reversi4", (0.07, 0.11, 0.19))):
        for n in (12, 24, 36):
            for factor in (0.7, 1, 1.4):
                deviations = np.array(base) * factor
                copies = rng.normal(size=(2000, n, 3)) * deviations
                # Shared rotations/boards and openings stay inside an original-player repeat.
                copies += rng.normal(size=(2000, n, 2, 3)).mean(2) * deviations * 0.5
                copies += rng.normal(size=(2000, n, 2, 2, 3)).mean((2, 3)) * deviations * 0.65
                width = t.ppf(0.975, n - 1) * copies.std(axis=1, ddof=1) / np.sqrt(n)
                rows.append(
                    dict(game=game, sources=n, sensitivity=factor, half_width=width.mean(0).tolist())
                )
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "planning.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
