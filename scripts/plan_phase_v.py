"""Simulate precision for two randomized holdouts and two fresh fits within each source."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

DEST = Path(__file__).resolve().parents[1] / "experiments" / "phase_v"

def main():
    rng = np.random.default_rng(730022)
    rows = []
    for game, base in (("connect4", (.085, .12, .21)), ("reversi4", (.07, .11, .19))):
        for n in (12, 24, 36):
            for factor in (.7, 1, 1.4):
                deviations = np.array(base) * factor
                copies = rng.normal(size=(2000, n, 3)) * deviations
                # Shared rotations/boards and openings stay inside an original-player repeat.
                copies += rng.normal(size=(2000,n,2,3)).mean(2) * deviations * .5
                copies += rng.normal(size=(2000,n,2,2,3)).mean((2,3)) * deviations * .65
                width = t.ppf(.975, n-1) * copies.std(axis=1, ddof=1) / np.sqrt(n)
                rows.append(dict(game=game, sources=n, sensitivity=factor, half_width=width.mean(0).tolist()))
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST/"planning.json").write_text(json.dumps(rows, indent=2)+"\n")
if __name__=="__main__":
    main()
