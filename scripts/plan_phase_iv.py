"""Size the paired tic-tac-toe experiment before observing any new outcomes."""

import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import t

OUT = Path(
    os.environ.get("PVI_PHASE_IV_OUTPUT", Path(__file__).resolve().parents[1] / "experiments" / "phase_iv")
)


def design():
    rng = np.random.default_rng(6200)
    rows = []
    # Fixed finite pool of symmetry groups. Fresh test samples can include the same group.
    groups = rng.normal(size=627)
    groups -= groups.mean()
    for n in (12, 24, 32):
        for scale in (0.6, 1.0, 1.4):
            # Per-source, per-partition, per-fit SDs in percentage points.
            source = rng.normal(0, 6 * scale, (4000, n))
            splits = rng.normal(0, 5 * scale, (4000, n, 2))
            fits = rng.normal(0, 8 * scale, (4000, n, 2, 2))
            finite = (
                np.array([groups[rng.choice(627, 125, replace=False)].mean() for _ in range(4000)]) * 20
            )
            blocks = source + (splits + rng.choice(finite, splits.shape)).mean(2) + fits.mean((2, 3))
            se = blocks.std(1, ddof=1) / np.sqrt(n)
            margin = t.ppf(0.975, n - 1) * se
            rows.append(
                dict(
                    sources=n,
                    splits_per_source=2,
                    fits_per_split=2,
                    sd_multiplier=scale,
                    mean_half_width_pp=float(margin.mean()),
                    mean_points_half_width=float(0.02 * margin.mean()),
                    power_conservative_six_tests={
                        str(effect): float(
                            (np.abs((blocks.mean(1) + effect) / se) > t.ppf(1 - 0.05 / 12, n - 1)).mean()
                        )
                        for effect in (0, 2, 5, 8)
                    },
                )
            )
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "planning.json").write_text(json.dumps(design(), indent=2) + "\n")
