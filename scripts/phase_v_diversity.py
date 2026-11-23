"""Evaluate the number of different visited positions at a matched eight-game budget."""

from concurrent.futures import ProcessPoolExecutor
import numpy as np
from policy_value_isomorph.phase_v import OUT, install_source, native, opening_panel, save, seed, spec
from policy_value_isomorph.phase_v_study import setup_fit, scoring, decoded
from policy_value_isomorph.phase_v_evaluate import TEMPS, methods


def run(job):
    game, b = job
    e = native().Engine(game == "reversi4")
    folder = OUT / game / f"block_{b:02}"
    install_source(e, folder)
    for k in range(2):
        for mix in (0, 1):
            e.configure(bool(mix), seed(game, b, 50, k))
            with np.load(folder / f"data_{k}_{mix}.npz") as d:
                counts = d["counts"]
                ids = np.flatnonzero((d["partition"] == 1) & (counts.sum(1) > 0))
                ids = np.random.default_rng(seed(game, b, 105, k)).choice(
                    ids, min(64, len(ids)), replace=False
                )
                valid = d["states"][ids].tolist()
                labels = counts[ids] / counts[ids].sum(1, keepdims=True)
            for f in range(2):
                setup_fit(e, folder, k, mix, f)
                rows = []
                pairs = opening_panel(e, game, b, k, f)
                for name, p in methods():
                    if p[0] not in (1, 2) or p[1] > 3:
                        continue
                    z = scoring(e, valid, p)
                    for t in TEMPS:
                        used = set()
                        turns = 0
                        for j in range(8):
                            s, op, key = pairs[4 * j]
                            rule = [*p[:3], t]
                            seat = j % 2
                            path = e.trajectory(
                                s, rule if seat == 0 else spec(op), spec(op) if seat == 0 else rule, key
                            )
                            decisions = list(map(tuple, path[:-1]))
                            used.update(decisions)
                            turns += len(decisions)
                        rows.append(
                            dict(
                                method=name,
                                temperature=t,
                                distinct_boards=len(used),
                                decisions=turns,
                                validation_tv=float(np.abs(decoded(z, t) - labels).sum(1).mean() / 2),
                            )
                        )
                save(folder / f"diversity_{k}_{mix}_{f}.jsonl.gz", rows)
    return job


if __name__ == "__main__":
    native()
    with ProcessPoolExecutor(6) as pool:
        for done in pool.map(run, [(g, b) for b in range(24) for g in ("connect4", "reversi4")]):
            print(done, flush=True)
