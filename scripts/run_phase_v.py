"""Execute one independently reproducible stage for each frozen source block."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from policy_value_isomorph.phase_v import OUT, native, source


def run(job):
    stage, game, block, force = job
    folder = OUT / game / f"block_{block:02}"
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / (stage + ".done")
    if marker.exists() and not force:
        return (game, block, "saved")
    inputs = {
        "source": (),
        "fit": ("source.npz",),
        "evaluate": ("source.npz", "fit_*.npz", "data_*.npz"),
        "diagnostics": ("source.npz", "fit_*.npz", "data_*.npz", "metrics_*.jsonl.gz"),
    }
    with TemporaryDirectory(dir=folder) as temporary:
        work = Path(temporary)
        linked = set()
        for pattern in inputs[stage]:
            files = list(folder.glob(pattern))
            if not files:
                raise FileNotFoundError(f"{folder} requires {pattern} before {stage}")
            for file in files:
                (work / file.name).hardlink_to(file)
                linked.add(file.name)
        if stage == "source":
            source(game, block, work)
        else:
            from policy_value_isomorph.phase_v_study import run_stage

            run_stage(stage, game, block, work)
        marker.unlink(missing_ok=True)
        for file in work.iterdir():
            if file.name not in linked:
                file.replace(folder / file.name)
        (work / "complete").write_text("complete\n")
        (work / "complete").replace(marker)
    return game, block, stage


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["source", "fit", "evaluate", "diagnostics"])
    p.add_argument("--games", nargs="+", default=["connect4", "reversi4"], choices=["connect4", "reversi4"])
    p.add_argument("--blocks", nargs="+", type=int, default=list(range(24)))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if (
        len(a.blocks) != len(set(a.blocks))
        or not all(0 <= b < 24 for b in a.blocks)
        or len(a.games) != len(set(a.games))
    ):
        p.error("provide distinct blocks from 0 through 23 and distinct games")
    native()
    with ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(run, [(a.stage, g, b, a.force) for b in a.blocks for g in a.games]):
            print(*r, flush=True)
