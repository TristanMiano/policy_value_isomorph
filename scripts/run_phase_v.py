"""Execute one independently reproducible stage for each frozen source block."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from policy_value_isomorph.phase_v import OUT,native,source

def run(job):
    stage,game,block,force=job
    folder=OUT/game/f"block_{block:02}"
    folder.mkdir(parents=True,exist_ok=True)
    marker=folder/(stage+".done")
    if marker.exists() and not force:return (game,block,"saved")
    if stage=="source": source(game,block,folder)
    else:
        from policy_value_isomorph.phase_v_study import run_stage
        run_stage(stage,game,block,folder)
    marker.write_text("complete\n")
    return game,block,stage

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["source","fit","evaluate","diagnostics"])
    p.add_argument("--games",nargs="+",default=["connect4","reversi4"],choices=["connect4","reversi4"])
    p.add_argument("--blocks",nargs="+",type=int,default=list(range(24)))
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--force",action="store_true")
    a=p.parse_args()
    native()
    with ProcessPoolExecutor(a.workers) as pool:
        for r in pool.map(run,[(a.stage,g,b,a.force) for b in a.blocks for g in a.games]):
            print(*r,flush=True)
