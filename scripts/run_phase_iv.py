"""Run and resume independent Phase IV source blocks in separate CPU processes."""
import argparse
import importlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

for name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[name]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))


def run(job):
    stage,block,force=job
    from policy_value_isomorph.phase_iv_game import OUT
    dest=OUT/f"block_{block:02}"
    dest.mkdir(parents=True,exist_ok=True)
    marker=dest/(stage+".done")
    if marker.exists() and not force:
        return f"{block} already saved"
    import torch
    torch.set_num_threads(1)
    module=importlib.import_module("policy_value_isomorph.phase_iv_"+stage)
    with TemporaryDirectory(dir=dest) as folder:
        tmp=Path(folder)
        module.run(block,dest,tmp)
        for p in tmp.iterdir():
            p.replace(dest/p.name)
    marker.write_text("complete\n")
    return str(block)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["source","baseline","visits","probes","returns","utility","diagnostics"])
    parser.add_argument("--force",action="store_true",help="recompute and replace a completed block")
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--blocks",type=int,nargs="+",default=list(range(32)))
    args=parser.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        for completed in pool.map(run,[(args.stage,i,args.force) for i in args.blocks]):
            print(args.stage,completed,flush=True)

