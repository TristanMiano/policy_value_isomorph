"""Measure representation bounds and group overlap directly from each saved full source."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from policy_value_isomorph.phase_iv_game import OUT,game
from policy_value_isomorph.phase_iv_metrics import dump,read


def main():
    b=game()
    rows=[]
    for block in range(32):
        folder=OUT/f"block_{block:02}"
        with np.load(folder/"baseline.npz") as base,np.load(folder/"returns.npz") as rt,np.load(folder/"source.npz") as frozen:
            signs=np.zeros((b.n,9))
            signs[b.dec,frozen["moves"]]=1
            ranks=rt["strict_ranks"][b.after]
            choice=b.greedy(b.dense(ranks))
            freq=signs[b.legal]
            entries=np.bincount(b.after,minlength=b.n)
            chosen=np.bincount(b.after,weights=freq,minlength=b.n)
            shared=(chosen>0)&(chosen<entries)
            full_min=np.mean(np.square(freq-(chosen/entries.clip(1))[b.after]))
            groups=[base[f"partition_{k}"] for k in range(2)]
            totals=[np.bincount(b.parent,weights=base[f"{kind}_{k}_counts"],minlength=b.n)
                    for kind in ("fixed","randomized") for k in range(2)]
            rows.append(dict(block=block,boards=len(b.dec),edges=len(ranks),successors=int((entries>0).sum()),
                conflicting_successors=int(shared.sum()),minimal_frequency_mse=float(full_min),
                strict_rank_levels=len(set(ranks)),rank_matches=int(np.sum(choice*signs)),
                partition_board_counts=[np.bincount(p[b.dec],minlength=3).tolist() for p in groups],
                common_test_boards=int(((groups[0]==2)&(groups[1]==2)&~b.ends).sum()),
                observed_boards=[int((v>0).sum()) for v in totals],retained_visits=[int(v.sum()) for v in totals]))
    dump(OUT/"finite_bounds.jsonl.gz",rows)


if __name__=="__main__":
    main()

