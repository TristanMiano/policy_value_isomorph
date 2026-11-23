"""Summarize frozen source performance before any fitted player is evaluated."""
import json
import numpy as np
from policy_value_isomorph.phase_v import OUT,read
from policy_value_isomorph.phase_iv_statistics import describe

def main():
    results=[]
    for game in ("connect4","reversi4"):
        for mix in (0,1):
            for group in (("all","tactical","priority") if game=="connect4" else ("all",)):
                cells=[]
                for b in range(24):
                    if group!="all" and ((b%2==0)!=(group=="tactical")):continue
                    records=read(OUT/game/f"block_{b:02}"/"baseline.jsonl.gz")
                    games=np.asarray([r["games"] for r in records if r["mixed"]==mix])[:,:,0].reshape(-1,2)
                    cells.append([[np.mean(games[:,seat]==v) for v in (1,0,-1)] for seat in range(2)])
                a=np.asarray(cells)
                for seat,v in (("first",a[:,0]),("second",a[:,1]),("both",a.mean(1))):
                    results.append(dict(game=game,mixed=mix,target=group,seat=seat,
                        **{name:describe(v[:,i]) for i,name in enumerate(("win","draw","loss"))}))
    (OUT/"sources.json").write_text(json.dumps(results,indent=2)+"\n")
if __name__=="__main__":main()
