"""Generate the quantitative fit curves and a searchable atlas of all reachable decisions."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from policy_value_isomorph.phase_iv_game import OUT, game
from policy_value_isomorph.phase_iv_statistics import records, block_means


def curves():
    data=block_means(records("training_steps"),["source","network","width","step"],
                     ["train_loss","test_exact_loss","agreement","points"])
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout="constrained")
    for col,xmetric in enumerate(("train_loss","test_exact_loss")):
        for row,ymetric in enumerate(("agreement","points")):
            ax=axes[row,col]
            for net,color in (("V","#297c77"),("Q","#a14f32")):
                for width,style in ((64,"--"),(192,"-")):
                    means=[];errors=[]
                    for step in (1,16,256,None):
                        cell=data["fixed",net,width,step]
                        values=[np.array(list(cell[m].values())) for m in (xmetric,ymetric)]
                        means.append([v.mean() for v in values])
                        errors.append([v.std(ddof=1) for v in values])
                    x,y=np.array(means).T
                    dx,dy=np.array(errors).T
                    ax.errorbar(x,y,xerr=dx,yerr=dy,color=color,linestyle=style,marker="o",
                                linewidth=1.3,elinewidth=.55,capsize=2,label=f"{net}, {width} units")
            ax.set_xlabel(("Training" if col==0 else "Withheld exact")+" squared frequency error")
            ax.set_ylabel("Fraction matching the original move" if row==0 else "Mean points per game")
            ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Direct selection during training · 32 independently trained optimal players\n"
                 "Means and one sample standard deviation across players; points follow 1, 16, 256 updates and convergence",
                 fontsize=12)
    fig.savefig(OUT/"training_curves.png",dpi=155)
    plt.close(fig)


def atlas():
    b=game()
    folder=OUT/"block_00"
    with np.load(folder/"visits.npz") as fits,np.load(folder/"baseline.npz") as data,np.load(folder/"source.npz") as original:
        arrays={}
        for net in ("V","Q"):
            for rule,suffix in (("direct","frequency"),("VNM","vnm_terminal_copies")):
                arrays[net+" "+rule]=np.stack([b.dense(fits[f"fixed_{k}_{j}_{net}_{suffix}"]) for k in range(2) for j in range(2)])
                if rule=="direct":
                    arrays[net+" "+rule]=arrays[net+" "+rule].clip(0,1)
        counts_by_board=[b.dense(data[f"fixed_{k}_counts"]) for k in range(2)]
        partitions=[data[f"partition_{k}"] for k in range(2)]
        moves=original["moves"]
        chosen={net+" "+rule:np.stack([fits[f"fixed_{k}_{f}_{net}_{tag}_choice"] for k in range(2) for f in range(2)])
                for net in ("V","Q") for rule,tag in (("direct","direct"),("VNM","vnm_terminal_copies"))}
        records_=[]
        for j,i in enumerate(b.dec):
            cells=np.flatnonzero(b.legal[i])
            cube={}
            for name,z in arrays.items():
                scores=z[:,i,:][:,cells]
                ordered=np.sort(scores,axis=1)
                cube[name]=dict(mean=np.round(scores.mean(0),5).tolist(),sd=np.round(scores.std(0,ddof=1),5).tolist(),
                           choices=chosen[name][:,j].tolist(),gap=np.round(ordered[:,-1]-ordered[:,-2],5).tolist() if len(cells)>1 else None)
            counts=[z[i,cells].tolist() for z in counts_by_board]
            records_.append(dict(board=b.names[i],side="X" if b.side[i]==1 else "O",
                    source=int(moves[j]),cells=cells.tolist(),
                    next=b.names[b.child[i,cells]].tolist(),groups=[int(a[i]) for a in partitions],
                    counts=counts,methods=cube))
    template=(Path(__file__).with_name("phase_iv_atlas.html")).read_text(encoding="utf8")
    (OUT/"atlas.html").write_text(template.replace("/* EMBEDDED_DATA */",json.dumps(records_,separators=(",",":"))),encoding="utf8")


if __name__=="__main__":
    curves()
    atlas()

