"""Random-feature neural regression and frequency collection under the frozen protocol."""
from functools import cache

import numpy as np
from scipy.linalg import eigh

from .choice_training import line_features
from .phase_iv_game import game, seed


@cache
def designs():
    b=game()
    cells=np.array([s.board for s in b.states])
    board=np.column_stack([cells,b.side,[line_features(s) for s in b.states]])
    return {"V":board[b.after], "Q":np.column_stack([board[b.parent],board[b.after],np.eye(9)[b.cell]])}


def masks(split,family):
    b=game()
    train=split[b.parent]==0
    val=split[b.parent]==1
    if family=="V":
        train &= ~np.isin(b.after,b.after[~train])
        val &= ~np.isin(b.after,b.after[split[b.parent]==2])
    return train,val,split[b.parent]==2


def collect(sampler,split,count,block,k):
    b=game()
    counts=np.zeros(len(b.parent),int)
    calls=[]
    for j in range(3):
        starts=np.flatnonzero((split==j)&~b.ends)
        sample=sampler.visits(starts,count,seed(block,21+j,k))
        own=split[b.parent]==j
        counts[own]=sample[own]
        calls.append(int(sample.sum()))
    return counts,calls


def candidates(x,y,train,validation,init,*,widths=(64,192),penalties=(.01,1.,100.),steps=(None,)):
    """Solve sum of squared errors plus lambda times squared non-intercept weights."""
    mean=x[train].mean(0)
    scale=x[train].std(0)
    scale[scale<1e-9]=1
    x=(x-mean)/scale
    rng=np.random.default_rng(init)
    h=np.maximum(0,x@rng.normal(0,1/np.sqrt(x.shape[1]),(x.shape[1],max(widths)))+
                 rng.normal(0,.5,max(widths)))
    for width in widths:
        features=h[:,:width]
        center=features[train].mean(0)
        f=features-center
        gram=f[train].T@f[train]
        ev,vec=eigh(gram)
        avg=y[train].mean(axis=0)
        rhs=vec.T@(f[train].T@(y[train]-avg))
        for penalty in penalties:
            denom=(ev.clip(0)+penalty)
            for step in steps:
                # Exact iterate of full-batch gradient descent on this quadratic, starting at zero
                # non-intercept weights. None is its limiting fitted solution.
                solution=rhs/(denom if rhs.ndim==1 else denom[:,None])
                if step is not None:
                    shrink=1-(1-denom/denom.max()).clip(0,1)**step
                    solution=solution*(shrink if rhs.ndim==1 else shrink[:,None])
                coef=vec@solution
                pred=f@coef+avg
                err=np.square(pred[train]-y[train]).mean()
                valid=np.square(pred[validation]-y[validation]).mean()
                yield pred,dict(width=width,penalty=penalty,train_loss=float(err),validation_loss=float(valid),
                            training_candidates=int(train.sum()),validation_candidates=int(validation.sum()),
                            **({"step":step} if len(steps)>1 else {}),
                            penalized_loss_per_candidate=float(err+penalty*np.square(coef).sum()/(train.sum()*(1 if y.ndim==1 else y.shape[1]))))


def selected(x,y,train,validation,init):
    return min(candidates(x,y,train,validation,init),key=lambda pair:pair[1]["validation_loss"])

