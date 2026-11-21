"""Behavioral encodings, sampled choices, and independent neural imitators."""
import numpy as np
import torch
from torch.nn import functional as F

from .neural_game_control import network, optimizer_step
from .phase_iv_game import game, seed, soft
from .phase_iv_fit import collect
from .phase_iv_source import actors
from .phase_iv_metrics import Evaluation, dump


def run(block,dest,tmp):
    b=game()
    _,sources,opponents=actors(block,dest)
    rows=[]
    arrays={}
    for kind,law in sources.items():
        ev=Evaluation(law,opponents)
        sampler=b.sampler(law)
        log=np.full_like(law,-np.inf)
        np.log(law,out=log,where=law>0)
        exact=soft(b,log,1)
        if not np.allclose(exact,law,atol=1e-15):
            raise ArithmeticError("log encoding lost probability")
        for k in range(2):
            split=b.split(block,k)
            counts,calls=collect(sampler,split,1 if kind=="fixed" else 8,block,k)
            target,_=sampler.frequencies(counts)
            train=torch.from_numpy(np.flatnonzero(split[b.dec]==0))
            obs=torch.tensor(b.dense(target)[b.dec],dtype=torch.float32)
            arrays[f"{kind}_{k}_counts"]=counts.astype("i2")
            arrays[f"partition_{k}"]=split.astype("i1")
            common=dict(block=block,source=kind,split=k,calls=calls,fit=-1)
            for name,p in (("source",law),("exact_log",exact)):
                rows.append(ev.measure(p,split==2,method=name,**common))
            for fit in range(2):
                torch.manual_seed(seed(block,31,k,fit))
                net=network(10,9)
                opt=torch.optim.Adam(net.parameters(),lr=.003)
                for _ in range(30):
                    for ids in train[torch.randperm(len(train))].split(256):
                        logp=F.log_softmax(net(b.x[ids]).masked_fill(~b.mask[ids],-1e9),1)
                        optimizer_step(-(obs[ids]*logp).sum(1).mean(),opt)
                with torch.no_grad():
                    scores=np.zeros_like(law)
                    scores[b.dec]=net(b.x).numpy()
                probs=soft(b,scores,1)
                trained=probs if kind=="randomized" else b.greedy(scores)
                arrays[f"{kind}_{k}_{fit}_imitation"]=scores[b.dec].astype("f4")
                train_prob=probs[b.dec][train]
                loss=-np.mean((obs[train].numpy()*np.log(train_prob.clip(1e-300))).sum(1))
                rows.append(ev.measure(trained,split==2,method="imitation",**{**common,"fit":fit},
                                       training_nats=float(loss)))
    np.savez_compressed(tmp/"baseline.npz",**arrays)
    dump(tmp/"baseline.jsonl.gz",rows)

