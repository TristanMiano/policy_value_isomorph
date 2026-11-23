"""Collect full-game visits, fit independent models and evaluate paired deployments."""
from collections import Counter
import json
import numpy as np
import torch

from .phase_v import OUT,SIZE,actions,fit_table,install_source,moved,native,opening_panel,panel,prefixes,read,save,seed,spec,split_of

def group(e,s):
    mask=e.legal(s)
    return 2 if mask==1<<16 else 1 if mask.bit_count()==1 else 0

def collect(e,game,b,k,mixed,folder):
    rev=game=="reversi4";rng=np.random.default_rng(seed(game,b,10,k))
    salt=seed(game,b,11,k)
    starts=prefixes(e,384,rng); initial=tuple(e.start())
    paths=[]; raw=Counter(); ending=Counter()
    for s in [initial]*96+starts*(3 if mixed else 1):
        path=[tuple(t) for t in e.trajectory(s,spec(0),spec(0),int(rng.integers(2**32)))]
        value=e.outcome(path[-1])
        paths.append(path)
        for h,t in zip(path,path[1:]):
            a=moved(h,t,rev);raw[h,a]+=1;ending[h,a,value*(1-2*h[2])]+=1
    observed=sorted({s for s,a in raw})
    hold=lambda s:split_of(e,s,salt)
    reach=Counter()
    for i in range(32):
        path=e.trajectory(initial,spec(0),spec(0),int(rng.integers(2**32)))
        reach.update(tuple(s) for s in path[:-1] if hold(tuple(s))==2)
    pool=([tuple(s) for s in e.all() if e.legal(s)] if rev else sorted(set(prefixes(e,1200,rng))))
    pool=[s for s in pool if hold(s)==2]
    selected=[];weights={}
    if rev:
        for g,quota in enumerate((128,32,24)):
            options=[s for s in pool if group(e,s)==g]
            ids=rng.choice(len(options),min(quota,len(options)),replace=False)
            for idx in ids:
                s=options[idx];selected.append(s);weights[s]=len(options)/len(ids)
    else:
        selected=[pool[i] for i in rng.choice(len(pool),min(160,len(pool)),replace=False)]
        weights={s:1. for s in selected}
    test=sorted(set(selected)|set(reach));states=sorted(set(observed)|set(test))
    idx={s:i for i,s in enumerate(states)};n=17 if rev else 7
    count=np.zeros((len(states),n),dtype="i4");out=np.zeros((len(states),n,3),dtype="i4")
    for (s,a),v in raw.items():count[idx[s],a]=v
    for (s,a,v),obs in ending.items():out[idx[s],a,1-v]=obs
    unique=sorted({s for p in paths for s in p});keys={s:i for i,s in enumerate(unique)}
    archive=dict(states=np.array(states,dtype="u8"),partition=np.array([hold(s) for s in states],dtype="i1"),
        counts=count,outcomes=out,test=np.array([idx[s] for s in test]),
        equal=np.array([weights.get(s,0) for s in test]),reach=np.array([reach[s] for s in test]),
        groups=np.array([group(e,s) for s in test]),pool=len(pool),path_boards=np.array(unique,dtype="u8"),
        path_ids=np.array([keys[s] for p in paths for s in p],dtype="i4"),
        path_offsets=np.cumsum([0]+list(map(len,paths))))
    np.savez_compressed(folder/f"data_{k}_{int(mixed)}.npz",**archive)
    return archive

def masks(e,states,partition,legal,counts):
    tr=legal & (partition[:,None]==0) & (counts.sum(1)[:,None]>0)
    va=legal & (partition[:,None]==1) & (counts.sum(1)[:,None]>0)
    reserved=set();testing=set()
    for i in np.flatnonzero(partition!=0):
        for a in np.flatnonzero(legal[i]):
            key=e.canonical(e.next(states[i],int(a)));reserved.add(key)
            if partition[i]==2:testing.add(key)
    vt=tr.copy();vv=va.copy()
    for mask,excluded in ((vt,reserved),(vv,testing)):
        for i,a in zip(*np.where(mask)):
            if e.canonical(e.next(states[i],int(a))) in excluded:mask[i,a]=False
    return (tr,va),(vt,vv)

def wmean(x,w):
    return float(np.average(x,weights=w)) if np.sum(w)>0 else None

def decoded(z,temperature):
    v=z-z.max(axis=1,keepdims=True)
    if temperature<0:return np.eye(z.shape[1])[z.argmax(1)]
    mass=(v==0).astype(float) if temperature==0 else np.exp(v/temperature)
    return mass/mass.sum(1,keepdims=True)

def scoring(e,states,p):
    return e.profiles(states,p,False)[:,:-2]

def probabilities(e,states,p):
    return e.profiles(states,p,True)[:,:-2]

def fit_block(e,game,b,k,mixed,folder):
    d=collect(e,game,b,k,mixed,folder)
    states=d["states"].tolist();count=d["counts"];total=count.sum(1,keepdims=True)
    target=count/total.clip(1);legal=np.array([[bool(e.legal(s)&(1<<a)) for a in range(count.shape[1])] for s in states])
    (tr,va),(vt,vv)=masks(e,states,d["partition"],legal,count)
    actual=scoring(e,states,spec(0));test=d["test"];teststates=[states[i] for i in test]
    original=probabilities(e,teststates,spec(0));weight=d["equal"]
    ordered=np.where(legal,actual,-np.inf)
    ranks=np.argsort(np.argsort(-ordered,axis=1,kind="stable"),axis=1,kind="stable")
    labels=np.maximum(0,3-ranks)/3
    # Elicit the three favorite remaining moves on a sampled fitting/validation menu.
    rng=np.random.default_rng(seed(game,b,32,k))
    asked=np.zeros(len(states),bool)
    for idx,n in ((np.flatnonzero(tr.any(1)),192),(np.flatnonzero(va.any(1)),64)):
        asked[rng.choice(idx,min(n,len(idx)),replace=False)]=True
    # Same ranking is observed by asking the frozen player with each preceding favorite removed.
    elicited=[]
    for i in np.flatnonzero(asked):
        menu=actions(e,states[i]); replies=[]
        for _ in range(min(3,len(menu))):
            answer=max(menu,key=lambda a:actual[i,a]);replies.append(answer);menu.remove(answer)
        elicited.append([int(i),replies])
    save(folder/f"rank_queries_{k}_{int(mixed)}.jsonl.gz",elicited)
    outcomes=d["outcomes"]/count[:,:,None].clip(1)
    logits=np.where(legal,actual-ordered.max(1,keepdims=True),0)
    for f in range(2):
        tables={};logs=[];curves=[]
        for mid in (1,2,7,8,3,4,6,11,12,13,14):
            v=mid in (1,7);layout=int(mid in (7,8))
            if mid in (6,12,13,14): train,valid=tr&(count>0),va&(count>0)
            elif mid==11:train,valid=tr&asked[:,None],va&asked[:,None]
            else:train,valid=(vt,vv) if v else (tr,va)
            y=labels if mid==11 else logits if mid==4 else outcomes[:,:,0]-outcomes[:,:,2] if mid==6 else outcomes[:,:,mid-12] if mid>=12 else target
            feat=e.features(states,not v,layout)
            snaps=[] if mid in (1,2) else None
            table,selected,curve=fit_table(feat,y,train,valid,seed(game,b,40+mid,k,f),
                "imitation" if mid==3 else "squared",snapshots=snaps)
            tables[f"m{mid}"]=table
            logs.append(dict(model=mid,selected=selected,curve=curve))
            if snaps:
                for checkpoint,row in zip(snaps,curve):
                    e.set(mid,not v,layout,checkpoint)
                    z=scoring(e,teststates,spec(mid))
                    p=decoded(z,-1);games=panel(e,spec(mid),opening_panel(e,game,b,k,f))
                    err=np.where(legal[test],(np.where(legal[test],z,0)-original)**2,0)
                    curves.append(dict(model=mid,**row,agreement=wmean((p*original).sum(1),weight),
                        held_squared=wmean(err.sum(1)/legal[test].sum(1),weight),
                        points=float(np.mean([g[0] for g in games])),games=games))
            e.set(mid,not v,layout,table)
        cell=np.zeros(SIZE);freq=count[tr.any(1)].sum(0);cell[:len(freq)]=freq/freq.sum()
        tables["m5"]=cell
        np.savez_compressed(folder/f"fit_{k}_{int(mixed)}_{f}.npz",**tables)
        save(folder/f"training_{k}_{int(mixed)}_{f}.jsonl.gz",logs)
        save(folder/f"curves_{k}_{int(mixed)}_{f}.jsonl.gz",curves)

def setup_fit(e,folder,k,mixed,f):
    with np.load(folder/f"fit_{k}_{int(mixed)}_{f}.npz") as data:
        for key in data.files:
            mid=int(key[1:]);e.set(mid,mid not in (1,7),int(mid in (7,8)),data[key])

def run_stage(stage,game,b,folder):
    torch.set_num_threads(1)
    e=native().Engine(game=="reversi4")
    install_source(e,folder)
    for k in range(2):
        for mix in (0,1):
            e.configure(bool(mix),seed(game,b,50,k))
            if stage=="fit":fit_block(e,game,b,k,mix,folder)
            else:
                if stage=="evaluate":
                    from .phase_v_evaluate import evaluate as evaluate_fit
                else:
                    from .phase_v_diagnostics import diagnostics as evaluate_fit
                for f in range(2):
                    setup_fit(e,folder,k,mix,f)
                    evaluate_fit(e,game,b,k,mix,f,folder)
