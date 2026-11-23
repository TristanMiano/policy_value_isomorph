"""Training and reproducible sampling for the two Phase V games."""
from collections import Counter
from functools import cache
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile

import numpy as np
import torch

OUT = Path(os.environ.get("PVI_PHASE_V_OUTPUT", Path(__file__).resolve().parents[2] / "experiments/phase_v"))
SIZE = 12 * 8192
STATE = tuple[int, int, int]

@cache
def native():
    """Build the optional CPU simulation extension outside the checkout (Linux/WSL or macOS)."""
    folder = Path(__file__).resolve().parent
    src = folder / "phase_v_native.cpp"
    files = src.read_bytes() + (folder/"phase_v_rules.hpp").read_bytes()
    key = hashlib.sha256(files + (sys.version + torch.__version__).encode()).hexdigest()[:16]
    build = Path(tempfile.gettempdir()) / ("pvi-phase-v-" + key)
    build.mkdir(exist_ok=True)
    lib = build / ("_phase_v_native" + sysconfig.get_config_var("EXT_SUFFIX"))
    if not lib.exists():
        tmp = build / (str(os.getpid()) + lib.suffix)
        cmd = [os.environ.get("CXX","c++"),"-std=c++17","-O3","-march=native","-shared","-fPIC",
               "-I"+sysconfig.get_paths()["include"],"-I"+str(Path(torch.__file__).parent/"include"),
               str(src),"-o",str(tmp)]
        subprocess.run(cmd,check=True)
        os.replace(tmp,lib)
    sys.path.insert(0,str(build))
    return importlib.import_module("_phase_v_native")

def seed(game, block, *parts):
    return int(np.random.SeedSequence([73022, int(game=="reversi4"),block,*parts]).generate_state(1)[0])

def save(path, rows):
    with gzip.open(path,"wt",encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")

def read(path):
    with gzip.open(path,"rt",encoding="utf8") as f:
        return [json.loads(line) for line in f]

def spec(model,rule=0,who=0,temp=-1):
    return [model,rule,who,temp]

def prefixes(e, n, rng, depth=None):
    result=[]
    for _ in range(n):
        path=e.trajectory(e.start(),spec(-1),spec(-1),int(rng.integers(2**32)))
        limit=len(path)-1 if depth is None else min(depth+1,len(path)-1)
        result.append(tuple(path[int(rng.integers(1 if depth and limit>1 else 0,limit))]))
    return result

def actions(e, s):
    mask=e.legal(s)
    return [i for i in range(mask.bit_length()) if mask&(1<<i)]

def moved(s,t,rev):
    diff=(s[0]|s[1])^(t[0]|t[1])
    return (diff.bit_length()-1 if diff else 16) if rev else (diff.bit_length()-1)//7

def split_of(e, s, salt):
    if list(s)==e.start():
        return 2
    # SplitMix64 finalizer of the symmetry orbit and independent partition salt.
    x=(e.canonical(s)^salt)*0xbf58476d1ce4e5b9 & ((1<<64)-1)
    x=(x^(x>>27))*0x94d049bb133111eb & ((1<<64)-1)
    bucket=(x^(x>>31))%100
    return 0 if bucket<65 else 1 if bucket<80 else 2

def fit_table(features, labels, train, valid, init, mode="squared", steps=(4,16,48), snapshots=None):
    """One active rectified pattern unit per group; optimize its linear readout in PyTorch."""
    legal=features[:,:,0]>=0
    tr=legal & train
    val=legal & valid
    if not tr.any() or not val.any():
        raise ValueError("training/validation needs legal candidates")
    # Compact frequently sparse feature indices, applying gradients only at observed inputs.
    ids, inverse=np.unique(features[tr],return_inverse=True)
    x=torch.from_numpy(inverse.reshape(-1,features.shape[-1])).long()
    y=torch.tensor(labels[tr],dtype=torch.float64)
    gen=torch.Generator().manual_seed(init)
    weight=torch.nn.Parameter(torch.randn(len(ids),dtype=torch.float64,generator=gen)*.012)
    occurrences=torch.bincount(x.ravel(),minlength=len(ids)).clamp(min=1)
    selected=None
    curves=[]
    for step in range(1,max(steps)+1):
        pred=weight[x].sum(1)
        if mode=="imitation":
            board, move=np.where(tr)
            dense=pred.new_full(legal.shape,-1e30)
            dense[board,move]=pred
            logits=torch.log_softmax(dense,dim=1)[board,move]
            loss=-(y*logits).sum()
        else:
            loss=((pred-y)**2).sum()/2
        loss.backward()
        with torch.no_grad():
            weight.sub_(weight.grad / occurrences * (.35 if mode=="imitation" else .75)/x.shape[1])
            weight.grad=None
        if step in steps:
            table=np.zeros(SIZE);table[ids]=weight.detach().numpy()
            def error(mask):
                if mode!="imitation":
                    return float(np.square(table[features[mask]].sum(1)-labels[mask]).mean())
                z=np.full(legal.shape,-1e30);z[mask]=table[features[mask]].sum(1)
                z-=z.max(1,keepdims=True)
                logp=z-np.log(np.exp(z).sum(1,keepdims=True))
                return float(-(logp[mask]*labels[mask]).sum()/mask.any(1).sum())
            row=dict(step=step,training=error(tr),validation=error(val),candidates=int(tr.sum()),
                     boards=int(tr.any(1).sum()),parameters=len(ids))
            curves.append(row)
            if snapshots is not None:
                snapshots.append(table.copy())
            if selected is None or row["validation"]<selected[1]["validation"]:
                selected=(table,row)
    return *selected,curves

def opening_panel(e, game, b, k, fit):
    rng=np.random.default_rng(seed(game,b,90,k,fit))
    pairs=[]
    for group,opps in enumerate(([-1],[9,10],[-2],[-3])):
        varied=prefixes(e,4,rng,6)
        for j in range(8):
            pairs.append((tuple(e.start()) if j<4 else varied[j-4],opps[j%len(opps)],int(rng.integers(2**32))))
    return pairs

def panel(e,p,pairs,against=None):
    games=[]
    for s,opp,rand in pairs:
        o=spec(opp) if against is None else against
        for seat in range(2):
            x,y=(p,o) if seat==0 else (o,p)
            value,ply,steps,calls=e.match(s,x,y,rand)
            games.append([value*(1-2*seat),ply,steps,calls])
    return games

def install_source(e,folder):
    data=np.load(folder/"source.npz")
    e.set(0,True,0,data["source"])
    e.correct(data["correct_states"].tolist(), data["correct_moves"].tolist(), data["correct_delta"].tolist())
    for j in (9,10):
        e.set(j,True,0,data[f"opponent_{j}"])
    return data

def source(game, block, folder):
    torch.set_num_threads(1)
    rev=game=="reversi4";e=native().Engine(rev)
    rng=np.random.default_rng(seed(game,block,1))
    all_states=[tuple(s) for s in e.all() if e.legal(s)] if rev else sorted(set(prefixes(e,5500,rng)))
    order=list(range(17 if rev else 7)) if rev else rng.permutation(7).tolist()
    kind=1 if rev or block%2==0 else 0
    targets=np.array([e.teacher(s,kind,order) for s in all_states])
    ids=rng.choice(len(all_states),min(6500,len(all_states)),replace=False)
    trainstates=[all_states[i] for i in ids]
    features=e.features(trainstates,True,0)
    labels=np.zeros(features.shape[:2]);labels[np.arange(len(ids)),targets[ids]]=1
    valid=(rng.random(len(ids))<.12)[:,None]
    weights,settings,curve=fit_table(features,labels,~valid,valid,seed(game,block,2),"imitation",(16,48,96))
    e.set(0,True,0,weights)
    logits=e.profiles(all_states,spec(0),False)[:,:labels.shape[1]]
    chosen=logits.argmax(1)
    incorrect=np.flatnonzero(chosen!=targets)
    fix=incorrect if rev else np.empty(0,dtype=int)
    deltas=1+logits[fix].max(1)-logits[fix,targets[fix]] if len(fix) else np.array([])
    corrections=np.asarray([all_states[i] for i in fix],dtype=np.uint64).reshape(-1,3)
    e.correct(corrections.tolist(),targets[fix].tolist(),deltas.tolist())
    after=e.profiles(all_states,spec(0),False)[:,:labels.shape[1]].argmax(1)
    extra={}
    if rev:
        given=np.array([e.solve(s)+e.solve(e.next(s,int(a))) for s,a in zip(all_states,after)])
        extra=dict(reachable=len(e.all()),optimal=int(np.sum(given==0)),points_forgone=given.mean(),worst=int(given.max()),opening_value=e.solve(e.start()))
    archive=dict(source=weights,correct_states=corrections,correct_moves=targets[fix],correct_delta=deltas)
    for j in (9,10):
        archive[f"opponent_{j}"]=rng.normal(0,.1,SIZE).astype("f4")
        e.set(j,True,0,archive[f"opponent_{j}"])
    folder.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(folder/"source.npz",**archive)
    (folder/"source.json").write_text(json.dumps(dict(game=game,block=block,teacher=kind,priority=order,
      decisions=len(all_states),before=int((chosen==targets).sum()),after=int((after==targets).sum()),
      curve=curve,selected=settings,**extra)))
    rows=[]
    for mixed in (False,True):
        e.configure(mixed,seed(game,block,3))
        for k in range(2):
            for f in range(2):
                openings=opening_panel(e,game,block,k,f)
                rows.append(dict(mixed=mixed,split=k,fit=f,openings=openings,games=panel(e,spec(0),openings)))
    save(folder/"baseline.jsonl.gz",rows)
