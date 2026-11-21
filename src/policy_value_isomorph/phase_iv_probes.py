"""Matched restricted-menu experiments, finite-sample probabilities and opening designs."""
from itertools import combinations

import numpy as np

from .coverage_sampling import random_openings, reference_visits, sample_games
from .phase_iv_fit import candidates, designs, masks
from .phase_iv_game import game, seed
from .phase_iv_metrics import dump
from .phase_iv_source import actors
from .policy_adapter import GameAdapter
from .preference_probes import probe


def menu_tests(block,raw):
    b=game()
    rows=[]
    index={s:i for i,s in enumerate(b.states)}
    for k in range(2):
        rng=np.random.default_rng(seed(block,40,k))
        available=np.flatnonzero((b.split(block,k)==2)&(b.legal.sum(1)>1))
        for idx in rng.choice(available,32,replace=False):
            s=b.states[idx]
            for contextual in (False,True):
                def choose(state,menu,unused):
                    sign=-1 if contextual and len(menu)==2 else 1
                    return max(sorted(menu),key=lambda a:sign*raw[index[state],a])
                adapter=GameAdapter("reverses pairs" if contextual else "stable scores",
                                    fixed=lambda state:choose(state,state.legal_moves(),None),choose_menu=choose)
                menu=adapter.candidates(s)
                pairs=list(combinations(menu,2))
                shuffled=[pairs[i] for permutation in range(8) for i in rng.permutation(len(pairs))]
                # Two additional questions directly compare a full-menu choice to a restricted pair.
                anchor=choose(s,menu,None)
                pair=(anchor,next(a for a in menu if a!=anchor))
                actual=choose(s,pair,None)
                for strategy in ("elimination","active pairs","random pairs"):
                    for budget in (1,3,8):
                        record=probe(adapter,s,b.names[idx],strategy=strategy,budget=budget,
                                     seed=seed(block,41,k),check_menus=shuffled[:budget] if strategy=="random pairs" else ())
                        diag=record.order_diagnostics()
                        good=sum((raw[idx,a],-a)>(raw[idx,c],-c) for a,c in diag["ordered_pairs"])
                        rows.append(dict(block=block,split=k,board=int(idx),contextual=contextual,strategy=strategy,
                            budget=budget,pairs=len(pairs),correct=int(good),ordered=len(diag["ordered_pairs"]),
                            unknown=len(diag["undetermined_pairs"]),conflicts=len(diag["conflicting_pairs"]),
                            calls=len(record.history),extra_calls=2,full_to_pair_reversal=int(actual!=anchor),
                            history=[[list(q.menu),q.choice] for q in record.history]))
    return rows


def distances(law,p,legal,ids):
    positive=law>0
    missing=(positive&(p==0)).any(1)
    kl=np.sum(np.where(positive,law*(np.log(law.clip(1e-300))-np.log(p.clip(1e-300))),0),1)
    finite=ids[~missing[ids]]
    return dict(boards=len(ids),tv=float(np.abs(law[ids]-p[ids]).sum(1).mean()/2),
        infinite_kl_boards=int(missing[ids].sum()),mean_kl="infinity" if missing[ids].any() else float(kl[ids].mean()),
        finite_mean_kl=float(kl[finite].mean()) if len(finite) else None,
        missed_probability=float(np.where(p[ids]==0,law[ids],0).sum(1).mean()),
        extra_probability=float(np.where(law[ids]==0,p[ids],0).sum(1).mean()))


def run(block,dest,tmp):
    b=game()
    raw,sources,_=actors(block,dest)
    dump(tmp/"menus.jsonl.gz",menu_tests(block,raw))
    stochastic=[]
    coverage=[]
    arrays={}
    top=sources["fixed"]
    second=b.greedy(np.where(top>0,-np.inf,raw))
    restricted=.75*top+.25*second
    # On forced rows there is exactly one choice.
    restricted[b.legal.sum(1)==1]=top[b.legal.sum(1)==1]
    for name,law in {"mixture":sources["randomized"],"top_two":restricted}.items():
        sampler=b.sampler(law)
        for k in range(2):
            ids=np.flatnonzero((b.split(block,k)==2)&~b.ends)
            rng=np.random.default_rng(seed(block,42,k))
            counts=np.zeros_like(law,dtype=int)
            for n in range(1,65):
                e=sampler.draw_edges(ids,rng)
                counts[ids,b.cell[e]]+=1
                if n not in (1,4,16,64):
                    continue
                for label,p in (("counts",counts/n),("plus_one",(counts+b.legal)/(n+b.legal.sum(1)[:,None])),
                                ("exact_log",np.exp(np.log(law,where=law>0,out=np.full_like(law,-np.inf))))):
                    stochastic.append(dict(block=block,split=k,source=name,n=n,method=label,
                                           **distances(law,p,b.legal,ids)))
                arrays[f"{name}_{k}_{n}"]=counts[ids].astype('u1')
    lookup={s:i for i,s in enumerate(b.states)}
    for name,law in sources.items():
        sampler=b.sampler(law)
        nat=reference_visits(sampler,False)
        opp=reference_visits(sampler,True)
        for k in range(2):
            split=b.split(block,k)
            rng=np.random.default_rng(seed(block,43,k))
            openings,cost=random_openings(512,rng)
            pool=np.array([lookup[s] for s in openings])
            test=np.flatnonzero((split==2)&~b.ends)
            for method in ("empty_self","empty_random","random_openings","low_visit_openings"):
                for budget,sample in sample_games(sampler,pool,method,(512,2048),seed(block,44,k)).items():
                    labels,totals=sampler.frequencies(sample.counts)
                    seen=np.bincount(b.parent,weights=sample.counts,minlength=b.n)>0
                    tr,val,_=masks(split,"Q")
                    tr &= totals>0
                    # A single fixed architecture for the coverage comparison, no evaluation labels selected.
                    pred,_=next(candidates(designs()["Q"],np.nan_to_num(labels),tr,val,seed(block,45,k),
                                           widths=(64,),penalties=(1.,)))
                    p=b.greedy(b.dense(pred.clip(0,1)))
                    fitted=b.dense(pred.clip(0,1))*b.legal
                    denom=fitted.sum(1,keepdims=True)
                    estimated=np.divide(fitted,denom,out=b.uniform.copy(),where=denom>0)
                    for pop,weight in (("selfplay",nat),("uniform_opposition",opp)):
                        positive=test[weight[test]>0]
                        median=np.median(weight[positive]) if len(positive) else 0
                        for group,ids in (("common",test[(weight[test]>0)&(weight[test]>=median)]),
                                          ("rare",test[(weight[test]>0)&(weight[test]<median)]),
                                          ("unreachable",test[weight[test]==0])):
                            coverage.append(dict(block=block,source=name,split=k,method=method,budget=budget,
                                design=pop,group=group,boards=len(ids),seen=int(seen.sum()),group_seen=int(seen[ids].sum()),
                                normal_visit_mass=float(weight[seen].sum()/weight.sum()),games=sample.games_started,
                                opening_pool_moves=cost if "openings" in method else 0,opponent_turns=sample.random_opponent_turns,
                                extra_turns=sample.extra_source_turns_to_finish,
                                agreement=float((p[ids]*law[ids]).sum(1).mean()) if len(ids) else None,
                                tv=float(np.abs(estimated[ids]-law[ids]).sum(1).mean()/2) if len(ids) else None))
                    arrays[f"coverage_{name}_{k}_{method}_{budget}"]=sample.counts.astype("i2")
    dump(tmp/"sample_laws.jsonl.gz",stochastic)
    dump(tmp/"coverage.jsonl.gz",coverage)
    np.savez_compressed(tmp/"sampling.npz",**arrays)

