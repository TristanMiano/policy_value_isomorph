"""Supplied game points, independent rollout noise, fits to points, and structural controls."""
from fractions import Fraction
from graphlib import CycleError

import numpy as np

from .choice_training import strict_board_ranks
from .phase_iv_fit import designs, masks, selected
from .phase_iv_game import game, seed, soft
from .phase_iv_metrics import Evaluation, corrected_counts, dump
from .phase_iv_source import actors
from .pilot_exact import evaluate_source


def ending_samples(b,paths):
    last=paths[0].copy()
    for layer in paths[1:]:
        last=np.where(layer>=0,layer,last)
    return b.terminal[b.after[last]]


def run(block,dest,tmp):
    b=game()
    _,sources,opponents=actors(block,dest)
    rows,calibration,bounds,best_response=[],[],[],[]
    archive={}
    for name,opp in opponents.items():
        wdl=[b.endings(b.best_reply(opp,1),opp)[0],b.endings(opp,b.best_reply(opp,-1))[0,::-1]]
        best_response.append(dict(block=block,opponent=name,seat_wdl=np.asarray(wdl).tolist()))
    for kind,source in sources.items():
        sampler=b.sampler(source)
        ev=Evaluation(source,opponents)
        rational={b.states[i]:{a:Fraction(float(source[i,a])).limit_denominator(2520)
                               for a in np.flatnonzero(source[i])} for i in b.dec}
        reference=evaluate_source(b.graph,rational)
        exact=b.dense(np.array([float(b.side[i]*reference.action_value[b.states[i]][a])
                               for i,a in zip(b.parent,b.cell)]))
        best=b.greedy(exact)
        exact_match=source*best
        high=np.max(np.where(b.legal,exact,-np.inf),axis=1)
        high[b.ends]=0
        tied=(exact==high[:,None])&b.legal
        action_gap=np.sort(np.where(b.legal,exact,-np.inf),axis=1)
        multis=b.legal.sum(1)>1
        gap=np.zeros(b.n)
        gap[multis]=action_gap[multis,-1]-action_gap[multis,-2]
        archive[kind+"_exact"]=exact[b.legal]
        frequencies=source[b.legal]
        bychild=np.bincount(b.after,weights=frequencies,minlength=b.n)/np.bincount(b.after,minlength=b.n).clip(1)
        child_oracle=b.dense(bychild[b.after])
        if kind=="fixed":
            try:
                ranks=strict_board_ranks(sampler,frequencies)
                rank_copy=b.greedy(b.dense(ranks[b.after]))
                archive["strict_ranks"]=ranks.astype('i2')
            except CycleError:
                rank_copy=None
        for k in range(2):
            split=b.split(block,k)
            test=split==2
            ids=np.flatnonzero(test&~b.ends)
            key=dict(block=block,source=kind,split=k)
            exact_row=ev.measure(best,test,method="supplied_exact",network="table",fit=-1,**key)
            rows.append(exact_row)
            for pop,weight in (("all_boards",np.ones(b.n)),("selfplay",ev.visits)):
                w=weight[ids]
                correct=exact_match[ids].sum(1)
                any_best=(tied[ids]*source[ids]).sum(1)
                forgone=((high[ids,None]-exact[ids])*source[ids]).sum(1)
                bounds.append(dict(**key,population=pop,weight=float(w.sum()),boards=len(ids),
                    fixed_best_mass=float(correct@w),other_equal_mass=float((any_best-correct)@w),
                    strictly_worse_mass=float((1-any_best)@w),points_given_up=float(forgone@w/w.sum()),
                    deterministic_tv_floor=float((1-source[ids].max(1))@w/w.sum()),
                    maximizing_mixture_tv_floor=float((1-any_best)@w/w.sum())))
            rows.append(ev.measure(b.greedy(child_oracle),test,method="exact_shared_frequency",network="V",fit=-1,**key,
                        full_min_squared_error=float(np.square(bychild[b.after]-frequencies).mean())))
            rows.append(ev.measure(b.greedy(b.dense(frequencies)),test,method="exact_edge_frequency",network="Q",fit=-1,**key))
            if kind=="fixed" and rank_copy is not None:
                rows.append(ev.measure(rank_copy,test,method="exact_ordered_board",network="V",fit=-1,**key,
                                       all_board_matches=int((rank_copy*source).sum()),levels=len(set(ranks[b.after]))))
            # Same squared perturbation on all candidates, with distinct consequences for ranking.
            for magnitude in (.1,.5):
                for pattern,noise in (("common",np.ones_like(exact)),
                                      ("favor_rival",np.where(best>0,-1.,1.))):
                    z=exact+magnitude*noise
                    p=b.greedy(z)
                    calibration.append(dict(**key,method=pattern,magnitude=magnitude,mean_squared=magnitude**2,
                        agreement=float((p[ids]*source[ids]).sum(1).mean()),
                        greedy_changes=int(np.sum(best[ids].argmax(1)!=p[ids].argmax(1))),
                        soft_shift=float(np.abs(soft(b,z,.25)[ids]-soft(b,exact,.25)[ids]).sum(1).mean()/2)))
            for fit in range(2):
                rolls=ending_samples(b,sampler.rollout_paths(16,seed(block,61,k,fit)))
                for budget in (1,4,16):
                    targets=rolls[:,:budget].mean(1)
                    mover=targets*b.side[b.parent]
                    p=b.greedy(b.dense(mover))
                    row=ev.measure(p,test,method=f"rollout_{budget}",network="table",fit=fit,**key)
                    held=test[b.parent]
                    row["test_exact_loss"]=float(np.square(mover[held]-exact[b.legal][held]).mean())
                    rows.append(row)
                    if budget!=16:
                        continue
                    wdl=np.column_stack([(rolls==outcome).mean(1) for outcome in (1,0,-1)])
                    for family,x in designs().items():
                        tr,val,_=masks(split,family)
                        init=seed(block,63 if family=="V" else 64,k,fit)
                        for label,y in (("points",targets),("outcomes",wdl)):
                            pred,settings=selected(x,y,tr,val,init)
                            if label=="outcomes":
                                prob=pred.clip(0,1)
                                prob=np.divide(prob,prob.sum(1,keepdims=True),out=np.full_like(prob,1/3),
                                               where=prob.sum(1,keepdims=True)>0)
                                result=prob[:,0]-prob[:,2]
                                archive[f"{kind}_{k}_{fit}_{family}_outcomes"]=prob.astype("f4")
                            else:
                                result=pred.clip(-1,1)
                            score=b.dense(result*b.side[b.parent])
                            archive[f"{kind}_{k}_{fit}_{family}_{label}_score"]=score[b.legal].astype('f4')
                            p=b.greedy(score)
                            row=ev.measure(p,test,method=label,network=family,fit=fit,**key,**settings,
                                test_exact_loss=float(np.square(score[b.legal][held]-exact[b.legal][held]).mean()))
                            row["restore_lose"]=corrected_counts(source,best,p,ids) if kind=="fixed" else None
                            rows.append(row)
                            for low,hi in ((0,0),(0,.25),(.25,1),(1,2)):
                                subset=ids[(gap[ids]==0) if hi==0 else ((gap[ids]>low)&(gap[ids]<=hi))]
                                calibration.append(dict(**key,fit=fit,network=family,method=label,
                                    gap_low=low,gap_high=hi,boards=len(subset),
                                    disagreement_mass=float((1-(p[subset]*source[subset]).sum(1)).sum())))
    dump(tmp/"returns.jsonl.gz",rows)
    dump(tmp/"bounds.jsonl.gz",bounds)
    dump(tmp/"return_errors.jsonl.gz",calibration)
    dump(tmp/"best_reply.jsonl.gz",best_response)
    np.savez_compressed(tmp/"returns.npz",**archive)

