"""Probability models, learned complete-game utilities and prospective changed choices."""
import hashlib

import numpy as np

from .phase_iv_game import game, seed, soft
from .phase_iv_metrics import Evaluation, dump
from .phase_iv_source import actors

GRID=np.linspace(-1,1,9)


def draw_scores(b,ending,d):
    # X's utility equals win probability minus loss probability plus d times draw probability.
    x=ending[:,0]-ending[:,2]+d*ending[:,1]
    return np.round(b.side[:,None]*x[b.child],14)


def fit_draw(b,ending,source,train):
    grid=[draw_scores(b,ending,d) for d in GRID]
    # For a random source this is the probability of matching an independent source draw.
    matches=np.array([(b.greedy(scores)[train]*source[train]).sum() for scores in grid])
    equivalent=np.flatnonzero(np.isclose(matches,matches.max(),atol=1e-10,rtol=0))
    i=min(equivalent,key=lambda j:(abs(GRID[j]),GRID[j]))
    return grid[i],float(GRID[i]),GRID[equivalent].tolist()


def from_ranking(scores,allowed,kind,edit=False):
    z=np.where(allowed,scores,-np.inf)
    best=z.argmax(1)
    ids=np.arange(len(best))
    if edit:
        z[ids,best]=-np.inf
        best=z.argmax(1)
    p=np.zeros_like(z)
    p[ids,best]=1
    if kind=="randomized":
        p=.75*p+.25*allowed/allowed.sum(1,keepdims=True)
    return p


def query_source(raw,ids,allowed,kind,edit):
    # Separate query implementation. The network sees the board and the offered menu.
    law=np.zeros((len(ids),9))
    for j,board in enumerate(ids):
        options=sorted(np.flatnonzero(allowed[j]),key=lambda a:(-raw[board,a],a))
        winner=options[1 if edit else 0]
        if kind=="randomized":
            law[j,options]=.25/len(options)
            law[j,winner]+=.75
        else:
            law[j,winner]=1.
    return law


def intervention_records(b,raw,source,kind,split,methods,cases,prefix):
    ids=np.flatnonzero((split==2)&(b.legal.sum(1)>1))
    legal=b.legal[ids]
    favorite=source[ids].argmax(1)
    other=np.where(legal & (np.arange(9)!=favorite[:,None]),np.arange(9),10).argmin(1)
    for event,remove,edit in (("remove_favorite",favorite,False),("remove_other",other,False),
                               ("edited_priority",None,True)):
        allowed=legal.copy()
        if remove is not None:
            allowed[np.arange(len(ids)),remove]=False
        predictions={m:from_ranking(z[ids],allowed,kind,edit) for m,z in methods.items()}
        digest={m:hashlib.sha256(p.tobytes()).hexdigest() for m,p in predictions.items()}
        answers=query_source(raw,ids,allowed,kind,edit)
        cases[prefix+event+"_board"]=ids.astype("i2")
        cases[prefix+event+"_answer"]=answers.argmax(1).astype("i1")
        cases[prefix+event+"_mask"]=np.packbits(allowed,axis=1)
        for m,p in predictions.items():
            cases[prefix+event+"_"+m]=p.argmax(1).astype("i1")
            yield dict(event=event,method=m,boards=len(ids),tv=float(np.abs(answers-p).sum(1).mean()/2),
                       independent_match=float((answers*p).sum(1).mean()),prediction_hash=digest[m])


def lottery_fixture(block,k,cases):
    rng=np.random.default_rng(seed(block,78,k))
    # Feature is the chance of an X win in a win/loss gamble; the alternative is a certain draw.
    train_prob=rng.random(64)
    test_prob=rng.random(128)
    train_first=rng.integers(2,size=64).astype(bool)
    test_first=rng.integers(2,size=128).astype(bool)
    true=.375
    for name in ("shared_draw","presentation"):
        train_choice=(true>=2*train_prob-1) if name=="shared_draw" else train_first
        errors=np.array([np.sum((d>=2*train_prob-1)!=train_choice) for d in GRID])
        eq=GRID[errors==errors.min()]
        chosen=min(eq,key=lambda v:(abs(v),v))
        # Direct action scores use the frequency of choosing draw in eight probability bands.
        bins=(train_prob*8).astype(int)
        den=np.bincount(bins,minlength=8)
        freq=(np.bincount(bins,weights=train_choice,minlength=8)+1)/(den+2)
        often=train_choice.mean()>=.5
        for event,p,first in (("fresh",test_prob,test_first),("changed_odds",1-test_prob,test_first),
                              ("changed_order",test_prob,~test_first),("other_player",test_prob,test_first)):
            predictions={"compact_VNM":chosen>=2*p-1,"supplied_VNM":0>=2*p-1,
                         "direct_bins":freq[(p*8).astype(int)]>=.5,"constant":np.full(len(p),often),
                         "first_presented":first}
            uses_utility=(name=="shared_draw")!=(event=="other_player")
            actual=true>=2*p-1 if uses_utility else first
            prefix=f"lottery_{k}_{name}_{event}_"
            cases[prefix+"win_probability"]=p
            cases[prefix+"draw_presented_first"]=first
            cases[prefix+"actual_draw"]=actual
            for m,pr in predictions.items():
                cases[prefix+m]=pr
                yield dict(block=block,split=k,source=name,event=event,method=m,n=len(p),
                       wrong=int(np.sum(pr!=actual)),error=float(np.mean(pr!=actual)),draw_utility=float(chosen),
                       equivalent=eq.tolist(),true_draw=true if name=="shared_draw" else None,
                       prediction_hash=hashlib.sha256(pr.tobytes()).hexdigest())


def run(block,dest,tmp):
    raw,sources,opponents=actors(block,dest)
    b=game()
    measures,interventions,lotteries=[],[],[]
    cases={}
    with np.load(dest/"visits.npz") as visit,np.load(dest/"returns.npz") as returns,np.load(dest/"baseline.npz") as base:
        for k in range(2):
            split=b.split(block,k)
            lotteries.extend(lottery_fixture(block,k,cases))
            for kind,source in sources.items():
                ev=Evaluation(source,opponents)
                ending=b.endings(source,source)
                train=np.flatnonzero((split==0)&~b.ends)
                valid=(split==1)&~b.ends
                test=(split==2)&~b.ends
                counts=base[f"{kind}_{k}_counts"]
                countlaw=b.dense(counts.astype(float))
                total=countlaw.sum(1,keepdims=True)
                data=np.divide(countlaw,total,out=np.zeros_like(countlaw),where=total>0)
                fitted,d,equal=fit_draw(b,ending,data,train)
                supplied=b.dense(returns[kind+"_exact"])
                cell_order=np.broadcast_to(countlaw[train].sum(0),raw.shape)
                for fit in range(2):
                    key=dict(block=block,source=kind,split=k,fit=fit)
                    imitate=np.zeros_like(raw)
                    imitate[b.dec]=base[f"{kind}_{k}_{fit}_imitation"]
                    methods={"original_scores":raw,"cell_frequency":cell_order,"imitation":imitate,
                             "compact_VNM":fitted,"supplied_VNM":supplied}
                    score_tables={"compact_VNM":fitted,"supplied_VNM":supplied}
                    for family in ("V","Q"):
                        prefix=f"{kind}_{k}_{fit}_{family}_"
                        freq=b.dense(visit[prefix+"frequency"].clip(0,1))
                        methods[family+"_direct"]=freq
                        score_tables[family+"_direct"]=freq
                        for rule in ("terminal","sum","decay"):
                            for who in ("copies","source"):
                                name=family+"_VNM_"+rule+"_"+who
                                score_tables[name]=b.dense(visit[prefix+"vnm_"+rule+"_"+who])
                                methods[name]=score_tables[name]
                        for mode in ("points","outcomes"):
                            name=family+"_"+mode
                            score_tables[name]=b.dense(returns[prefix+mode+"_score"])
                            methods[name]=score_tables[name]
                    # Same visit observations and no original-player deployment queries for these controls.
                    copies=b.greedy(methods["Q_direct"])
                    fitted_lotteries=b.endings(copies,copies)
                    new_score,new_draw,new_equal=fit_draw(b,fitted_lotteries,data,train)
                    score_tables["compact_VNM_copies"]=new_score
                    score_tables["supplied_VNM_copies"]=draw_scores(b,fitted_lotteries,0)
                    methods.update({n:score_tables[n] for n in ("compact_VNM_copies","supplied_VNM_copies")})
                    for row in intervention_records(b,raw,source,kind,split,methods,cases,f"{kind}_{k}_{fit}_"):
                        interventions.append(dict(**key,**row))
                    for name,z in score_tables.items():
                        probs=[soft(b,z,t) for t in (0,.05,.25,1)]
                        validation=[np.abs(p[valid]-source[valid]).sum(1).mean()/2 for p in probs]
                        selected=int(np.argmin(validation))
                        for j,(t,p) in enumerate(zip((0,.05,.25,1),probs)):
                            support=(source>0)&test[:,None]
                            # Center both observed and implied scores over the source-supported menu.
                            delta=np.where(support,z-t*np.log(source.clip(1e-300)),0)
                            avg=delta.sum(1)/support.sum(1).clip(1)
                            residual=(delta-avg[:,None])[support]
                            measures.append(ev.measure(p,test,method=name,temperature=t,selected=j==selected,**key,
                                utility_draw=d if name=="compact_VNM" else (new_draw if name=="compact_VNM_copies" else None),
                                equivalent_draws=equal if name=="compact_VNM" else (new_equal if name=="compact_VNM_copies" else None),
                                rms_score_residual=float(np.sqrt(np.mean(residual**2))) if t>0 else None,
                                missing=float(np.where(p[test]==0,source[test],0).sum(1).mean()),
                                outside_source=float(np.where(source[test]==0,p[test],0).sum(1).mean()),
                                validation_tv=float(validation[j])))
                    # Equal mixture of two complete-play utilities. One chosen utility is used per decision.
                    mixture=(b.greedy(draw_scores(b,ending,-1))+b.greedy(draw_scores(b,ending,1)))/2
                    for name,p in (("random_utility_VNM",mixture),("compact_VNM_max",b.greedy(fitted)),
                                   ("compact_VNM_copies_max",b.greedy(new_score)),
                                   ("supplied_VNM_copies_max",b.greedy(score_tables["supplied_VNM_copies"])),
                                   ("cell_frequency_max",b.greedy(cell_order)),
                                   ("compact_VNM_masked",soft(b,fitted,.25,source>0))):
                        measures.append(ev.measure(p,test,method=name,temperature=None,selected=False,
                                        utility_draw=new_draw if name=="compact_VNM_copies_max" else d if name.startswith("compact") else None,
                                        equivalent_draws=new_equal if name=="compact_VNM_copies_max" else equal if name.startswith("compact") else None,**key))
    np.savez_compressed(tmp/"intervention_cases.npz",**cases)
    dump(tmp/"soft.jsonl.gz",measures)
    dump(tmp/"interventions.jsonl.gz",interventions)
    dump(tmp/"lotteries.jsonl.gz",lotteries)

