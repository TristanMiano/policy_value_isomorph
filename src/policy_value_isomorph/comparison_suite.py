"""Paired comparisons. Scores used to select actions are always oriented for the mover."""
from dataclasses import dataclass
from itertools import combinations
from math import fsum, inf, isclose, isfinite, log, sqrt
from statistics import mean
from typing import Callable, Mapping

from .checkpointing import load_checkpoint
from .pilot_exact import play_match, total_variation
from .recovery_contract import ValueArtifact
from .score_alignment import align, cardinal_mse


# Meaning, units, population, invariance, required access travel with metric exports.
METRICS = {
    "agreement": ("deployed probability of the deterministic source's chosen move", "fraction", "weighted decisions with a fixed source choice", "equivalent deployed laws", "source and deployed laws"),
    "tv": ("minimum probability mass reassigned", "fraction", "weighted decisions", "equivalent laws", "source and deployed laws"),
    "cross_entropy": ("negative log probability of a source draw", "nats per draw", "weighted decisions", "equivalent laws", "source and deployed laws"),
    "kl": ("extra natural-log loss relative to the source's own probabilities", "nats per draw", "weighted decisions", "equivalent laws", "both legal probability rows"),
    "game": ("win, draw, loss and win minus loss", "probability and points per game", "declared opponents and seats from opening", "equivalent deployed laws", "rules and both complete players"),
    "ranks": ("within-menu rank correlation and top-k overlap", "correlation and fraction", "matched legal actions", "strictly increasing score transformations", "both score rows and tie priority"),
    "cardinal": ("mean squared score difference", "squared aligned units", "common entries", "declared common gauge", "aligned scores"),
    "geometry": ("mean squared error of changes in move gaps", "squared aligned units", "common move pairs across neighboring histories", "additive shifts within rows", "neighbors, matching moves and shared score scale"),
    "calibration": ("mean prediction, mean outcome and squared residual", "points and squared points", "fixed target entries per bin", "joint change of payoff perspective for squared error", "paired predictions and outcomes"),
    "intervention": ("distance to fresh choice under changed conditions", "probability mass", "equal observed families and equal observations within family", "equivalent laws", "fresh source responses"),
    "gap": ("largest score minus runner-up", "mover score units", "multi-move decisions", "common additive shift", "all legal scores"),
}


def _law(row, menu):
    if set(row).difference(menu) or any(not isfinite(v) or v<0 for v in row.values()) or not isclose(sum(row.values()),1,abs_tol=1e-12):
        raise ValueError("need normalized probabilities on legal actions")


def choice_metrics(original, recovered, menu, scores, *, tie_order=None) -> dict:
    """Rank equal scores by the supplied priority, defaulting to the offered menu's order."""
    legal = tuple(menu)
    menu = legal if tie_order is None else tuple(tie_order)
    if len(menu) != len(set(menu)) or set(menu) != set(legal):
        raise ValueError("tie priority must list each legal move once")
    _law(original,menu)
    _law(recovered,menu)
    if (not menu or set(scores)!=set(menu) or max(scores.values()) == -inf
            or any(not isfinite(s) and s != -inf for s in scores.values())):
        raise ValueError("provide legal mover scores, using negative infinity only for excluded moves")
    ranked = sorted(menu,key=scores.__getitem__,reverse=True)
    best = ranked[0]
    tied = [a for a in ranked if scores[a]==scores[best]]
    missing = sum(p for a,p in original.items() if recovered.get(a,0)==0)
    ce = inf if missing else -fsum(float(p)*log(recovered[a]) for a,p in original.items() if p)
    point = [a for a,p in original.items() if p==1]
    entropy = -fsum(float(p)*log(float(p)) for p in original.values() if p)
    return {"tv":total_variation(original,recovered),"cross_entropy":ce,"kl":ce-entropy,"missing_mass":float(missing),
            "agreement":float(recovered.get(point[0],0)) if point else None,
            "ties":len(tied),"gap":scores[best]-scores[ranked[1]] if len(ranked)>1 else None,
            "source_best_mass":float(sum(original.get(a,0) for a in tied)),
            "source_fixed_match":float(original.get(best,0)),
            "source_score_forgone":fsum(float(p)*(scores[best]-scores[a]) for a,p in original.items() if p)}


def ranking(left: Mapping, right: Mapping, k: int) -> dict:
    if left.keys()!=right.keys() or not 1<=k<=len(left):
        raise ValueError("need the same menu and a valid k")
    keys = tuple(left)
    ranks = [align(row,"rank",units="rank",reference="common menu").values for row in (left,right)]
    center = (len(keys)-1)/2
    x,y = [[r[a]-center for a in keys] for r in ranks]
    norm = sqrt(sum(z*z for z in x)*sum(z*z for z in y))
    tops = [set(sorted(keys,key=row.__getitem__,reverse=True)[:k]) for row in (left,right)]
    reversals = ties_left = ties_right = 0
    for a,b in combinations(keys,2):
        d,e = left[a]-left[b],right[a]-right[b]
        reversals += d*e < 0
        ties_left += d==0
        ties_right += e==0
    return {"spearman":sum(a*b for a,b in zip(x,y))/norm if norm else None,
            "top_k_overlap":len(tops[0]&tops[1])/k,"k":k,"pairs":len(keys)*(len(keys)-1)//2,
            "reversals":reversals,"ties_left":ties_left,"ties_right":ties_right}


def geometry(left: Mapping, right: Mapping, neighbors) -> dict:
    errors = []
    for s,t in neighbors:
        rows = (left[s],left[t],right[s],right[t])
        if len({r.gauge for r in rows})!=1 or any(not r.cardinal for r in rows):
            raise ValueError("neighbor gap changes need common aligned units and perspective")
        common = set.intersection(*(set(row.values) for row in rows))
        for a,b in combinations(common,2):
            u,v,w,x = [row.values[a]-row.values[b] for row in rows]
            errors.append(((v-u)-(x-w))**2)
    return {"move_pairs":len(errors),"mse":mean(errors) if errors else None}


def calibration(pairs, edges=()) -> list[dict]:
    """A bin starts at each edge. Preserve empty bins and the unbinned population."""
    from bisect import bisect_right
    edges = tuple(edges)
    if tuple(sorted(set(edges)))!=edges or not all(isfinite(x) for x in edges):
        raise ValueError("bin edges must increase")
    rows = list(pairs)
    if any(not isfinite(x) or not isfinite(y) for x,y in rows):
        raise ValueError("calibration needs finite paired values")
    bins = [[] for _ in range(len(edges)+1)]
    for x,y in rows:
        bins[bisect_right(edges,x)].append((x,y))
    def summary(group):
        return {"entries":len(group),"prediction":mean(x for x,_ in group) if group else None,
                "target":mean(y for _,y in group) if group else None,
                "mse":mean((x-y)**2 for x,y in group) if group else None}
    return [{"population":"all",**summary(rows)}, *[
        {"lower":lo,"upper":hi,**summary(group)} for lo,hi,group in zip((None,*edges),(*edges,None),bins)]]


def changed_choices(families: Mapping) -> dict:
    """Each pair is (fresh source law, predicted law), under the same intervention."""
    result = {}
    for family, observations in families.items():
        values = []
        for p,q in observations:
            menu = p.keys()|q.keys()
            _law(p,menu)
            _law(q,menu)
            values.append(total_variation(p,q))
        result[family] = {"observations":len(values),"mean_tv":mean(values) if values else None}
    available = [r["mean_tv"] for r in result.values() if r["observations"]]
    return {"families":result,"equally_weighted_observed_families":mean(available) if available else None}


@dataclass
class GameMethod:
    artifact: ValueArtifact
    law: Mapping
    mover_scores: Mapping
    tie_order: Callable | None = None


def compare_game(graph, source: GameMethod, methods, populations, opponents) -> dict:
    """Exact tic-tac-toe games. Opponents map names to (panel weight, full decision law).
    Populations map names to nonnegative decision-board weights. Total weight may be zero.
    """
    if not opponents or "source head-to-head" in opponents or "panel" in opponents:
        raise ValueError("provide a named opponent panel")
    weights = [w for w,_ in opponents.values()]
    if any(w<0 or not isfinite(w) for w in weights) or not isclose(sum(weights),1,abs_tol=1e-12):
        raise ValueError("panel weights must sum to one")
    records, games, decisions = [], [], []
    baselines = {}
    for index, method in enumerate([source,*methods]):
        if not method.artifact.contract.game or any(getattr(method.artifact.contract,key)!=getattr(source.artifact.contract,key)
                for key in ("domain","source","split","opponent_panel")):
            raise ValueError("match the game, frozen source, split and opponent manifest")
        key = method.artifact.evaluation_key()
        results = {}
        for name,(_,opp) in {**opponents,"source head-to-head":(0,source.law)}.items():
            results[name,"first"] = play_match(graph,method.law,opp)
            results[name,"second"] = play_match(graph,opp,method.law)[::-1]
        for seat in ("first","second"):
            results["panel",seat] = tuple(fsum(w*results[name,seat][i] for name,(w,_) in opponents.items()) for i in range(3))
        for name in (*opponents,"source head-to-head","panel"):
            results[name,"balanced"] = tuple((x+y)/2 for x,y in zip(results[name,"first"],results[name,"second"]))
        for pair,wdl in results.items():
            values = dict(zip(("win","draw","loss"),wdl),points=wdl[0]-wdl[2])
            if index==0:
                baselines[pair] = values
            games.append({**key,"opponent":pair[0],"seat":pair[1],**values,
                          **{"delta_"+k:v-baselines[pair][k] for k,v in values.items()}})
        for group, board_weights in populations.items():
            if any(w<0 or not isfinite(w) for w in board_weights.values()):
                raise ValueError("invalid board weights")
            total = sum(board_weights.values())
            rows = []
            for s,w in board_weights.items():
                priority = method.tie_order(s) if method.tie_order is not None else None
                row = choice_metrics(source.law[s],method.law[s],graph.children[s],method.mover_scores[s],
                                     tie_order=priority)
                rows.append((row,w))
                decisions.append({**key,"population_name":group,"board":tuple(s.board),"to_move":s.to_move,
                                  "weight":w,**row})
            avg = {}
            for stat in ("tv","agreement","cross_entropy","kl","missing_mass","gap","source_score_forgone"):
                defined = [(r[stat],w) for r,w in rows if r[stat] is not None and w>0]
                mass = sum(w for _,w in defined)
                avg[stat] = fsum(v*w for v,w in defined)/mass if mass else None
                avg[stat+"_weight"] = mass
            if index==0:
                baselines[group] = avg
            for stat in ("tv","agreement"):
                v,b=avg[stat],baselines[group][stat]
                avg["delta_"+stat] = v-b if v is not None and b is not None else None
            panel = next(row for row in reversed(games) if row["opponent"]=="panel" and row["seat"]=="balanced")
            records.append({**key,**{k:panel[k] for k in ("win","draw","loss","points","delta_win",
                            "delta_draw","delta_loss","delta_points")},"population_name":group,"boards":len(rows),"weight_sum":total,
                            "positive_weight_boards":sum(w>0 for _,w in rows),
                            "boards_missing_mass":sum(r["missing_mass"]>0 for r,_ in rows),**avg})
    return {"definitions":METRICS,"fidelity":records,"game_outcomes":games,"decisions":decisions}


def return_checkpoints(paths, model_type, predict, training_targets, heldout_exact, build_method,
                       graph, source, populations, opponents):
    """Replay saved public weights on the same target entries, boards and full games at every step.
    predict(model,key) returns fixed-perspective points. Targets are sequences and may repeat.
    """
    training, heldout = tuple(training_targets),tuple(heldout_exact)
    if not training or not heldout:
        raise ValueError("declare both target populations")
    for path in paths:
        saved = load_checkpoint(path,model_type=model_type)
        model = saved["model"]
        fit = build_method(model)
        result = compare_game(graph,source,[fit],populations,opponents)
        residuals = {name:calibration((predict(model,k),y) for k,y in pairs)[0]
                     for name,pairs in (("training",training),("heldout_exact",heldout))}
        yield {"path":str(path),"step":saved["step"],"return_error":residuals,**result}
