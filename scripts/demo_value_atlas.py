"""Inspect one late tic-tac-toe position and its entire continuation tree."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
from random import Random

import torch

from policy_value_isomorph.candidate_sets import reachable
from policy_value_isomorph.consequence_utility import ExactLotteries, Utility, expectation, sample_lottery
from policy_value_isomorph.distillation import ChoiceExample, fit_scores
from policy_value_isomorph.policy_adapter import GameAdapter
from policy_value_isomorph.preference_probes import probe
from policy_value_isomorph.recovery_contract import Continuation, RecoveryContract, ValueArtifact
from policy_value_isomorph.score_encoding import sample_probabilities
from policy_value_isomorph.tictactoe import state_from_rows
from policy_value_isomorph.value_atlas import AtlasRecovery, ValueAtlas


ROOT = state_from_rows(["XX.","OO.","XO."],1)
ORDER = (5,8,2,0,1,3,4,6,7)


def favorite(menu):
    return next(a for a in ORDER if a in menu)


def source_law(s):
    return {favorite(s.legal_moves()):1}


def uniform(s):
    legal = s.legal_moves()
    return dict.fromkeys(legal,Fraction(1,len(legal)))


def outcomes(law):
    """Deploy every turn in the evaluated seat, against uniform play, from ROOT."""
    seats = []
    for first in (True,False):
        lottery = ExactLotteries(law if first else uniform,uniform if first else law).suffix(ROOT)
        seats.append([sum(p for (points,_),p in lottery.items() if points==(z if first else -z))
                      for z in (1,0,-1)])
    return [float((x+y)/2) for x,y in zip(*seats)]


def experiment():
    name = "first available cell in order 5,8,2,0,1,3,4,6,7"
    source = GameAdapter(name, fixed=lambda s:favorite(s.legal_moves()), law=source_law,
                         choose_menu=lambda s,m,r:favorite(m))
    states = [s for s in (ROOT,*reachable(source,ROOT,depth=3)) if not s.is_terminal()]
    counts = {s:Counter() for s in states}
    for start in states:
        for _ in range(8):
            s = start
            while not s.is_terminal():
                a = source.action(s)
                counts[s][a] += 1
                s = source.transition(s,a)
    labels = {s:tuple(counts[s][a]/counts[s].total() for a in s.legal_moves()) for s in states}
    base = RecoveryContract("tic-tac-toe",name,"behavior","complete source games from each decision",
        "own weights and legal moves","equal reachable decisions from worked position","all legal",
        "direct maximum","lowest cell","probabilities sum to one within board","move probability",
        "all worked positions fitted","descendants of XX.OO.XO.",game=True,
        opponent_panel="uniform legal opponent, equal seats, start at worked position")
    result = {"decision_boards":len(states),"starts":8*len(states),
              "observations":sum(row.total() for row in counts.values()),
              "removed_favorite_predictions":{},
              "source_wdl":outcomes(source_law),"fits":[],"atlases":[],"disagreements":[]}

    def fit_one(network,seed):
        def features(s,a):
            if network=="V":
                return source.encode(s.apply_move(a))
            return (*source.encode(s),*(int(i==a) for i in range(9)))
        examples = [ChoiceExample(tuple(features(s,a) for a in s.legal_moves()),labels[s]) for s in states]
        fitted = fit_scores(examples,"probability",hidden=12,epochs=80,learning_rate=0.03,seed=seed)
        @lru_cache(None)
        def visit(s):
            menu=s.legal_moves()
            logits=fitted.scores([features(s,a) for a in menu])
            return dict(zip(menu,sample_probabilities(logits,[1]*len(menu),1)))
        @lru_cache(None)
        def vnm(s):
            row = {}
            for a in s.legal_moves():
                # The forced move and all later transitions contribute one signed visit score.
                state, move, score = s, a, 0.
                while True:
                    score += state.to_move * visit(state)[move]
                    state = state.apply_move(move)
                    if state.is_terminal():
                        break
                    move = source.action(state)
                row[a] = s.to_move * score
            return row
        payload={"network":network,"seed":seed,"epochs":fitted.epochs,"hidden":fitted.hidden,
                 "step_size":fitted.learning_rate,"loss_nats":fitted.loss_nats,
                 "weights":{k:v.tolist() for k,v in fitted.model.state_dict().items()}}
        methods = []
        for rule, score in (("direct",visit),("VNM",vnm)):
            contract=base if rule=="direct" else replace(base,decoder="VNM maximum",units="signed visit score",
                normalization="sum signed fitted move probabilities along complete play, orient for mover",
                deployment_access="weights, simulator and source copies for both continuations",
                continuation=Continuation("sum of signed fitted move probabilities","tic-tac-toe",
                    name,name,"until terminal","X positive, selection oriented for mover",1,1))
            artifact=ValueArtifact(contract,f"{network} {rule}","worked tree visits",
                                   "two weight starts, complete boards; deterministic continuation",payload)
            def law(s, score=score):
                menu = sorted(s.legal_moves())
                return {max(menu,key=score(s).__getitem__):1}
            fit=AtlasRecovery(artifact,score,law,visit)
            matches=sum(law(s)==source_law(s) for s in states)
            result["fits"].append({"network":network,"rule":rule,"seed":seed,"loss_nats":fitted.loss_nats,
                    "matches":matches,"wdl":outcomes(law)})
            for s in states:
                if law(s)!=source_law(s) and len(result["disagreements"])<2:
                    n=counts[s].total()
                    result["disagreements"].append({"board":s.board,"network":network,"rule":rule,"seed":seed,
                        "legal":s.legal_moves(),"visits":n,"observed":[counts[s][a] for a in s.legal_moves()],
                        "predicted_counts":[n*visit(s)[a] for a in s.legal_moves()],"selection_scores":score(s)})
            methods.append(fit)
        return methods

    for network in ("V","Q"):
        repeats=[fit_one(network,seed) for seed in (13,29)]
        for fits in zip(*repeats):
            atlas=ValueAtlas(source,source_law,fits).inspect(ROOT,visits={a:counts[ROOT][a] for a in ROOT.legal_moves()})
            result["atlases"].append(atlas)
            remaining = [a for a in ROOT.legal_moves() if a!=source.action(ROOT)]
            result["removed_favorite_predictions"][fits[0].artifact.method] = [
                max(remaining,key=fit.scores(ROOT).__getitem__) for fit in fits]
    # Obtain the changed-menu answers only after locking every model prediction above.
    probe_record = probe(source,ROOT,"worked position",strategy="elimination",budget=2,seed=12)
    result["probe"] = [[list(q.menu),q.choice] for q in probe_record.history]
    rnd=GameAdapter("uniform legal moves",law=uniform)
    sampled=sample_lottery(ROOT,5,rnd,rnd,draws=32,rng=Random(71))
    result["payoff_check"]={"forced_move":5,"draws":32,"exact":float(expectation(
        ExactLotteries(uniform,uniform).force(ROOT,5),Utility())),"estimate_and_standard_error":sampled.estimate(Utility())}
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json",action="store_true",help="emit every defined field and per-fit row")
    args=parser.parse_args()
    torch.set_num_threads(1)
    result=experiment()
    if args.json:
        print(json.dumps(result,allow_nan=False))
    else:
        print("Original player picks first available cell in order 5,8,2,0,1,3,4,6,7.")
        print("V scores the resulting board. Q scores the current board and move.")
        print("VNM averages complete-play utility, the signed sum of fitted move probabilities, with source copies finishing both seats.")
        print("Game probabilities give equal weight to both seats against uniform legal moves from the worked position.")
        print("Fitted decisions, games started and observed moves:",result["decision_boards"],result["starts"],result["observations"])
        print("Source win, draw and loss probabilities:",result["source_wdl"])
        print("Network | rule | weight seed | training loss (natural-log units per board) | moves matched | win, draw, loss")
        for row in result["fits"]:
            print(row["network"],row["rule"],row["seed"],round(row["loss_nats"],7),row["matches"],row["wdl"],sep=" | ")
        print("Per-move view at the opening of the example, averaged over two fitted networks")
        for atlas in result["atlases"]:
            for a in atlas["actions"]:
                print(atlas["contract"]["method"],"cell",a["move"],"observed",a["observed"],"of",a["visits"],
                      "predicted count",round(a["predicted_visit_count"],4),"move probability",a["mean_probability"])
        print("Actual restricted-menu questions and answers:",result["probe"])
        print("Frozen predictions after removing the original move:",result["removed_favorite_predictions"])
        print("Supplied game payoff for forced cell 5, both seats uniform:",result["payoff_check"])
        for example in result["disagreements"]:
            print("Fitted-board disagreement",example)
