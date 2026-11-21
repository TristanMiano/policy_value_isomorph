"""Fresh source responses to offered menus. Move numbers have no built-in preference."""
from dataclasses import asdict, dataclass
from itertools import combinations
from math import exp, sqrt
from random import Random

import torch

from .recovery_contract import RecoveryContract, ValueArtifact


@dataclass(frozen=True)
class Query:
    menu: tuple[int, ...]
    choice: int


@dataclass(frozen=True)
class Probe:
    context: str
    moves: tuple[int, ...]
    strategy: str
    budget: int
    seed: int
    history: tuple[Query, ...]

    def artifact(self, contract: RecoveryContract) -> ValueArtifact:
        record = asdict(self)
        record["moves"] = list(self.moves)
        record["history"] = [{"menu":list(q.menu), "choice":q.choice} for q in self.history]
        return ValueArtifact(contract, self.strategy, "fresh constrained source calls",
                             "one observed response per recorded query", record)

    def order_diagnostics(self) -> dict:
        better = {a:set() for a in self.moves}
        for q in self.history:
            better[q.choice].update(set(q.menu)-{q.choice})
        for k in self.moves:
            for a in self.moves:
                if k in better[a]:
                    better[a].update(better[k])
        determined, conflicts, unknown = [], [], []
        for a,b in combinations(self.moves,2):
            ab, ba = b in better[a], a in better[b]
            if ab and ba:
                conflicts.append((a,b))
            elif ab or ba:
                determined.append((a,b) if ab else (b,a))
            else:
                unknown.append((a,b))
        return {"ordered_pairs":determined, "conflicting_pairs":conflicts, "undetermined_pairs":unknown,
                "queries_used":len(self.history)}


def probe(adapter, state, context: str, *, strategy: str, budget: int, seed: int,
          check_menus=()) -> Probe:
    """Eliminate selected moves, or query pairs. Optional check menus consume the same budget first."""
    moves, rng = tuple(adapter.candidates(state)), Random(seed)
    if not context or not moves or budget < 0 or strategy not in {"elimination","active pairs","random pairs"}:
        raise ValueError("invalid probing context, strategy or budget")
    history = []
    def ask(menu):
        winner = adapter.constrained(state,menu,rng)
        history.append(Query(tuple(menu),winner))
        return winner
    for menu in check_menus:
        if len(history) == budget:
            break
        ask(tuple(menu))
    remaining = list(moves)
    pairs = list(combinations(moves,2))
    # A balanced design covers every pair before revisiting one. Among equal counts,
    # query the pair whose smoothed win fraction is closest to one half.
    counts = {pair:[0,0] for pair in pairs}
    for q in history:
        if len(q.menu) == 2:
            key = next(p for p in pairs if set(p)==set(q.menu))
            counts[key][key.index(q.choice)] += 1
    while len(history) < budget and pairs:
        if strategy == "elimination":
            if len(remaining) == 1:
                break
            remaining.remove(ask(tuple(remaining)))
        else:
            if strategy == "active pairs":
                pair = min(pairs,key=lambda p:(sum(counts[p]),abs(counts[p][0]-counts[p][1])))
            else:
                pair = rng.choice(pairs)
            winner = ask(pair)
            counts[pair][pair.index(winner)] += 1
    return Probe(context,moves,strategy,budget,seed,tuple(history))


def pair_model(record: Probe, penalty: float = 1.0) -> dict:
    """Fit pair win chances by logistic score differences with a quadratic penalty.
    Report empirical pair counts and 95% Wilson intervals under independent repeated draws.
    """
    if not 0 < penalty < float("inf"):
        raise ValueError("use a positive finite penalty for sparse or unanimous comparisons")
    rows = [q for q in record.history if len(q.menu)==2]
    if not rows:
        raise ValueError("no pair responses")
    moves = record.moves
    wins = torch.tensor([moves.index(q.choice) for q in rows])
    losses = torch.tensor([moves.index(next(a for a in q.menu if a!=q.choice)) for q in rows])
    scores = torch.zeros(len(moves),dtype=torch.float64,requires_grad=True)
    opt = torch.optim.LBFGS([scores],max_iter=80,line_search_fn="strong_wolfe")
    def objective():
        opt.zero_grad()
        loss = torch.nn.functional.softplus(scores[losses]-scores[wins]).sum() + penalty*scores.square().sum()/2
        loss.backward()
        return loss
    opt.step(objective)
    values = scores.detach().tolist()
    result = []
    for i,j in combinations(range(len(moves)),2):
        a,b = moves[i],moves[j]
        observed = [q.choice for q in rows if set(q.menu)=={a,b}]
        n, k = len(observed), observed.count(a)
        interval = None
        if n:
            z = 1.959963984540054
            p = k/n
            center = (p + z*z/(2*n))/(1+z*z/n)
            radius = z*sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
            interval = (max(0,center-radius),min(1,center+radius))
        result.append({"a":a,"b":b,"draws":n,"a_wins":k,"interval":interval,
                       "model_probability_a":1/(1+exp(values[j]-values[i]))})
    return {"scores":dict(zip(moves,values)), "penalty":penalty, "pairs":result}
