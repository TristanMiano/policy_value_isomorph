"""Inspectable per-context views. Each view compares repeats of one declared formulation."""
from dataclasses import dataclass
from math import inf, isfinite
from statistics import mean, stdev
from typing import Callable, Mapping, Sequence

from .candidate_sets import reachable
from .comparison_suite import choice_metrics
from .policy_adapter import PolicyAdapter
from .recovery_contract import ValueArtifact


@dataclass(frozen=True)
class AtlasRecovery:
    artifact: ValueArtifact
    scores: Callable
    probabilities: Callable
    visit_probabilities: Callable | None = None
    tie_order: Callable | None = None


@dataclass
class ValueAtlas:
    adapter: PolicyAdapter
    source_law: Callable
    recoveries: Sequence[AtlasRecovery]

    def inspect(self, history, *, depth: int = 1, action_neighbors: Callable | None = None,
                visits: Mapping | None = None) -> dict:
        """Scores use the declared mover's units, or positive distances for minimum energy.
        Standard deviations describe the supplied repeats, conditional on their shared data.
        """
        menu = tuple(self.adapter.candidates(history))
        if not menu or not self.recoveries:
            raise ValueError("inspection needs a decision and at least one recovery")
        first = self.recoveries[0].artifact
        original = self.source_law(history)
        if visits is not None and (set(visits)!=set(menu) or any(type(n) is not int or n<0 for n in visits.values())):
            raise ValueError("provide all legal observed counts at this board")
        raw, laws, diagnostics, predictions = [], [], [], []
        for fit in self.recoveries:
            if fit.artifact.contract != first.contract or fit.artifact.method != first.method:
                raise ValueError("compare repeats of the same method, score units and deployment contract")
            row = dict(fit.scores(history))
            oriented = ({a:-v for a,v in row.items()} if first.contract.decoder=="minimum energy" else row)
            law = dict(fit.probabilities(history))
            priority = fit.tie_order(history) if fit.tie_order is not None else None
            diagnostics.append(choice_metrics(original,law,menu,oriented,tie_order=priority))
            raw.append(row)
            laws.append(law)
            if fit.visit_probabilities is not None:
                # Independent per-move regressions target fractions of visits. Only
                # the deployed law must sum to one. Preserve raw estimates for diagnosis.
                estimates = fit.visit_probabilities(history)
                if set(estimates).difference(menu) or any(not 0 <= p <= 1 for p in estimates.values()):
                    raise ValueError("visit-frequency predictions must lie in [0, 1] on legal moves")
                predictions.append(estimates)
        if predictions and len(predictions) != len(laws):
            raise ValueError("all repeats must expose the same visit prediction")
        def spread(values):
            return stdev(values) if len(values)>1 and all(isfinite(v) for v in values) else None
        total = sum(visits.values()) if visits is not None else None
        actions = []
        for a in menu:
            values = [float(row[a]) for row in raw]
            probabilities = [float(row.get(a,0)) for row in laws]
            neighbors = tuple(action_neighbors(history,a)) if action_neighbors else tuple(b for b in menu if b!=a)
            if a in neighbors or not set(neighbors).issubset(menu):
                raise ValueError("action neighbors must be other legal candidates")
            actions.append({"move":a,"source_probability":float(original.get(a,0)),
                "observed":visits[a] if visits is not None else None,"visits":total,
                "scores":values,"mean_score":mean(values),"score_sd":spread(values),
                "probabilities":probabilities,"mean_probability":mean(probabilities),
                "probability_sd":spread(probabilities),
                "expected_deployed_count_at_same_visits":None if total is None else total*mean(probabilities),
                "predicted_visit_count":None if total is None or not predictions else
                    total*mean(float(row.get(a,0)) for row in predictions),
                "neighboring_moves":neighbors,"successor":self.adapter.encode(self.adapter.transition(history,a))})
        states = reachable(self.adapter,history,depth=depth)
        return {"history":self.adapter.encode(history),"source":first.contract.source,
                "contract":first.evaluation_key(),"recovery_ids":[fit.artifact.artifact_id for fit in self.recoveries],
                "repeats":len(raw),"variation":"sample standard deviation across supplied fits, undefined for one fit",
                "actions":actions,"per_fit_decisions":diagnostics,
                "neighboring_states":[{"history":self.adapter.encode(s),"distance_in_moves":n} for s,n in states.items()]}
