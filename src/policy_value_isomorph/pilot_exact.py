"""Exact policy evaluation and gameplay for the frozen tic-tac-toe pilot."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from math import exp, fsum, isclose, log
from typing import Mapping

from .pilot_protocol import GameGraph, Law, State

Distribution = Mapping[int, float | Fraction]
PolicyTable = Mapping[State, Distribution]


@dataclass
class ExactReturns:
    value: dict[State, Fraction]
    action_value: dict[State, dict[int, Fraction]]
    best: dict[State, tuple[int, ...]]
    visitation: dict[State, Fraction]


def evaluate_source(graph: GameGraph, policy: dict[State, Law]) -> ExactReturns:
    """Compute fixed-X V and independently compute direct Q in mover coordinates."""
    value: dict[State, Fraction] = {}
    mover_q: dict[State, dict[int, Fraction]] = {}
    action_value: dict[State, dict[int, Fraction]] = {}
    best: dict[State, tuple[int, ...]] = {}
    for state in reversed(graph.states):
        children = graph.children[state]
        if not children:
            value[state] = Fraction(state.terminal_return(1))
            continue
        law = policy[state]
        if not law or sum(law.values()) != 1 or any(a not in children or p <= 0 for a, p in law.items()):
            raise ValueError(f"invalid source distribution at {state}")
        value[state] = sum(law[a] * value[children[a]] for a in law)
        scores = {}
        for move, child in children.items():
            if graph.children[child]:
                scores[move] = -sum(p * mover_q[child][a] for a, p in policy[child].items())
            else:
                scores[move] = Fraction(child.terminal_return(state.to_move))
        mover_q[state] = scores
        action_value[state] = {a: state.to_move * q for a, q in scores.items()}
        if any(action_value[state][a] != value[child] for a, child in children.items()):
            raise ArithmeticError("independent direct-Q and successor-V evaluations diverged")
        high = max(scores.values())
        best[state] = tuple(a for a in children if scores[a] == high)

    # A board can be reached by multiple move orders. Sum all incoming path masses.
    visitation = dict.fromkeys(graph.states, Fraction(0))
    visitation[State.initial()] = Fraction(1)
    for state in graph.states:
        if graph.children[state] and visitation[state]:
            for a, p in policy[state].items():
                visitation[graph.children[state][a]] += visitation[state] * p
    return ExactReturns(value, action_value, best, visitation)


def decode_behavior(policy: dict[State, Law]) -> dict[State, Law]:
    """Verify legal log-score decoding and retain its exact rational distribution."""
    decoded = {}
    for state, law in policy.items():
        # Absent actions have zero mass (log score -infinity).
        weights = {a: exp(log(float(p))) for a, p in law.items()}
        normalizer = fsum(weights.values())
        if any(not isclose(weights[a] / normalizer, float(p), abs_tol=1e-15) for a, p in law.items()):
            raise ArithmeticError("behavioral decoder failed to reproduce the source")
        decoded[state] = dict(law)
    return decoded


def total_variation(p: Distribution, q: Distribution) -> float:
    return 0.5 * fsum(abs(float(p.get(a, 0)) - float(q.get(a, 0))) for a in p.keys() | q.keys())


def play_match(graph: GameGraph, player_x: PolicyTable, player_o: PolicyTable) -> tuple[float, float, float]:
    """Enumerate W/D/L from X's perspective for a full game from the empty board.

    Transition probabilities are integrated in double precision. No games are sampled.
    """
    @cache
    def outcome(state: State) -> tuple[float, float, float]:
        children = graph.children[state]
        if not children:
            payoff = state.terminal_return(1)
            return float(payoff == 1), float(payoff == 0), float(payoff == -1)
        law = player_x[state] if state.to_move == 1 else player_o[state]
        terms = [(float(p), outcome(children[a])) for a, p in law.items() if p]
        return tuple(fsum(p * wdl[i] for p, wdl in terms) for i in range(3))

    result = outcome(State.initial())
    outcome.cache_clear()
    if abs(fsum(result) - 1) > 1e-12:
        raise ArithmeticError("game outcome probabilities did not sum to one")
    return result
