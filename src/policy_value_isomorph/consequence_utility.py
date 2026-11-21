"""Expected utilities of terminal consequences, with one shared rule across decision boards."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import product
from math import sqrt
from random import Random
from typing import Callable, Mapping, Sequence

Consequence = tuple[int, int]  # First player's terminal payoff, moves from the forced decision.
Lottery = Mapping[Consequence, Fraction]


@dataclass(frozen=True)
class Utility:
    """Win +1 and loss -1 at equal length offset; draw and points per extra move are shared."""
    draw: Fraction = Fraction(0)
    per_move: Fraction = Fraction(0)

    def __call__(self, consequence: Consequence) -> Fraction:
        payoff, length = consequence
        if payoff not in (-1,0,1) or length < 0:
            raise ValueError("expected a terminal game payoff and nonnegative duration")
        return (self.draw if payoff == 0 else payoff) + self.per_move * length


def expectation(lottery: Lottery, utility: Callable[[Consequence], Fraction]) -> Fraction:
    if sum(lottery.values()) != 1 or any(p < 0 for p in lottery.values()):
        raise ValueError("expected a normalized consequence distribution")
    return sum((p * utility(z) for z,p in lottery.items()), Fraction())


class ExactLotteries:
    """Enumerate reachable continuations in a finite acyclic game under two specified board policies."""
    def __init__(self, first_law: Callable, second_law: Callable):
        def integrate(state):
            if state.is_terminal():
                return {(state.terminal_return(1),0): Fraction(1)}
            legal = state.legal_moves()
            law = (first_law if state.to_move == 1 else second_law)(state)
            if set(law).difference(legal) or any(p < 0 for p in law.values()) or sum(law.values()) != 1:
                raise ValueError("exact enumeration requires normalized rational probabilities on legal moves")
            result = Counter()
            for move, p in law.items():
                if p:
                    for (outcome, length), mass in cached(state.apply_move(move)).items():
                        result[outcome,length+1] += Fraction(p)*mass
            return result
        cached = lru_cache(maxsize=None)(integrate)
        self._suffix = cached

    def suffix(self, state) -> dict[Consequence, Fraction]:
        return dict(self._suffix(state))

    def force(self, state, move: int) -> dict[Consequence, Fraction]:
        return {(outcome,length+1): p for (outcome,length),p in self._suffix(state.apply_move(move)).items()}


@dataclass(frozen=True)
class SampledLotteries:
    distribution: dict[Consequence, Fraction]
    independent_draws: int
    conditional_observations: tuple[Lottery, ...]

    def estimate(self, utility: Callable) -> tuple[float, float | None]:
        """Mean points and standard error across independent rollouts, conditional on any suffix model."""
        values = [expectation(row, utility) for row in self.conditional_observations]
        n = self.independent_draws
        mean = sum(values) / n
        error = sqrt(float(sum((x-mean)**2 for x in values)/(n*(n-1)))) if n > 1 else None
        return float(mean), error


def sample_lottery(state, move: int, first, second, *, draws: int, rng: Random,
                   steps: int | None = None, suffix: Callable | None = None) -> SampledLotteries:
    """Force a move, then sample to termination or replace the remaining play by a supplied lottery.
    steps includes the forced move. The suffix gives remaining duration and fixed-first-seat payoff.
    """
    if draws < 1 or (steps is None) != (suffix is None) or (steps is not None and steps < 1):
        raise ValueError("need positive draws and a suffix model for a truncated rollout")
    child = state.apply_move(move)
    rows = []
    combined = Counter()
    for _ in range(draws):
        current, length = child, 1
        while not current.is_terminal() and (steps is None or length < steps):
            actor = first if current.to_move == 1 else second
            current = current.apply_move(actor.sample(current,rng))
            length += 1
        tail = {(current.terminal_return(1),0): Fraction(1)} if current.is_terminal() else suffix(current)
        row = {(payoff,length+extra): Fraction(p) for (payoff,extra),p in tail.items()}
        expectation(row, Utility())  # Validate support and mass before retaining the sample.
        rows.append(row)
        combined.update(row)
    return SampledLotteries({z: p/draws for z,p in combined.items()}, draws, tuple(rows))


@dataclass(frozen=True)
class UtilityObservation:
    lotteries: Mapping[int, Lottery]
    source_move: int
    perspective: int  # +1 for first player, -1 for the second.


def utility_move(lotteries: Mapping[int, Lottery], utility: Utility, perspective: int) -> int:
    if perspective not in (-1,1) or not lotteries:
        raise ValueError("expected a decision and the mover's sign")
    return max(sorted(lotteries), key=lambda a: perspective * expectation(lotteries[a],utility))


@dataclass(frozen=True)
class UtilityFit:
    best: tuple[Utility, ...]
    matches: tuple[tuple[Utility, int], ...]
    decisions: int
    free_parameters: int
    features: tuple[str, ...] = ("terminal result for first player", "moves from decision to termination")
    tie_rule: str = "lowest move number"


def fit_utility(observations: Sequence[UtilityObservation], draw_values: Sequence,
                length_values: Sequence = (0,)) -> UtilityFit:
    """Maximize equal-board agreement over the declared finite family. Keep every optimum."""
    if not observations or not draw_values or not length_values:
        raise ValueError("need choices and a nonempty parameter grid")
    if any(row.source_move not in row.lotteries for row in observations):
        raise ValueError("source choice outside its menu")
    draws = sorted(set(map(Fraction, draw_values)))
    lengths = sorted(set(map(Fraction, length_values)))
    if draws[0] < -1 or draws[-1] > 1:
        raise ValueError("draw payoffs must lie between the loss and win anchors")
    matches = []
    for d,l in product(draws,lengths):
        u = Utility(d,l)
        count = sum(utility_move(row.lotteries,u,row.perspective) == row.source_move for row in observations)
        matches.append((u,count))
    top = max(n for _,n in matches)
    return UtilityFit(tuple(u for u,n in matches if n==top),tuple(matches),len(observations),
                      int(len(draws)>1)+int(len(lengths)>1))
