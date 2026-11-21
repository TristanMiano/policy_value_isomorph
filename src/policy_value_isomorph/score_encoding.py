"""Exact constructions for known policy choices. Ties follow the supplied menu order."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import exp, fsum, inf, isfinite, log
from typing import Callable, Mapping, Sequence, TypeVar

A = TypeVar("A")


def indicator_scores(menu: Sequence[A], choice: A, gap: float = 1) -> dict[A, float]:
    if choice not in menu or len(set(menu)) != len(menu) or not isfinite(gap) or gap <= 0:
        raise ValueError("need a chosen member, distinct moves and positive finite gap")
    return {a: gap if a == choice else 0 for a in menu}


def _log_fraction(x: Fraction) -> float:
    return log(x.numerator) - log(x.denominator)


@dataclass(frozen=True)
class LogEncoding:
    """Ratios are the exact exponential weights before multiplication by the base weights."""
    base: tuple[Fraction, ...]
    ratios: tuple[Fraction, ...]
    temperature: float

    @property
    def scores(self) -> tuple[float, ...]:
        return tuple(self.temperature * _log_fraction(r) if r else -inf for r in self.ratios)

    def exact_probabilities(self) -> tuple[Fraction, ...]:
        masses = [b*r for b, r in zip(self.base, self.ratios, strict=True)]
        total = sum(masses)
        return tuple(m / total for m in masses)


def encode_probabilities(law: Sequence, base: Sequence, temperature: float = 1) -> LogEncoding:
    """Encode exact unit-sum rational masses. Use Fraction for exact nonbinary decimals."""
    p, weights = tuple(map(Fraction, law)), tuple(map(Fraction, base))
    if not p or sum(p) != 1 or min(p) < 0 or not isfinite(temperature) or temperature <= 0:
        raise ValueError("expected a probability distribution and positive finite temperature")
    if len(p) != len(weights) or min(weights) <= 0:
        raise ValueError("each eligible move needs a positive base weight")
    return LogEncoding(weights, tuple(v / w for v, w in zip(p, weights)), temperature)


def sample_probabilities(scores: Sequence[float], base: Sequence[float], temperature: float) -> tuple[float, ...]:
    """Numerical exponential decoder. Negative infinity excludes an action."""
    if not isfinite(temperature) or temperature <= 0 or not scores or len(scores) != len(base):
        raise ValueError("need aligned nonempty scores and weights and a positive temperature")
    if any(not isfinite(w) or w <= 0 for w in base) or any(not isfinite(s) and s != -inf for s in scores):
        raise ValueError("invalid weights or scores")
    high = max(scores)
    if high == -inf:
        raise ValueError("no eligible move")
    logs = [(s-high)/temperature + log(w) for s,w in zip(scores,base)]
    anchor = max(logs)
    masses = [exp(x-anchor) for x in logs]
    total = fsum(masses)
    return tuple(x/total for x in masses)


def distance_energies(menu: Sequence[A], target: A, metric: Callable[[A, A], float]) -> dict[A, float]:
    """Positive distance is an energy to minimize, in the action space's distance units."""
    result = {a: metric(a, target) for a in menu}
    if not result or len(result) != len(menu) or any(v < 0 or not isfinite(v) for v in result.values()):
        raise ValueError("expected distinct candidates with finite nonnegative distances")
    return result


def select(scores: Mapping[A, float], *, minimize: bool = False) -> A:
    """Choose the first extremum in insertion order."""
    if not scores or any(v != v for v in scores.values()):
        raise ValueError("expected a nonempty score row without NaN")
    return (min if minimize else max)(scores, key=scores.__getitem__)
