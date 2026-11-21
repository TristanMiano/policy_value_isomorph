"""Exact finite witnesses for the Phase II proofs, using only the standard library."""

from collections.abc import Hashable, Iterable, Sequence
from fractions import Fraction
from typing import TypeVar


Vertex = TypeVar("Vertex", bound=Hashable)


def choice_constraints(
    successors: Sequence[Vertex], chosen_index: int, *, maximize: bool = True
) -> list[tuple[Vertex, Vertex, bool]]:
    """Return lower-to-higher comparisons; True means strict. Menu order breaks ties."""
    if not 0 <= chosen_index < len(successors):
        raise ValueError("chosen_index must identify an offered action")
    chosen = successors[chosen_index]
    comparisons = []
    for index, rival in enumerate(successors):
        if index != chosen_index:
            lower, upper = (rival, chosen) if maximize else (chosen, rival)
            comparisons.append((lower, upper, index < chosen_index))
    return comparisons


def successor_levels(
    vertices: Iterable[Vertex], comparisons: Iterable[tuple[Vertex, Vertex, bool]]
) -> dict[Vertex, int] | None:
    """Find integer scores satisfying the comparisons, or None for a strict cycle."""
    levels = dict.fromkeys(vertices, 0)
    edges = list(comparisons)
    for low, high, _ in edges:
        levels.setdefault(low, 0)
        levels.setdefault(high, 0)

    # A feasible system has a longest path of strict increments with at most
    # n-1 edges after removing cycles of equalities. Relax each edge per pass.
    for _ in range(len(levels)):
        changed = False
        for low, high, strict in edges:
            required = levels[low] + int(strict)
            if levels[high] < required:
                levels[high] = required
                changed = True
        if not changed:
            return levels
    return None if levels else {}


def normalized_weights(weights: Sequence[Fraction | int]) -> tuple[Fraction, ...]:
    """Convert nonnegative rational masses of positive total weight to an exact law."""
    total = sum(weights, Fraction())
    if total <= 0 or any(weight < 0 for weight in weights):
        raise ValueError("weights need nonnegative entries and positive total mass")
    return tuple(Fraction(weight) / total for weight in weights)


def _check_law(law: Sequence[Fraction | int]) -> None:
    if sum(law) != 1 or any(mass < 0 for mass in law):
        raise ValueError("probabilities must be nonnegative and sum to one")


def expected_utility(law: Sequence[Fraction | int], utilities: Sequence[Fraction | int]) -> Fraction:
    """Mean utility in the declared player's units for one consequence lottery."""
    _check_law(law)
    return sum((p * u for p, u in zip(law, utilities, strict=True)), Fraction())


def nearest_maximizer_law(
    source: Sequence[Fraction | int], scores: Sequence[Fraction | int]
) -> tuple[Fraction, ...]:
    """Keep source mass at all maxima and move its remaining mass to the first maximum."""
    _check_law(source)
    if len(scores) != len(source):
        raise ValueError("scores and source probabilities must describe the same menu")
    best = max(scores)
    result = [Fraction(p) if score == best else Fraction() for p, score in zip(source, scores)]
    result[scores.index(best)] += 1 - sum(result)
    return tuple(result)
