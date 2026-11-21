"""Reproducible offered moves and reachable neighborhoods, with coverage accounting."""
from collections import deque
from dataclasses import dataclass
from math import fsum, isclose, isfinite
from random import Random
from typing import Callable, Generic, Mapping, Sequence, TypeVar

A = TypeVar("A")


@dataclass(frozen=True)
class Candidates(Generic[A]):
    values: tuple[A, ...]
    generator: str
    proposals: int
    seed: int | None

    def coverage(self, source: Mapping[A,float]) -> dict:
        """Missing source probability is a lower bound on mass that any restricted choice must move."""
        if any(not isfinite(p) or p<0 for p in source.values()) or not isclose(fsum(source.values()),1,abs_tol=1e-12):
            raise ValueError("supply a normalized reference distribution on the full action space")
        offered = set(self.values)
        omitted = fsum(p for a,p in source.items() if a not in offered)
        return {"proposals":self.proposals,"distinct":len(offered),"omitted_probability":omitted,
                "covered_positive_moves":sum(p>0 and a in offered for a,p in source.items()),
                "reference_positive_moves":sum(p>0 for p in source.values())}

    def closest_distance(self, target: A, distance: Callable[[A,A],float]) -> float | None:
        distances = [distance(target,a) for a in self.values]
        if any(not isfinite(x) or x<0 for x in distances):
            raise ValueError("invalid action distance")
        return min(distances,default=None)


def legal_candidates(adapter, history, *, limit: int | None = None, seed: int = 0) -> Candidates:
    legal = tuple(adapter.candidates(history))
    if len(set(legal)) != len(legal) or (limit is not None and limit < 0):
        raise ValueError("need distinct legal moves and a nonnegative limit")
    if limit is None or limit >= len(legal):
        return Candidates(legal,"all legal moves",len(legal),None)
    chosen = set(Random(seed).sample(legal,limit))
    return Candidates(tuple(a for a in legal if a in chosen),"uniform legal subset",limit,seed)


def reachable(adapter, history, *, depth: int) -> dict:
    """Distinct legal successor histories by shortest number of transitions, excluding the start."""
    if depth < 0:
        raise ValueError("negative depth")
    distances = {history:0}
    queue = deque([history])
    while queue:
        parent = queue.popleft()
        if distances[parent] == depth:
            continue
        for action in adapter.candidates(parent):
            child = adapter.transition(parent,action)
            if child not in distances:
                distances[child] = distances[parent]+1
                queue.append(child)
    del distances[history]
    return distances


def sample_continuous(bounds: Sequence[tuple[float,float]], *, draws: int, seed: int) -> Candidates:
    """Independent uniform coordinates over a finite box, one tuple per draw."""
    if draws < 0 or not bounds or any(not isfinite(lo) or not isfinite(hi) or lo>hi for lo,hi in bounds):
        raise ValueError("need finite ordered bounds and a nonnegative budget")
    rng = Random(seed)
    proposals = [tuple(rng.uniform(lo,hi) for lo,hi in bounds) for _ in range(draws)]
    return Candidates(tuple(dict.fromkeys(proposals)),"uniform box",draws,seed)


def sequence_candidates(prefix: tuple, propose: Callable[[tuple,Random], A], *, draws: int,
                        seed: int, generator: str) -> Candidates[A]:
    """Repeated proposals conditional on the prefix. Count repetitions before deduplication."""
    if draws < 0 or not generator.strip():
        raise ValueError("declare a generator and a nonnegative budget")
    rng = Random(seed)
    return Candidates(tuple(dict.fromkeys(propose(prefix,rng) for _ in range(draws))),
                      generator,draws,seed)
