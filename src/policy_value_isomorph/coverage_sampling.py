"""Collect moves from a fixed player while varying starting positions and opponents."""

from dataclasses import dataclass

import numpy as np

from .pilot_protocol import GameGraph, State
from .trajectory_utility import MoveSampler


def enumerated_counts(sampler: MoveSampler, graph: GameGraph, repetitions: int) -> tuple[np.ndarray, int]:
    """Start complete games at every decision board. Keep labels only within the starting data split."""
    if repetitions < 1:
        raise ValueError("Each enumerated board needs at least one start.")
    split = np.array([graph.split[s] for s in graph.states])
    counts = np.zeros(len(sampler.edges), dtype=np.int64)
    total_calls = 0
    for i, group in enumerate(("train", "validation", "test")):
        starts = np.flatnonzero(sampler.active & (split == group))
        visits = sampler.visits(starts, repetitions, 8101 + i)
        mask = split[sampler.parent] == group
        counts[mask] = visits[mask]
        total_calls += int(visits.sum())
    return counts, total_calls


def random_openings(n: int, rng: np.random.Generator) -> tuple[list[State], int]:
    """Draw the move number uniformly, then play random legal prefixes conditional on reaching it."""
    openings = []
    moves_played = 0
    for depth in rng.integers(0, 9, size=n):
        while True:
            s = State.initial()
            for a in rng.permutation(9)[:depth]:
                if s.is_terminal():
                    break
                s = s.apply_move(int(a))
                moves_played += 1
            if not s.is_terminal():
                openings.append(s)
                break
    return openings, moves_played


@dataclass
class Sample:
    counts: np.ndarray
    games_started: int
    random_opponent_turns: int
    extra_source_turns_to_finish: int


def sample_games(
    sampler: MoveSampler, pool: np.ndarray, design: str, budgets: tuple[int, ...], seed: int
) -> dict[int, Sample]:
    """Only the frozen player's actions enter counts. Finish each game, including budget-crossing games."""
    if design not in ("empty_self", "empty_random", "random_openings", "low_visit_openings"):
        raise ValueError(f"Unknown design {design}")
    if not budgets or min(budgets) < 1 or tuple(sorted(set(budgets))) != budgets:
        raise ValueError("Budgets must be increasing positive numbers")
    mixed = design in ("random_openings", "low_visit_openings")
    if mixed and not len(pool):
        raise ValueError("Mixed starts require a pool of legal decision boards")
    schedule, starts, source, opponent = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)
    ]
    empty = sampler.states.index(State.initial())
    counts = np.zeros(len(sampler.edges), dtype=np.int64)
    visits = np.zeros(len(sampler.states), dtype=np.int64)
    legal_edges = [row[row >= 0] for row in sampler.legal]
    snapshots = {}
    calls = games = opposing_turns = 0
    while calls < budgets[-1]:
        state = empty
        random_opponent = design == "empty_random"
        source_seat = 1 if games % 2 == 0 else -1
        if mixed:
            random_opponent = schedule.random() < 0.5
            source_seat = 1 if schedule.random() < 0.5 else -1
            if schedule.random() >= 0.25:
                candidates = starts.choice(pool, size=16)
                state = (
                    candidates[visits[candidates].argmin()]
                    if design == "low_visit_openings"
                    else candidates[0]
                )
                source_seat = sampler.states[state].to_move
        games += 1
        crossed = []
        while sampler.active[state]:
            if random_opponent and sampler.states[state].to_move != source_seat:
                options = legal_edges[state]
                edge = options[opponent.integers(len(options))]
                opposing_turns += 1
            else:
                if sampler.randomized:
                    move = sampler.cdf[state].searchsorted(source.random(), side="right")
                else:
                    move = sampler.fixed_move[state]
                edge = sampler.legal[state, move]
                calls += 1
                counts[edge] += 1
                visits[state] += 1
                if calls in budgets:
                    snapshots[calls] = Sample(counts.copy(), games, opposing_turns, 0)
                    crossed.append(calls)
            state = sampler.child[edge]
        for budget in crossed:
            snapshots[budget].extra_source_turns_to_finish = calls - budget
    return snapshots


def reference_visits(sampler: MoveSampler, random_opponent: bool) -> np.ndarray:
    """Expected source turns at each board per empty-board game, averaging X/O in opposed games."""
    empty = sampler.states.index(State.initial())
    totals = np.zeros(len(sampler.states))
    for seat in (1, -1):
        reach = np.zeros_like(totals)
        reach[empty] = 1
        for i, state in enumerate(sampler.states):
            if not reach[i] or not sampler.active[i]:
                continue
            legal = sampler.legal[i] >= 0
            if not random_opponent or state.to_move == seat:
                totals[i] += reach[i] / 2
                probs = np.diff(sampler.cdf[i], prepend=0)[legal]
            else:
                probs = np.full(legal.sum(), 1 / legal.sum())
            reach[sampler.child[sampler.legal[i, legal]]] += reach[i] * probs
    return totals
