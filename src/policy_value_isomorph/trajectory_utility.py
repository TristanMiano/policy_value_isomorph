"""Empirical move counts and sampled expectations of learned trajectory utilities."""

from __future__ import annotations

import numpy as np

from .pilot_protocol import GameGraph, Law, State


class MoveSampler:
    """Sample complete plays under a frozen policy. Edge numbers index legal (board, move) pairs."""

    def __init__(self, graph: GameGraph, laws: dict[State, Law]) -> None:
        self.states = graph.states
        ids = {s: i for i, s in enumerate(self.states)}
        self.edges = [(s, a) for s in self.states for a in graph.children[s]]
        self.parent = np.array([ids[s] for s, _ in self.edges])
        self.child = np.array([ids[graph.children[s][a]] for s, a in self.edges])
        self.sign = np.array([s.to_move for s, _ in self.edges])
        self.legal = np.full((len(self.states), 9), -1, dtype=int)
        probs = np.zeros_like(self.legal, dtype=float)
        for i, (s, a) in enumerate(self.edges):
            self.legal[self.parent[i], a] = i
            probs[self.parent[i], a] = float(laws[s].get(a, 0))
        self.active = np.any(self.legal >= 0, axis=1)
        self.randomized = any(len(p) > 1 for p in laws.values())
        self.cdf = probs.cumsum(axis=1)
        # End each distribution at exactly one, including trailing zero-probability cells.
        last_supported = 8 - (probs[:, ::-1] > 0).argmax(axis=1)
        self.cdf = np.where(np.arange(9) >= last_supported[:, None], 1.0, self.cdf)
        self.fixed_move = probs.argmax(axis=1)

    def draw_edges(self, states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.randomized:
            actions = (rng.random(len(states))[:, None] >= self.cdf[states]).sum(axis=1)
        else:
            actions = self.fixed_move[states]
        return self.legal[states, actions]

    def visits(self, start_ids: np.ndarray, repetitions: int, seed: int) -> np.ndarray:
        """Count each immediate transition actually taken, through to the end of every game."""
        rng = np.random.default_rng(seed)
        counts = np.zeros(len(self.edges), dtype=np.int64)
        states = np.repeat(start_ids[self.active[start_ids]], repetitions)
        while len(states):
            edges = self.draw_edges(states, rng)
            counts += np.bincount(edges, minlength=len(counts))
            children = self.child[edges]
            states = children[self.active[children]]
        return counts

    def frequencies(self, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """C(s,a) / sum_b C(s,b); unvisited parent boards have undefined frequencies."""
        totals = np.bincount(self.parent, weights=counts, minlength=len(self.states))[self.parent]
        return np.divide(counts, totals, out=np.full(len(counts), np.nan), where=totals > 0), totals

    def rollout_paths(self, repetitions: int, seed: int) -> list[np.ndarray]:
        """Force every candidate first move, then draw from the original on both sides."""
        rng = np.random.default_rng(seed)
        step = np.repeat(np.arange(len(self.edges), dtype=np.int32)[:, None], repetitions, axis=1)
        paths = [step]
        while True:
            children = self.child[step]
            active = (step >= 0) & self.active[children]
            if not active.any():
                return paths
            step = np.full_like(step, -1)
            step[active] = self.draw_edges(children[active], rng)
            paths.append(step)


def trajectory_scores(paths: list[np.ndarray], x_scores: np.ndarray) -> dict:
    """Average complete-path utilities. Inputs and returned scores are from X's perspective.

    Signed sums use every move. Terminal utility uses only the last transition.
    Padding contributes zero.
    """
    total = np.zeros(paths[0].shape)
    discounted = np.zeros_like(total)
    terminal = np.zeros_like(total)
    for t, edges in enumerate(paths):
        mask = edges >= 0
        value = np.where(mask, x_scores[edges], 0)
        total += value
        discounted += 0.9 ** t * value
        terminal = np.where(mask, value, terminal)
    result = {}
    for name, samples in (("vnm_sum", total), ("vnm_decay", discounted), ("vnm_terminal", terminal)):
        mean = samples.mean(axis=1)
        # Deterministic sources have exactly one possible continuation per candidate.
        variance = samples.var(axis=1, ddof=1) / samples.shape[1] if samples.shape[1] > 1 else np.zeros(len(mean))
        result[name] = mean, variance
    return result
