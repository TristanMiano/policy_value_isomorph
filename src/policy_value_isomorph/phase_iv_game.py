"""Finite tic-tac-toe arrays. Returns are from X's perspective unless marked as mover scores."""

from functools import cache
from pathlib import Path
import os

import numpy as np

from .neural_game_control import policy_tensors
from .pilot_protocol import GameGraph, board_id, minimax_value
from .policy import heuristic_policy_action
from .symmetry import canonicalize_state
from .trajectory_utility import MoveSampler


def seed(block: int, purpose: int, split: int = 0, fit: int = 0) -> int:
    return int(np.random.SeedSequence([6221, block, purpose, split, fit]).generate_state(1)[0])


class Boards:
    def __init__(self):
        self.graph = GameGraph.build(dict(salt="ttt-confirmation-v1", train_end=65, validation_end=80))
        self.states = self.graph.states
        self.n = len(self.states)
        self.dec = np.array([i for i, s in enumerate(self.states) if self.graph.children[s]])
        self.states_dec = [self.states[i] for i in self.dec]
        self.x, self.mask = policy_tensors(self.states_dec)
        self.ply = np.array([9 - s.board.count(0) for s in self.states])
        self.side = 1 - 2 * (self.ply % 2)
        self.layers = [self.dec[self.ply[self.dec] == i] for i in range(9)]
        self.terminal = np.array([s.terminal_return(1) if s.is_terminal() else 0 for s in self.states])
        self.ends = np.array([s.is_terminal() for s in self.states])
        self.names = np.array([board_id(s) for s in self.states])
        self.orbit = np.unique([board_id(canonicalize_state(s)) for s in self.states], return_inverse=True)[
            1
        ]
        idx = {s: i for i, s in enumerate(self.states)}
        self.child = np.zeros((self.n, 9), int)
        self.legal = np.zeros_like(self.child, bool)
        for i in self.dec:
            for a, c in self.graph.children[self.states[i]].items():
                self.child[i, a] = idx[c]
                self.legal[i, a] = True
        self.parent, self.cell = np.where(self.legal)
        self.after = self.child[self.legal]
        self.guarantee = np.array([minimax_value(s) for s in self.states])
        self.optimal = self.side[:, None] * self.guarantee[self.child]
        self.uniform = self.legal / self.legal.sum(1, keepdims=True).clip(1)
        self.perfect = self.greedy(self.optimal)
        self.priority = self.point(
            np.array([heuristic_policy_action(s) if not s.is_terminal() else 0 for s in self.states])
        )

    def point(self, moves):
        law = np.zeros((self.n, 9))
        law[self.dec, moves[self.dec]] = 1
        return law

    def greedy(self, scores):
        return self.point(np.where(self.legal, scores, -np.inf).argmax(1))

    def dense(self, edge):
        scores = np.zeros((self.n, 9), dtype=edge.dtype)
        scores[self.legal] = edge
        return scores

    def sampler(self, law):
        return MoveSampler(
            self.graph, {self.states[i]: {a: law[i, a] for a in np.flatnonzero(law[i])} for i in self.dec}
        )

    def split(self, b, k):
        rng = np.random.default_rng(seed(b, 11, k))
        labels = rng.choice(3, self.orbit.max() + 1, p=[0.65, 0.15, 0.2])
        labels[self.orbit[0]] = 2
        return labels[self.orbit]

    def visitation(self, law):
        reach = np.zeros(self.n)
        reach[0] = 1
        for ids in self.layers:
            np.add.at(reach, self.child[ids].ravel(), (law[ids] * reach[ids, None]).ravel())
        return reach

    def endings(self, x, o):
        """Probability of X win, draw, O win at every state by backward induction."""
        result = np.zeros((self.n, 3))
        result[self.ends] = self.terminal[self.ends, None] == np.array([1, 0, -1])
        for d in range(8, -1, -1):
            ids = self.layers[d]
            law = x if d % 2 == 0 else o
            result[ids] = np.einsum("ij,ijk->ik", law[ids], result[self.child[ids]])
        return result

    def best_reply(self, opp, seat):
        val = self.terminal.astype(float).copy()
        law = self.uniform.copy()
        for d in range(8, -1, -1):
            ids = self.layers[d]
            scores = val[self.child[ids]]
            if self.side[ids[0]] == seat:
                move = np.where(self.legal[ids], seat * scores, -np.inf).argmax(1)
                law[ids] = np.eye(9)[move]
            else:
                law[ids] = opp[ids]
            val[ids] = (law[ids] * scores).sum(1)
        return law


@cache
def game() -> Boards:
    return Boards()


def soft(b: Boards, scores: np.ndarray, temp: float, support: np.ndarray | None = None) -> np.ndarray:
    mask = b.legal if support is None else support
    z = np.where(mask, np.asarray(scores, dtype=float), -np.inf)
    z = z - np.where(mask.any(1), z.max(1), 0)[:, None]
    if temp == 0:
        mass = (z == 0).astype(float)
    else:
        mass = np.exp(z / temp)
    return mass / mass.sum(1, keepdims=True).clip(min=1e-300)


OUT = Path(
    os.environ.get("PVI_PHASE_IV_OUTPUT", Path(__file__).resolve().parents[2] / "experiments" / "phase_iv")
)
