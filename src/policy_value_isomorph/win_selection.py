"""Whole-game utilities and continuation models for win-rate comparisons."""

from dataclasses import dataclass
from collections.abc import Callable, Sequence
from itertools import product
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .phase_iv_game import Boards

FINISH = ("copies", "guarded", "mixture", "guarded_mixture", "original")
PRIZES = ("last", "sum", "discounted", "win", "points", "quarter_draw", "hybrid", "direct")


@dataclass(frozen=True)
class Selection:
    network: str
    finish: str
    prize: str
    fitted_ties: bool = False
    budget: int = 4

    @property
    def name(self) -> str:
        name = f"{self.network}/{self.finish}/{self.prize}"
        if self.fitted_ties:
            name += "/fit_ties"
        if self.budget != 4:
            name += f"/paths{self.budget}"
        return name

    @classmethod
    def parse(cls, name: str) -> "Selection":
        network, finish, prize, *flags = name.split("/")
        budget = next((int(f[5:]) for f in flags if f.startswith("paths")), 4)
        return cls(network, finish, prize, "fit_ties" in flags, budget)

    def native_args(self) -> tuple[int, int, bool]:
        return FINISH.index(self.finish), PRIZES.index(self.prize), self.fitted_ties


def candidates() -> list[Selection]:
    return [
        Selection(net, finish, prize) for net, finish, prize in product(("V", "Q"), FINISH[:4], PRIZES[:7])
    ]


def standard_vnm(network: str = "V") -> Selection:
    """The three-game win-rate standard. Q uses the same utility and continuation."""
    if network not in ("V", "Q"):
        raise ValueError("the visit-frequency network must be V or Q")
    return Selection(network, "guarded_mixture", "hybrid", budget=16)


def board_game_player(engine, seed: int, rule: Selection | None = None) -> Callable[[Sequence[int]], int]:
    """Bind the larger-game selector to installed frozen weights, returning a move function."""
    from .phase_v import native

    rule = standard_vnm() if rule is None else rule
    planner = native().WinPlanner(engine, {"V": 1, "Q": 2}[rule.network], seed, rule.budget)
    args = rule.native_args()
    return lambda state: planner.choose(state, *args)


class FinitePlanner:
    """Compute exact candidate expectations on the finite alternating tic-tac-toe graph."""

    def __init__(self, board: "Boards", scores: np.ndarray, original: np.ndarray | None = None):
        self.board = board
        if np.shape(scores) != board.legal.shape or not np.isfinite(np.asarray(scores)[board.legal]).all():
            raise ValueError("supply a finite fitted frequency for every legal candidate on the board graph")
        scores = np.clip(scores, 0, 1)
        self.scores = np.where(board.legal, scores, -np.inf)
        self.direct = board.greedy(self.scores)
        self.original = original
        self.menus = {}
        self.signed = np.where(board.legal, scores * board.side[:, None], 0)
        self.guarded = self.guard()
        self.position = board.greedy(np.broadcast_to([1, 0, 1, 0, 2, 0, 1, 0, 1], scores.shape))

    def guard(self) -> np.ndarray:
        b = self.board
        payoff = b.terminal[b.child] * b.side[:, None]
        winning_replies = ((payoff == 1) & b.legal).sum(1)[b.child]
        best_now = np.where(b.legal, payoff, -9).max(1, keepdims=True)
        allowed = b.legal & (payoff == best_now)
        threats = np.where(allowed, winning_replies, 99)
        return b.greedy(np.where(allowed & (threats == threats.min(1, keepdims=True)), self.scores, -1e20))

    def pair(self, own: np.ndarray, opponent: np.ndarray, seat: int) -> np.ndarray:
        b = self.board
        # Terminal rows store win and draw from the current root player's perspective.
        tail = np.zeros((b.n, 5))
        tail[b.ends, 3] = b.terminal[b.ends] * seat > 0
        tail[b.ends, 4] = b.terminal[b.ends] == 0
        menu = np.zeros((b.n, 9, 5))
        for ids in reversed(b.layers):
            children = b.child[ids]
            values = tail[children].copy()
            signed = self.signed[ids]
            values[:, :, 0] = np.where(b.ends[children], signed, values[:, :, 0])
            values[:, :, 1] += signed
            values[:, :, 2] = signed + 0.9 * values[:, :, 2]
            actor = own if b.side[ids[0]] == seat else opponent
            tail[ids] = np.einsum("ij,ijk->ik", actor[ids], values)
            if b.side[ids[0]] == seat:
                values[:, :, :3] *= seat
                menu[ids] = values
        return menu

    def menu(self, finish: str) -> np.ndarray:
        if finish == "original" and self.original is None:
            raise ValueError("the original-continuation diagnostic requires the original player")
        if finish not in self.menus:
            own = (
                self.original
                if finish == "original"
                else (self.guarded if finish in ("guarded", "guarded_mixture") else self.direct)
            )
            others = (
                (self.direct, self.guarded, self.position, self.board.uniform)
                if "mixture" in finish
                else (own,)
            )
            self.menus[finish] = sum(
                self.pair(own, other, seat) for other in others for seat in (1, -1)
            ) / len(others)
            # Uniform moves have successive denominators drawn from 1..9.
            # A quarter mixture therefore puts exact endpoints on the 1/(4*9!) grid.
            endpoints = self.menus[finish][:, :, 3:]
            endpoints[:] = np.rint(endpoints * 1451520) / 1451520
        return self.menus[finish]

    def law(self, rule: Selection | None = None, *, offered: np.ndarray | None = None) -> np.ndarray:
        """Choose from the offered root moves. Continuation players keep their original legal menus."""
        rule = standard_vnm() if rule is None else rule
        legal = self.board.legal
        if offered is not None:
            if offered.shape != legal.shape:
                raise ValueError("supply a move mask for each board")
            legal = legal & offered
            if not legal[self.board.dec].any(1).all():
                raise ValueError("every decision board needs at least one offered legal move")
        if rule.prize == "direct":
            return (
                self.direct if offered is None else self.board.greedy(np.where(legal, self.scores, -np.inf))
            )
        last, total, decay, win, draw = np.moveaxis(self.menu(rule.finish), 2, 0)
        scores = {
            "last": last,
            "sum": total,
            "discounted": decay,
            "win": win,
            "points": win + 0.5 * draw,
            "quarter_draw": win + 0.25 * draw,
            "hybrid": win + 0.25 * draw + 0.02 * decay,
        }[rule.prize]
        scores = np.where(legal, scores, -np.inf)
        if rule.prize in ("win", "points", "quarter_draw"):
            scores = np.rint(scores * 5806080)
        if rule.fitted_ties:
            scores = np.where(scores == scores.max(1, keepdims=True), self.scores, -np.inf)
        return self.board.greedy(scores)
