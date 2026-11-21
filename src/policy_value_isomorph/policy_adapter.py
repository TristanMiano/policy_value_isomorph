"""What an experiment may ask a frozen player. Game rules stay in the game objects."""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isclose, isfinite
from random import Random
from typing import Callable, Generic, Mapping, Protocol, Sequence, TypeVar

from .connect_four import ConnectFourState
from .tictactoe import TicTacToeState

H = TypeVar("H")
A = TypeVar("A")
Board = TypeVar("Board", TicTacToeState, ConnectFourState)


class PolicyAdapter(Protocol[H, A]):
    """Capabilities identify which optional queries supply actual source observations."""
    @property
    def capabilities(self) -> frozenset[str]: ...
    def encode(self, history: H) -> Sequence[float]: ...
    def candidates(self, history: H) -> Sequence[A]: ...
    def action(self, history: H) -> A: ...
    def sample(self, history: H, rng: Random) -> A: ...
    def probabilities(self, history: H) -> Mapping[A, float]: ...
    def logits(self, history: H) -> Mapping[A, float]: ...
    def constrained(self, history: H, menu: Sequence[A], rng: Random) -> A: ...
    def transition(self, history: H, move: A) -> H: ...
    def rollout(self, history: H, move: A, rng: Random, opponent: PolicyAdapter) -> Sequence[H]: ...
    def internals(self, history: H) -> object: ...


@dataclass(frozen=True)
class GameAdapter(Generic[Board]):
    """Wrap either board game and source callbacks without fabricating optional access."""
    name: str
    fixed: Callable[[Board], int] | None = None
    draw: Callable[[Board, Random], int] | None = None
    law: Callable[[Board], Mapping[int, float]] | None = None
    raw_scores: Callable[[Board], Mapping[int, float]] | None = None
    choose_menu: Callable[[Board, tuple[int, ...], Random], int] | None = None
    inspect: Callable[[Board], object] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or all(x is None for x in (self.fixed, self.draw, self.law)):
            raise ValueError("name the player and provide its decision rule")

    @property
    def capabilities(self) -> frozenset[str]:
        optional = {"action": self.fixed, "probabilities": self.law, "logits": self.raw_scores,
                    "constrained": self.choose_menu, "internals": self.inspect}
        return frozenset({"sample", "transition", "rollout"} | {k for k,v in optional.items() if v is not None})

    def encode(self, history: Board) -> tuple[int, ...]:
        return (*history.board, history.to_move)

    def candidates(self, history: Board) -> tuple[int, ...]:
        return tuple(history.legal_moves())

    def _menu(self, history: Board, subset: Sequence[int] | None = None) -> tuple[int, ...]:
        legal = self.candidates(history)
        menu = legal if subset is None else tuple(subset)
        if not menu or len(set(menu)) != len(menu) or not set(menu).issubset(legal):
            raise ValueError("expected a nonempty menu of distinct legal moves")
        return menu

    @staticmethod
    def _checked(move: int, menu: Sequence[int]) -> int:
        if move not in menu:
            raise ValueError("source selected a move outside the offered menu")
        return move

    def action(self, history: Board) -> int:
        if self.fixed is None:
            raise NotImplementedError("no deterministic action query")
        menu = self._menu(history)
        return self._checked(self.fixed(history), menu)

    def sample(self, history: Board, rng: Random) -> int:
        menu = self._menu(history)
        if self.draw is not None:
            return self._checked(self.draw(history, rng), menu)
        if self.law is not None:
            law = self.probabilities(history)
            return rng.choices(menu, weights=[law[a] for a in menu], k=1)[0]
        return self.action(history)

    def probabilities(self, history: Board) -> dict[int, float]:
        if self.law is None:
            raise NotImplementedError("no source probability query")
        menu = self._menu(history)
        values = self.law(history)
        if set(values).difference(menu) or any(not isfinite(p) or p < 0 for p in values.values()):
            raise ValueError("invalid legal probabilities")
        if not isclose(fsum(values.values()), 1, rel_tol=1e-12):
            raise ValueError("probabilities must sum to one")
        return {a: values.get(a, 0) for a in menu}

    def logits(self, history: Board) -> dict[int, float]:
        if self.raw_scores is None:
            raise NotImplementedError("no source logit query")
        menu = self._menu(history)
        row = dict(self.raw_scores(history))
        if set(row) != set(menu) or not all(isfinite(v) or v == -float("inf") for v in row.values()) or max(row.values()) == -float("inf"):
            raise ValueError("expected legal log scores with positive support")
        return row

    def constrained(self, history: Board, menu: Sequence[int], rng: Random) -> int:
        if self.choose_menu is None:
            raise NotImplementedError("no fresh restricted-menu query")
        offered = self._menu(history, menu)
        return self._checked(self.choose_menu(history, offered, rng), offered)

    def transition(self, history: Board, move: int) -> Board:
        return history.apply_move(move)

    def rollout(self, history: Board, move: int, rng: Random, opponent: PolicyAdapter) -> tuple[Board, ...]:
        """Force the first move. Self continues its original seat, opponent the other."""
        path = [history, self.transition(history, move)]
        while not path[-1].is_terminal():
            state = path[-1]
            player = self if state.to_move == history.to_move else opponent
            path.append(self.transition(state, player.sample(state, rng)))
        return tuple(path)

    def internals(self, history: Board) -> object:
        if self.inspect is None:
            raise NotImplementedError("no internal activity query")
        return self.inspect(history)
