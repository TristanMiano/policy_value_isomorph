from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .tictactoe import Move, TicTacToeState

PolicyFn = Callable[[TicTacToeState], Move]


@dataclass(frozen=True)
class GameplayEvalRecord:
    """One periodic gameplay evaluation snapshot.

    `score` is win_rate - loss_rate from the evaluated policy perspective.
    """

    step: int
    n_games: int
    win_rate: float
    draw_rate: float
    loss_rate: float
    score: float


def first_legal_policy(state: TicTacToeState) -> Move:
    """Deterministic baseline policy that picks the smallest legal move."""
    legal = state.legal_moves()
    if not legal:
        raise ValueError("first_legal_policy called on terminal state")
    return min(legal)

