from __future__ import annotations

from typing import List

from .tictactoe import Move, TicTacToeState


def _find_immediate_winning_move(state: TicTacToeState, player: int) -> Move | None:
    for mv in state.legal_moves():
        next_state = state.apply_move(mv)
        if next_state.winner() == player:
            return mv
    return None


def heuristic_policy_action(state: TicTacToeState) -> Move:
    """Deterministic baseline policy used as frozen pi.

    Priority order:
    1) Win immediately if possible.
    2) Block opponent's immediate win.
    3) Take center.
    4) Take first available corner (0,2,6,8).
    5) Take first remaining legal move.
    """
    legal = state.legal_moves()
    if not legal:
        raise ValueError("policy called on terminal state")

    me = state.to_move
    opp = -me

    win_now = _find_immediate_winning_move(state, me)
    if win_now is not None:
        return win_now

    for mv in legal:
        # If opponent could win by playing mv now, we block mv.
        synthetic_board = list(state.board)
        synthetic_board[mv] = opp
        synthetic = TicTacToeState(board=tuple(synthetic_board), to_move=me)
        if synthetic.winner() == opp:
            return mv

    if 4 in legal:
        return 4

    for corner in (0, 2, 6, 8):
        if corner in legal:
            return corner

    return legal[0]
