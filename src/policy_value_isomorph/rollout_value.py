from __future__ import annotations

from typing import Callable

from .tictactoe import Move, TicTacToeState

PolicyFn = Callable[[TicTacToeState], Move]


def rollout_return(state: TicTacToeState, policy: PolicyFn, root_player: int) -> int:
    """Play from `state` until terminal using `policy` for both players.

    Returns utility in {-1,0,+1} from fixed `root_player` perspective.
    """
    cur = state
    while not cur.is_terminal():
        move = policy(cur)
        cur = cur.apply_move(move)
    return cur.terminal_return(root_player=root_player)


def estimate_v_pi(
    state: TicTacToeState,
    policy: PolicyFn,
    root_player: int,
    n_rollouts: int = 1,
) -> float:
    """Monte-Carlo estimate of V^pi(state) from fixed root-player perspective."""
    if n_rollouts <= 0:
        raise ValueError("n_rollouts must be >= 1")
    total = 0.0
    for _ in range(n_rollouts):
        total += rollout_return(state, policy=policy, root_player=root_player)
    return total / n_rollouts


def recovered_action_from_v(
    state: TicTacToeState,
    policy: PolicyFn,
    root_player: int,
    n_rollouts: int = 1,
) -> Move:
    """Choose action using successor-state values induced by `policy`.

    Sign/perspective convention:
    - V^pi(s) is always from fixed `root_player` viewpoint.
    - If it is root player's turn, choose argmax over successor values.
    - If it is opponent's turn, choose argmin (opponent minimizes root payoff).
    """
    legal = state.legal_moves()
    if not legal:
        raise ValueError("recovered action called on terminal state")

    scored = []
    for mv in legal:
        nxt = state.apply_move(mv)
        v = estimate_v_pi(nxt, policy=policy, root_player=root_player, n_rollouts=n_rollouts)
        scored.append((mv, v))

    if state.to_move == root_player:
        return max(scored, key=lambda t: (t[1], -t[0]))[0]
    return min(scored, key=lambda t: (t[1], t[0]))[0]
