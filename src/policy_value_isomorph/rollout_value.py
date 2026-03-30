from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .policy_mlp import TinyMLPPolicy, policy_mlp_action
from .tictactoe import Move, TicTacToeState

PolicyFn = Callable[[TicTacToeState], Move]


@dataclass(frozen=True)
class StateValueTarget:
    """Monte-Carlo value label for one state under a fixed policy.

    Values are from a fixed root-player perspective and lie in [-1, +1]
    because tic-tac-toe terminal returns are {-1, 0, +1}.
    """

    state: TicTacToeState
    n_rollouts: int
    value: float


def frozen_policy_from_mlp(model: TinyMLPPolicy) -> PolicyFn:
    """Return a deterministic policy function backed by a trained MLP model.

    The returned callable captures model parameters by closure and should be
    treated as a frozen policy during value-target generation.
    """

    def _policy(state: TicTacToeState) -> Move:
        return policy_mlp_action(state, model)

    return _policy


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


def generate_value_targets(
    states: Sequence[TicTacToeState],
    policy: PolicyFn,
    *,
    root_player: int,
    rollout_budgets: Sequence[int],
) -> list[StateValueTarget]:
    """Generate Monte-Carlo value labels for states using configurable budgets.

    For each input state and each entry in `rollout_budgets`, this returns one
    `StateValueTarget` using that rollout count.
    """
    if not states:
        raise ValueError("states must be non-empty")
    if not rollout_budgets:
        raise ValueError("rollout_budgets must be non-empty")

    targets: list[StateValueTarget] = []
    for n_rollouts in rollout_budgets:
        if n_rollouts <= 0:
            raise ValueError("rollout budgets must be >= 1")
        for state in states:
            value = estimate_v_pi(state, policy=policy, root_player=root_player, n_rollouts=n_rollouts)
            targets.append(StateValueTarget(state=state, n_rollouts=n_rollouts, value=value))
    return targets


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
