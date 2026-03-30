from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List

from typing import Callable
from .tictactoe import Move, TicTacToeState

PolicyFn = Callable[[TicTacToeState], Move]


@dataclass(frozen=True)
class StateActionSample:
    """Single state-action sample from a tic-tac-toe trajectory."""

    state: TicTacToeState
    action: Move


def _collect_episode_samples(policy: PolicyFn) -> List[StateActionSample]:
    cur = TicTacToeState.initial()
    samples: List[StateActionSample] = []

    while not cur.is_terminal():
        action = policy(cur)
        samples.append(StateActionSample(state=cur, action=action))
        cur = cur.apply_move(action)

    return samples


def generate_on_policy_dataset(policy: PolicyFn, n_episodes: int) -> List[StateActionSample]:
    """Generate state-action samples by rolling out the provided policy.

    The same `policy` is used for both players. Samples contain non-terminal
    states paired with the action selected by the policy at that state.
    """
    if n_episodes <= 0:
        raise ValueError("n_episodes must be >= 1")

    dataset: List[StateActionSample] = []
    for _ in range(n_episodes):
        dataset.extend(_collect_episode_samples(policy))
    return dataset


def generate_off_policy_dataset(n_episodes: int, seed: int | None = None) -> List[StateActionSample]:
    """Generate off-policy samples using uniformly random legal actions."""
    if n_episodes <= 0:
        raise ValueError("n_episodes must be >= 1")

    rng = random.Random(seed)

    def random_policy(state: TicTacToeState) -> Move:
        legal = state.legal_moves()
        if not legal:
            raise ValueError("random_policy called on terminal state")
        return rng.choice(legal)

    dataset: List[StateActionSample] = []
    for _ in range(n_episodes):
        dataset.extend(_collect_episode_samples(random_policy))
    return dataset
