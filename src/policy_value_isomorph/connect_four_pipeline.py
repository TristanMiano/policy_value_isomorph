from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Iterable, List, Sequence

from .connect_four import ConnectFourState

PolicyFn = Callable[[ConnectFourState], int]


@dataclass(frozen=True)
class ConnectFourStateActionSample:
    state: ConnectFourState
    action: int


@dataclass(frozen=True)
class ConnectFourStateValueTarget:
    state: ConnectFourState
    value: float
    rollouts: int


def random_policy_action(state: ConnectFourState, *, rng: random.Random | None = None) -> int:
    legal = state.legal_moves()
    if not legal:
        raise ValueError("policy called on terminal state")
    local_rng = rng if rng is not None else random
    return local_rng.choice(legal)


def generate_on_policy_dataset(policy: PolicyFn, n_episodes: int, *, seed: int = 0) -> List[ConnectFourStateActionSample]:
    rng = random.Random(seed)
    samples: List[ConnectFourStateActionSample] = []
    for _ in range(n_episodes):
        s = ConnectFourState.initial()
        while not s.is_terminal():
            a = policy(s)
            samples.append(ConnectFourStateActionSample(state=s, action=a))
            s = s.apply_move(a)
    rng.shuffle(samples)
    return samples


def generate_off_policy_dataset(n_episodes: int, *, seed: int = 0) -> List[ConnectFourStateActionSample]:
    rng = random.Random(seed)
    samples: List[ConnectFourStateActionSample] = []
    for _ in range(n_episodes):
        s = ConnectFourState.initial()
        while not s.is_terminal():
            a = random_policy_action(s, rng=rng)
            samples.append(ConnectFourStateActionSample(state=s, action=a))
            s = s.apply_move(a)
    return samples


def estimate_v_pi(state: ConnectFourState, policy: PolicyFn, *, root_player: int, n_rollouts: int, seed: int = 0) -> float:
    rng = random.Random(seed)
    total = 0.0
    for _ in range(n_rollouts):
        s = state
        while not s.is_terminal():
            a = policy(s)
            s = s.apply_move(a)
        total += s.terminal_return(root_player)
    return total / float(n_rollouts)


def generate_value_targets(
    states: Iterable[ConnectFourState],
    policy: PolicyFn,
    *,
    root_player: int,
    rollout_budgets: Sequence[int],
    seed: int = 0,
) -> List[ConnectFourStateValueTarget]:
    targets: List[ConnectFourStateValueTarget] = []
    for i, state in enumerate(states):
        for budget in rollout_budgets:
            value = estimate_v_pi(state, policy, root_player=root_player, n_rollouts=budget, seed=seed + i)
            targets.append(ConnectFourStateValueTarget(state=state, value=value, rollouts=budget))
    return targets


def recovered_action_from_v(state: ConnectFourState, value_fn: Callable[[ConnectFourState], float]) -> int:
    legal = state.legal_moves()
    if not legal:
        raise ValueError("no legal actions in terminal state")
    if state.to_move == 1:
        return max(legal, key=lambda a: value_fn(state.apply_move(a)))
    return min(legal, key=lambda a: value_fn(state.apply_move(a)))
