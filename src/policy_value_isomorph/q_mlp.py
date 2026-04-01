from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import List, Sequence

from .policy_mlp import encode_state
from .rollout_value import PolicyFn, rollout_return
from .tictactoe import Move, TicTacToeState


@dataclass(frozen=True)
class StateActionValueTarget:
    """Monte-Carlo action-value label for one state-action pair.

    Values follow the same fixed-root-player sign convention used by
    state-value labels, i.e. utilities in [-1,+1] from `root_player` view.
    """

    state: TicTacToeState
    action: Move
    n_rollouts: int
    value: float


@dataclass
class TinyMLPQ:
    """Small 1-hidden-layer MLP for action-value regression.

    Input encoding is state encoding (10 floats) concatenated with a one-hot
    action encoding (9 floats), for total dimension 19.
    """

    w1: List[List[float]]
    b1: List[float]
    w2: List[float]
    b2: float

    @property
    def input_dim(self) -> int:
        return len(self.w1)

    @property
    def hidden_dim(self) -> int:
        return len(self.b1)


@dataclass
class QTrainingLog:
    losses: List[float]


@dataclass
class TrainedQ:
    model: TinyMLPQ
    training_log: QTrainingLog


def _encode_state_action(state: TicTacToeState, action: Move) -> List[float]:
    x = encode_state(state)
    action_one_hot = [0.0 for _ in range(9)]
    action_one_hot[action] = 1.0
    return x + action_one_hot


def _init_model(input_dim: int, hidden_dim: int, rng: random.Random) -> TinyMLPQ:
    scale1 = 1.0 / math.sqrt(input_dim)
    scale2 = 1.0 / math.sqrt(hidden_dim)

    w1 = [[rng.uniform(-scale1, scale1) for _ in range(hidden_dim)] for _ in range(input_dim)]
    b1 = [0.0 for _ in range(hidden_dim)]
    w2 = [rng.uniform(-scale2, scale2) for _ in range(hidden_dim)]
    b2 = 0.0
    return TinyMLPQ(w1=w1, b1=b1, w2=w2, b2=b2)


def _forward(model: TinyMLPQ, x: Sequence[float]) -> tuple[List[float], float]:
    hidden_pre = [0.0 for _ in range(model.hidden_dim)]
    for j in range(model.hidden_dim):
        v = model.b1[j]
        for i in range(model.input_dim):
            v += x[i] * model.w1[i][j]
        hidden_pre[j] = v

    hidden = [math.tanh(v) for v in hidden_pre]

    out = model.b2
    for j in range(model.hidden_dim):
        out += hidden[j] * model.w2[j]

    return hidden, out


def estimate_q_pi(
    state: TicTacToeState,
    action: Move,
    policy: PolicyFn,
    *,
    root_player: int,
    n_rollouts: int = 1,
) -> float:
    """Monte-Carlo estimate of Q^pi(state, action) with fixed root perspective."""
    if n_rollouts <= 0:
        raise ValueError("n_rollouts must be >= 1")
    if action not in state.legal_moves():
        raise ValueError("action must be legal for the provided state")

    next_state = state.apply_move(action)
    total = 0.0
    for _ in range(n_rollouts):
        total += rollout_return(next_state, policy=policy, root_player=root_player)
    return total / n_rollouts


def generate_q_targets(
    states: Sequence[TicTacToeState],
    policy: PolicyFn,
    *,
    root_player: int,
    rollout_budgets: Sequence[int],
) -> list[StateActionValueTarget]:
    """Generate action-value targets for each legal move of each state/budget."""
    if not states:
        raise ValueError("states must be non-empty")
    if not rollout_budgets:
        raise ValueError("rollout_budgets must be non-empty")

    targets: list[StateActionValueTarget] = []
    for n_rollouts in rollout_budgets:
        if n_rollouts <= 0:
            raise ValueError("rollout budgets must be >= 1")
        for state in states:
            for action in state.legal_moves():
                value = estimate_q_pi(
                    state,
                    action,
                    policy,
                    root_player=root_player,
                    n_rollouts=n_rollouts,
                )
                targets.append(
                    StateActionValueTarget(
                        state=state,
                        action=action,
                        n_rollouts=n_rollouts,
                        value=value,
                    )
                )
    return targets


def train_q_mlp(
    dataset: Sequence[StateActionValueTarget],
    *,
    hidden_dim: int = 32,
    learning_rate: float = 0.05,
    epochs: int = 80,
    seed: int = 0,
) -> TrainedQ:
    """Train a tiny MLP regressor for Q_phi(s,a)."""
    if not dataset:
        raise ValueError("dataset must be non-empty")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be >= 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if epochs <= 0:
        raise ValueError("epochs must be >= 1")

    rng = random.Random(seed)
    model = _init_model(input_dim=19, hidden_dim=hidden_dim, rng=rng)
    losses: List[float] = []
    order = list(range(len(dataset)))

    for _ in range(epochs):
        rng.shuffle(order)
        total_loss = 0.0

        for idx in order:
            sample = dataset[idx]
            x = _encode_state_action(sample.state, sample.action)
            hidden, pred = _forward(model, x)

            err = pred - sample.value
            total_loss += err * err

            d_out = 2.0 * err

            d_hidden = [0.0 for _ in range(model.hidden_dim)]
            for j in range(model.hidden_dim):
                d_hidden[j] = model.w2[j] * d_out
                grad_w2 = hidden[j] * d_out
                model.w2[j] -= learning_rate * grad_w2
            model.b2 -= learning_rate * d_out

            for j in range(model.hidden_dim):
                dpre = (1.0 - hidden[j] * hidden[j]) * d_hidden[j]
                for i in range(model.input_dim):
                    grad_w1 = x[i] * dpre
                    model.w1[i][j] -= learning_rate * grad_w1
                model.b1[j] -= learning_rate * dpre

        losses.append(total_loss / len(dataset))

    return TrainedQ(model=model, training_log=QTrainingLog(losses=losses))


def q_mlp_predict(state: TicTacToeState, action: Move, model: TinyMLPQ) -> float:
    """Predict Q_phi(s,a) from fixed-root-player training perspective."""
    _, pred = _forward(model, _encode_state_action(state, action))
    return pred


def recovered_action_from_q(state: TicTacToeState, model: TinyMLPQ, *, root_player: int) -> Move:
    """Choose action by direct Q_phi(s,a) under fixed-root sign convention."""
    legal = state.legal_moves()
    if not legal:
        raise ValueError("recovered_action_from_q called on terminal state")

    if state.to_move == root_player:
        return max(legal, key=lambda mv: (q_mlp_predict(state, mv, model), -mv))
    return min(legal, key=lambda mv: (q_mlp_predict(state, mv, model), mv))
