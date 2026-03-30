from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import List, Sequence

from .policy_mlp import encode_state
from .rollout_value import StateValueTarget
from .tictactoe import TicTacToeState


@dataclass
class TinyMLPValue:
    """Small 1-hidden-layer MLP for tic-tac-toe state-value regression.

    Input encoding matches policy MLP: 9 board cells in {-1,0,+1} plus
    side-to-move in {-1,+1}. Output is one scalar value estimate.
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
class ValueTrainingLog:
    losses: List[float]


@dataclass
class TrainedValue:
    model: TinyMLPValue
    training_log: ValueTrainingLog


def _init_model(input_dim: int, hidden_dim: int, rng: random.Random) -> TinyMLPValue:
    scale1 = 1.0 / math.sqrt(input_dim)
    scale2 = 1.0 / math.sqrt(hidden_dim)

    w1 = [[rng.uniform(-scale1, scale1) for _ in range(hidden_dim)] for _ in range(input_dim)]
    b1 = [0.0 for _ in range(hidden_dim)]
    w2 = [rng.uniform(-scale2, scale2) for _ in range(hidden_dim)]
    b2 = 0.0
    return TinyMLPValue(w1=w1, b1=b1, w2=w2, b2=b2)


def _forward(model: TinyMLPValue, x: Sequence[float]) -> tuple[List[float], float]:
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


def train_value_mlp(
    dataset: Sequence[StateValueTarget],
    *,
    hidden_dim: int = 32,
    learning_rate: float = 0.05,
    epochs: int = 80,
    seed: int = 0,
) -> TrainedValue:
    """Train a tiny MLP value regressor on rollout-generated labels.

    Value-sign convention is inherited from dataset labels: each target value
    is interpreted as V^pi(s) from a fixed root-player perspective.
    """
    if not dataset:
        raise ValueError("dataset must be non-empty")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be >= 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if epochs <= 0:
        raise ValueError("epochs must be >= 1")

    rng = random.Random(seed)
    model = _init_model(input_dim=10, hidden_dim=hidden_dim, rng=rng)
    losses: List[float] = []
    order = list(range(len(dataset)))

    for _ in range(epochs):
        rng.shuffle(order)
        total_loss = 0.0

        for idx in order:
            sample = dataset[idx]
            x = encode_state(sample.state)
            hidden, pred = _forward(model, x)

            err = pred - sample.value
            total_loss += err * err

            # d/dpred [ (pred-target)^2 ] = 2*(pred-target)
            d_out = 2.0 * err

            # second layer gradients / update
            d_hidden = [0.0 for _ in range(model.hidden_dim)]
            for j in range(model.hidden_dim):
                d_hidden[j] = model.w2[j] * d_out
                grad_w2 = hidden[j] * d_out
                model.w2[j] -= learning_rate * grad_w2
            model.b2 -= learning_rate * d_out

            # first layer gradients / update
            for j in range(model.hidden_dim):
                dpre = (1.0 - hidden[j] * hidden[j]) * d_hidden[j]
                for i in range(model.input_dim):
                    grad_w1 = x[i] * dpre
                    model.w1[i][j] -= learning_rate * grad_w1
                model.b1[j] -= learning_rate * dpre

        losses.append(total_loss / len(dataset))

    return TrainedValue(model=model, training_log=ValueTrainingLog(losses=losses))


def value_mlp_predict(state: TicTacToeState, model: TinyMLPValue) -> float:
    """Predict V_phi(s) from fixed-root-player training perspective."""
    _, pred = _forward(model, encode_state(state))
    return pred
