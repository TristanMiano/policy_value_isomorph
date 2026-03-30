from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import List, Sequence

from .sampling import StateActionSample
from .tictactoe import Move, TicTacToeState


@dataclass
class TinyMLPPolicy:
    """Small 1-hidden-layer MLP for tic-tac-toe policy imitation.

    Input encoding is length-10: 9 board cells in {-1,0,+1} followed by
    side-to-move in {-1,+1}. Output is 9 action logits (one per board index).
    """

    w1: List[List[float]]
    b1: List[float]
    w2: List[List[float]]
    b2: List[float]

    @property
    def input_dim(self) -> int:
        return len(self.w1)

    @property
    def hidden_dim(self) -> int:
        return len(self.b1)


@dataclass
class PolicyTrainingLog:
    losses: List[float]


@dataclass
class TrainedPolicy:
    model: TinyMLPPolicy
    training_log: PolicyTrainingLog


def encode_state(state: TicTacToeState) -> List[float]:
    return [float(x) for x in state.board] + [float(state.to_move)]


def _init_model(input_dim: int, hidden_dim: int, output_dim: int, rng: random.Random) -> TinyMLPPolicy:
    scale1 = 1.0 / math.sqrt(input_dim)
    scale2 = 1.0 / math.sqrt(hidden_dim)

    w1 = [[rng.uniform(-scale1, scale1) for _ in range(hidden_dim)] for _ in range(input_dim)]
    b1 = [0.0 for _ in range(hidden_dim)]
    w2 = [[rng.uniform(-scale2, scale2) for _ in range(output_dim)] for _ in range(hidden_dim)]
    b2 = [0.0 for _ in range(output_dim)]
    return TinyMLPPolicy(w1=w1, b1=b1, w2=w2, b2=b2)


def _forward(model: TinyMLPPolicy, x: Sequence[float]) -> tuple[List[float], List[float]]:
    hidden_pre = [0.0 for _ in range(model.hidden_dim)]
    for j in range(model.hidden_dim):
        v = model.b1[j]
        for i in range(model.input_dim):
            v += x[i] * model.w1[i][j]
        hidden_pre[j] = v

    hidden = [math.tanh(v) for v in hidden_pre]

    logits = [0.0 for _ in range(9)]
    for k in range(9):
        v = model.b2[k]
        for j in range(model.hidden_dim):
            v += hidden[j] * model.w2[j][k]
        logits[k] = v

    return hidden, logits


def _masked_softmax(logits: Sequence[float], legal_moves: Sequence[Move]) -> List[float]:
    if not legal_moves:
        raise ValueError("masked softmax called without legal moves")

    allowed = set(legal_moves)
    max_logit = max(logits[m] for m in legal_moves)

    exps = [0.0 for _ in range(9)]
    denom = 0.0
    for k in range(9):
        if k in allowed:
            exps[k] = math.exp(logits[k] - max_logit)
            denom += exps[k]

    if denom <= 0.0:
        raise ValueError("numerical issue in masked softmax")

    return [e / denom for e in exps]


def train_policy_mlp(
    dataset: Sequence[StateActionSample],
    *,
    hidden_dim: int = 32,
    learning_rate: float = 0.05,
    epochs: int = 60,
    seed: int = 0,
) -> TrainedPolicy:
    """Train a tiny MLP policy by supervised imitation on state-action pairs."""
    if not dataset:
        raise ValueError("dataset must be non-empty")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be >= 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if epochs <= 0:
        raise ValueError("epochs must be >= 1")

    rng = random.Random(seed)
    model = _init_model(input_dim=10, hidden_dim=hidden_dim, output_dim=9, rng=rng)
    losses: List[float] = []

    order = list(range(len(dataset)))

    for _ in range(epochs):
        rng.shuffle(order)
        total_loss = 0.0

        for idx in order:
            sample = dataset[idx]
            x = encode_state(sample.state)
            hidden, logits = _forward(model, x)
            probs = _masked_softmax(logits, sample.state.legal_moves())

            target = sample.action
            p_target = max(probs[target], 1e-12)
            total_loss += -math.log(p_target)

            dlogits = list(probs)
            dlogits[target] -= 1.0

            # second layer gradients / update
            dhidden = [0.0 for _ in range(model.hidden_dim)]
            for j in range(model.hidden_dim):
                for k in range(9):
                    dhidden[j] += model.w2[j][k] * dlogits[k]
                    grad_w2 = hidden[j] * dlogits[k]
                    model.w2[j][k] -= learning_rate * grad_w2

            for k in range(9):
                model.b2[k] -= learning_rate * dlogits[k]

            # first layer gradients / update
            for j in range(model.hidden_dim):
                dpre = (1.0 - hidden[j] * hidden[j]) * dhidden[j]
                for i in range(model.input_dim):
                    grad_w1 = x[i] * dpre
                    model.w1[i][j] -= learning_rate * grad_w1
                model.b1[j] -= learning_rate * dpre

        losses.append(total_loss / len(dataset))

    return TrainedPolicy(model=model, training_log=PolicyTrainingLog(losses=losses))


def policy_mlp_action(state: TicTacToeState, model: TinyMLPPolicy) -> Move:
    """Choose argmax legal action under the trained model logits."""
    legal = state.legal_moves()
    if not legal:
        raise ValueError("policy_mlp_action called on terminal state")

    _, logits = _forward(model, encode_state(state))
    return max(legal, key=lambda mv: (logits[mv], -mv))
