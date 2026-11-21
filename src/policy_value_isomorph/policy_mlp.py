from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import List, Sequence

from .training_config import TrainingConfig

from .torch_mlp import TorchMLP, TrainingBatch
from .checkpointing import TrainingRun
from .gameplay_eval import GameplayEvalRecord
from .gameplay_samples import GameplaySample, append_gameplay_samples_text
from .sampling import StateActionSample
from .telemetry import TelemetryRecord, TrainingTelemetry
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
    telemetry: TrainingTelemetry
    gameplay_evals: List[GameplayEvalRecord]


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


def untrained_policy_mlp(hidden_dim: int, *, seed: int) -> TinyMLPPolicy:
    """Initialize the same legal-move policy architecture used by the trainer."""
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be >= 1")
    return _init_model(10, hidden_dim, 9, random.Random(seed))


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
    learning_rate: float | None = None,
    epochs: int = 60,
    seed: int = 0,
    eval_interval: int = 0,
    eval_n_games: int = 20,
    gameplay_samples_path: str | None = None,
    gameplay_samples_per_checkpoint: int = 2,
    checkpoint_interval: int = 0,
    checkpoint_dir: str | None = None,
    resume_checkpoint_path: str | None = None,
    resume_latest: bool = False,
    config: TrainingConfig | None = None,
) -> TrainedPolicy:
    """Train a tiny MLP policy by supervised imitation on state-action pairs."""
    if not dataset:
        raise ValueError("dataset must be non-empty")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be >= 1")
    if learning_rate is not None and learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if epochs <= 0:
        raise ValueError("epochs must be >= 1")
    if eval_interval < 0:
        raise ValueError("eval_interval must be >= 0")
    if eval_interval > 0 and eval_n_games <= 0:
        raise ValueError("eval_n_games must be >= 1 when eval_interval is enabled")
    if gameplay_samples_per_checkpoint <= 0:
        raise ValueError("gameplay_samples_per_checkpoint must be >= 1")
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be >= 0")
    if resume_checkpoint_path is not None and resume_latest:
        raise ValueError("set only one of resume_checkpoint_path or resume_latest")

    rng = random.Random(seed)
    model = _init_model(input_dim=10, hidden_dim=hidden_dim, output_dim=9, rng=rng)
    run = TrainingRun("policy", model, rng, len(dataset), seed, learning_rate, config, checkpoint_dir,
                      resume_checkpoint_path, resume_latest)
    model = run.model
    losses: List[float] = []

    records: list[TelemetryRecord] = []
    gameplay_evals: list[GameplayEvalRecord] = []
    rows = [encode_state(sample.state) for sample in dataset]
    masks = []
    for sample in dataset:
        legal = sample.state.legal_moves()
        masks.append([move in legal for move in range(9)])
    batch = TrainingBatch.policy(run.network, rows, [sample.action for sample in dataset], masks)

    for record in run.records(batch, epochs, checkpoint_interval):
        model = TinyMLPPolicy(**run.network.weights())
        losses.append(record.loss)
        records.append(record)
        if eval_interval > 0 and record.step % eval_interval == 0:
            wins = 0
            draws = 0
            losses_eval = 0
            checkpoint_samples: list[GameplaySample] = []
            for i in range(eval_n_games):
                state = TicTacToeState.initial()
                model_as_x = (i % 2 == 0)
                states = [state]
                moves: list[int] = []
                while not state.is_terminal():
                    if model_as_x:
                        move = policy_mlp_action(state, model) if state.to_move == 1 else min(state.legal_moves())
                    else:
                        move = min(state.legal_moves()) if state.to_move == 1 else policy_mlp_action(state, model)
                    moves.append(move)
                    state = state.apply_move(move)
                    states.append(state)
                result_for_x = state.terminal_return(root_player=1)
                result_for_model = result_for_x if model_as_x else -result_for_x
                if result_for_model > 0:
                    wins += 1
                elif result_for_model < 0:
                    losses_eval += 1
                else:
                    draws += 1
                if len(checkpoint_samples) < gameplay_samples_per_checkpoint:
                    checkpoint_samples.append(
                        GameplaySample(
                            checkpoint_step=record.step,
                            game_index=i,
                            moves=tuple(moves),
                            states=tuple(states),
                        )
                    )

            denom = float(eval_n_games)
            win_rate = wins / denom
            draw_rate = draws / denom
            loss_rate = losses_eval / denom
            gameplay_evals.append(
                GameplayEvalRecord(
                    step=record.step,
                    n_games=eval_n_games,
                    win_rate=win_rate,
                    draw_rate=draw_rate,
                    loss_rate=loss_rate,
                    score=win_rate - loss_rate,
                )
            )
            if gameplay_samples_path is not None:
                append_gameplay_samples_text(gameplay_samples_path, checkpoint_samples)


    telemetry = TrainingTelemetry(
        run_type="policy",
        game="tic_tac_toe",
        seed=run.seed,
        checkpoint_step=epochs,
        hyperparameters={**run.config.metadata(),
            "hidden_dim": model.hidden_dim,
            "learning_rate": run.rate,
            "epochs": epochs,
            "eval_interval": eval_interval,
            "eval_n_games": eval_n_games,
            "gameplay_samples_path": gameplay_samples_path or "",
            "gameplay_samples_per_checkpoint": gameplay_samples_per_checkpoint,
        },
        records=tuple(records),
    )
    return TrainedPolicy(model=model, training_log=PolicyTrainingLog(losses=losses, telemetry=telemetry, gameplay_evals=gameplay_evals))


def policy_mlp_probabilities(state: TicTacToeState, model: TinyMLPPolicy) -> list[float]:
    """Return the temperature-one softmax over the nine cells, masking illegal moves."""
    logits = TorchMLP(model).predict([encode_state(state)])[0].tolist()
    return _masked_softmax(logits, state.legal_moves())


def policy_mlp_action(state: TicTacToeState, model: TinyMLPPolicy) -> Move:
    """Choose argmax legal action under the trained model logits."""
    legal = state.legal_moves()
    if not legal:
        raise ValueError("policy_mlp_action called on terminal state")

    logits = TorchMLP(model).predict([encode_state(state)])[0].tolist()
    return max(legal, key=lambda mv: (logits[mv], -mv))
