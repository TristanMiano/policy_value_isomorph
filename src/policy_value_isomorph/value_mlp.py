from __future__ import annotations

from dataclasses import dataclass
import ast
import math
import random
import time
from typing import List, Sequence

import torch

from .training_config import TrainingConfig

from .torch_mlp import TorchMLP, TrainingBatch, train_epoch
from .checkpointing import latest_checkpoint_path, load_checkpoint, save_checkpoint
from .policy_mlp import encode_state
from .rollout_value import StateValueTarget
from .telemetry import TelemetryRecord, TrainingTelemetry
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
    telemetry: TrainingTelemetry


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


def train_value_mlp(
    dataset: Sequence[StateValueTarget],
    *,
    hidden_dim: int = 32,
    learning_rate: float = 0.05,
    epochs: int = 80,
    seed: int = 0,
    checkpoint_interval: int = 0,
    checkpoint_dir: str | None = None,
    resume_checkpoint_path: str | None = None,
    resume_latest: bool = False,
    config: TrainingConfig | None = None,
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
    if checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be >= 0")
    if resume_checkpoint_path is not None and resume_latest:
        raise ValueError("set only one of resume_checkpoint_path or resume_latest")

    rng = random.Random(seed)
    model = _init_model(input_dim=10, hidden_dim=hidden_dim, rng=rng)
    order = list(range(len(dataset)))
    start_epoch = 0
    if resume_latest or resume_checkpoint_path is not None:
        if checkpoint_dir is None and resume_latest:
            raise ValueError("checkpoint_dir is required when resume_latest is true")
        checkpoint_to_load = resume_checkpoint_path or str(latest_checkpoint_path(checkpoint_dir, "value"))
        loaded = load_checkpoint(checkpoint_to_load, model_type=TinyMLPValue)
        model = loaded["model"]
        metadata = loaded.get("metadata", {})
        start_epoch = int(loaded["step"])
        saved_order = metadata.get("order")
        if saved_order is not None and len(saved_order) == len(dataset):
            order = saved_order
        rng_state_text = metadata.get("rng_state")
        if isinstance(rng_state_text, str):
            rng.setstate(ast.literal_eval(rng_state_text))
    losses: List[float] = []
    records: list[TelemetryRecord] = []
    start_time = time.perf_counter()

    config = config or TrainingConfig()
    network = TorchMLP(model, torch.device(config.device), getattr(torch, config.dtype))
    optimizer = config.make_optimizer(network, learning_rate)
    rows = [encode_state(sample.state) for sample in dataset]
    batch = TrainingBatch.regression(network, rows, [sample.value for sample in dataset])

    with config.execution(seed):
        for epoch in range(start_epoch, epochs):
            rng.shuffle(order)
            epoch_loss = train_epoch(network, optimizer, batch, order)
            model = TinyMLPValue(**network.weights())
            losses.append(epoch_loss)
            records.append(TelemetryRecord(step=epoch + 1, loss=epoch_loss, wall_time_seconds=time.perf_counter() - start_time))
            if checkpoint_dir is not None and checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
                save_checkpoint(
                    checkpoint_dir,
                    run_type="value",
                    step=epoch + 1,
                    model=model,
                    metadata={"seed": seed, "epoch": epoch + 1, "learning_rate": learning_rate, "rng_state": repr(rng.getstate()), "order": order},
                )

    if checkpoint_dir is not None and (checkpoint_interval == 0 or epochs % checkpoint_interval != 0):
        save_checkpoint(
            checkpoint_dir,
            run_type="value",
            step=epochs,
            model=model,
            metadata={"seed": seed, "epoch": epochs, "learning_rate": learning_rate, "rng_state": repr(rng.getstate()), "order": order},
        )

    telemetry = TrainingTelemetry(
        run_type="value",
        game="tic_tac_toe",
        seed=seed,
        checkpoint_step=epochs,
        hyperparameters={**config.metadata(), "hidden_dim": hidden_dim, "learning_rate": learning_rate, "epochs": epochs},
        records=tuple(records),
    )
    return TrainedValue(model=model, training_log=ValueTrainingLog(losses=losses, telemetry=telemetry))


def value_mlp_predict(state: TicTacToeState, model: TinyMLPValue) -> float:
    """Predict V_phi(s) from fixed-root-player training perspective."""
    return TorchMLP(model).predict([encode_state(state)]).item()
