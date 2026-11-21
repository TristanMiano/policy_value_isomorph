"""Configuration shared by the three supervised tic-tac-toe trainers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Iterator

import torch
from torch import nn


@dataclass(frozen=True)
class TrainingConfig:
    """SGD preserves the original update. AdamW uses adaptive steps and separate decay."""

    optimizer: str = "sgd"
    device: str = "cpu"
    dtype: str = "float64"
    deterministic: bool | None = None
    weight_decay: float | None = None
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.optimizer not in ("sgd", "adamw"):
            raise ValueError("optimizer must be sgd or adamw")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        device = torch.device(self.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("device must select CPU or CUDA")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable in this PyTorch installation")

    @property
    def decay(self) -> float:
        if self.weight_decay is not None:
            return self.weight_decay
        return 0.01 if self.optimizer == "adamw" else 0.0

    def metadata(self) -> dict:
        return {**asdict(self), "weight_decay": self.decay}

    def make_optimizer(self, model: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
        if self.optimizer == "sgd":
            return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=self.decay)
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=self.decay,
                                 betas=(self.beta1, self.beta2), eps=self.epsilon)

    @contextmanager
    def execution(self, seed: int) -> Iterator[None]:
        """Scope PyTorch's random seed and deterministic-algorithm mode to this fit."""
        device = torch.device(self.device)
        devices = [device] if device.type == "cuda" else []
        mode = torch.are_deterministic_algorithms_enabled()
        warnings = torch.is_deterministic_algorithms_warn_only_enabled()
        with torch.random.fork_rng(devices=devices):
            torch.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            try:
                if self.deterministic is not None:
                    torch.use_deterministic_algorithms(self.deterministic)
                yield
            finally:
                torch.use_deterministic_algorithms(mode, warn_only=warnings)
