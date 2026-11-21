"""Tensor computation for the one-hidden-layer networks and their JSON weights."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Weights(Protocol):
    """The input-major layout used in the original JSON checkpoints."""

    @property
    def w1(self) -> list[list[float]]: ...
    @property
    def b1(self) -> list[float]: ...
    @property
    def w2(self) -> list[float] | list[list[float]]: ...
    @property
    def b2(self) -> float | list[float]: ...


class TorchMLP(nn.Module):
    """One tanh hidden layer, with a linear output for each score."""

    def __init__(self, weights: Weights, device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        # Load supplied weights directly, without drawing from the global torch RNG.
        def parameter(values: object) -> nn.Parameter:
            return nn.Parameter(torch.tensor(values, dtype=dtype, device=device))

        self.w1 = parameter(weights.w1)
        self.b1 = parameter(weights.b1)
        self.w2 = parameter(weights.w2)
        self.b2 = parameter(weights.b2)
        hidden = self.b1.numel()
        if self.b1.ndim != 1 or self.w1.ndim != 2 or self.w1.shape[1] != hidden:
            raise ValueError("incompatible input and hidden layer shapes")
        if self.b2.ndim > 1 or self.w2.shape != (hidden, *self.b2.shape):
            raise ValueError("incompatible hidden and output layer shapes")

    def forward(self, inputs: Tensor) -> Tensor:
        """Rows are independent positions or position/move pairs."""
        return torch.tanh(inputs @ self.w1 + self.b1) @ self.w2 + self.b2

    def features(self, rows: Sequence[Sequence[float]]) -> Tensor:
        inputs = self.w1.new_tensor(rows)
        if inputs.ndim != 2 or inputs.shape[1] != self.w1.shape[0]:
            raise ValueError("expected one row per example with the model's input width")
        return inputs

    @torch.no_grad()
    def predict(self, rows: Sequence[Sequence[float]]) -> Tensor:
        return self(self.features(rows))

    def weights(self) -> dict:
        """Export an independent snapshot in the original scalar or multi-output layout."""
        return {name: p.detach().tolist() for name, p in self.named_parameters()}


@dataclass(frozen=True)
class TrainingBatch:
    """One feature row per example, with real-valued targets or legal action labels."""

    inputs: Tensor
    targets: Tensor
    legal: Tensor | None = None

    @classmethod
    def regression(cls, model: TorchMLP, rows: Sequence[Sequence[float]], values: Sequence[float]) -> TrainingBatch:
        inputs = model.features(rows)
        targets = inputs.new_tensor(values)
        if model.b2.ndim != 0 or targets.shape != (len(rows),):
            raise ValueError("regression needs one scalar prediction and target per example")
        return cls(inputs, targets)

    @classmethod
    def policy(cls, model: TorchMLP, rows: Sequence[Sequence[float]], actions: Sequence[int],
               masks: Sequence[Sequence[bool]]) -> TrainingBatch:
        inputs = model.features(rows)
        targets = torch.as_tensor(actions, dtype=torch.long, device=inputs.device)
        legal = torch.as_tensor(masks, dtype=torch.bool, device=inputs.device)
        if model.b2.ndim != 1 or targets.shape != (len(rows),) or legal.shape != (len(rows), model.b2.numel()):
            raise ValueError("expected one action and a mask over output moves per example")
        if (targets < 0).any() or (targets >= legal.shape[1]).any() or not legal.gather(1, targets[:, None]).all():
            raise ValueError("policy targets must be legal moves")
        return cls(inputs, targets, legal)


def train_epoch(model: TorchMLP, optimizer: torch.optim.Optimizer, batch: TrainingBatch,
                order: Sequence[int]) -> float:
    """Apply one gradient update per example in the supplied order; report the mean loss."""
    total = 0.0
    for i in order:
        optimizer.zero_grad(set_to_none=True)
        scores = model(batch.inputs[i])
        if batch.legal is None:
            loss = (scores - batch.targets[i]).square()
            total += loss.item()
        else:
            loss = F.cross_entropy(scores.masked_fill(~batch.legal[i], -torch.inf), batch.targets[i])
            # The original log floors the observed move probability at 1e-12.
            # Its update uses the uncapped cross-entropy gradient.
            total += min(loss.item(), -math.log(1e-12))
        loss.backward()
        optimizer.step()
    return total / len(order)
