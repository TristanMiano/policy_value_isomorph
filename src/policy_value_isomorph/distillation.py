"""Fit one scalar neural score per candidate using the source access actually queried."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from random import Random
from typing import Callable, Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .policy_adapter import PolicyAdapter
from .score_encoding import sample_probabilities

OBJECTIVES = {
    "action": "negative log probability of the sampled move",
    "probability": "cross entropy with source move probabilities",
    "logit": "cross entropy with source logits decoded at unit temperature and equal base weights",
    "pair": "negative log probability of the fresh winner among the two offered moves",
}


@dataclass(frozen=True)
class ChoiceExample:
    features: tuple[tuple[float, ...], ...]
    target: tuple[float, ...]


def observe(adapter: PolicyAdapter, history, features: Callable, mode: str, rng: Random,
            pair: Sequence | None = None) -> ChoiceExample:
    """The feature callback can encode the successor alone or the original history and move."""
    menu = tuple(adapter.candidates(history))
    if mode not in OBJECTIVES or not menu:
        raise ValueError("need a supported observation and a decision")
    if mode == "pair":
        if pair is None or len(pair) != 2 or len(set(pair)) != 2 or not set(pair).issubset(menu):
            raise ValueError("provide a legal pair for the fresh source query")
        menu = tuple(pair)
        chosen = adapter.constrained(history, menu, rng)
        target = tuple(float(a == chosen) for a in menu)
    elif mode == "action":
        chosen = adapter.sample(history, rng)
        target = tuple(float(a == chosen) for a in menu)
    elif mode == "probability":
        law = adapter.probabilities(history)
        target = tuple(float(law.get(a, 0)) for a in menu)
    else:
        logits = adapter.logits(history)
        target = sample_probabilities([logits[a] for a in menu], [1]*len(menu), 1)
    return ChoiceExample(tuple(tuple(features(history, a)) for a in menu), target)


def _batch(examples: Sequence[ChoiceExample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("no observations")
    xs, ys, masks = [], [], []
    for row in examples:
        x, y = torch.tensor(row.features, dtype=torch.float64), torch.tensor(row.target, dtype=torch.float64)
        if (x.ndim != 2 or not x.numel() or y.shape != (len(x),) or not x.isfinite().all()
                or not y.isfinite().all() or (y < 0).any() or abs(y.sum().item()-1) > 1e-10):
            raise ValueError("need finite candidate features and matching normalized labels")
        xs.append(x)
        ys.append(y)
        masks.append(torch.ones(len(y), dtype=torch.bool))
    return tuple(pad_sequence(seq, batch_first=True) for seq in (xs, ys, masks))


@dataclass
class DistilledScores:
    model: nn.Module
    mode: str
    loss_nats: float
    observations: int
    hidden: int
    epochs: int
    learning_rate: float
    seed: int

    def scores(self, features: Sequence[Sequence[float]]) -> tuple[float, ...]:
        with torch.no_grad():
            return tuple(self.model(torch.tensor(features, dtype=torch.float64)).flatten().tolist())

    def evaluate(self, examples: Sequence[ChoiceExample]) -> dict:
        """Common legal menus and target distributions; equal weight per observed row."""
        with torch.no_grad():
            x, truth, mask = _batch(examples)
            scores = self.model(x).squeeze(-1).masked_fill(~mask, -torch.inf)
            law = scores.softmax(-1)
            distance = 0.5 * (law-truth).abs().sum(-1)
            chose = truth.gather(1, scores.argmax(-1, keepdim=True))
        return {"rows": len(examples), "mean_probability_mass_reassigned": distance.mean().item(),
                "mean_source_probability_of_selected_move": chose.mean().item()}


def fit_scores(examples: Sequence[ChoiceExample], mode: str, *, hidden: int = 16, epochs: int = 60,
               learning_rate: float = 0.02, seed: int = 0) -> DistilledScores:
    """Full-batch Adam. Mean natural-log loss per observation, equally weighted."""
    if mode not in OBJECTIVES or hidden < 1 or epochs < 1 or not isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("invalid training settings")
    x, truth, mask = _batch(examples)
    if mode in {"action","pair"} and not ((truth == 0) | (truth == 1)).all():
        raise ValueError("sampled winners need one chosen move")
    if mode == "pair" and not (mask.sum(-1) == 2).all():
        raise ValueError("pair observations need exactly two moves")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = nn.Sequential(nn.Linear(x.shape[-1], hidden, dtype=x.dtype), nn.Tanh(),
                              nn.Linear(hidden, 1, dtype=x.dtype))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def loss():
        scores = model(x).squeeze(-1).masked_fill(~mask, -torch.inf)
        log_law = scores.log_softmax(-1).masked_fill(~mask, 0)
        return -(truth * log_law).sum(-1).mean()

    for _ in range(epochs):
        optimizer.zero_grad()
        loss().backward()
        optimizer.step()
    return DistilledScores(model, mode, loss().detach().item(), len(examples), hidden, epochs, learning_rate, seed)
