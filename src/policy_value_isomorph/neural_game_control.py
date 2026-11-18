"""Training and exact game checks for the optional tic-tac-toe neural control."""

from __future__ import annotations

from fractions import Fraction

import torch
from torch import Tensor, nn

from .pilot_protocol import GameGraph, Law, State, minimax_action, minimax_value
from .policy_mlp import encode_state


def network(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, 128), nn.ReLU(), nn.Linear(128, 128),
                         nn.ReLU(), nn.Linear(128, outputs))


def optimizer_step(loss: Tensor, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def adjust_rate(optimizer: torch.optim.Optimizer, epoch: int) -> None:
    if epoch in (101, 201):
        for group in optimizer.param_groups:
            group["lr"] *= 0.35


def policy_tensors(states: list[State]) -> tuple[Tensor, Tensor]:
    x = torch.tensor([encode_state(s) for s in states])
    legal = torch.tensor([[a in s.legal_moves() for a in range(9)] for s in states])
    return x, legal


@torch.no_grad()
def policy_choices(model: nn.Module, x: Tensor, legal: Tensor) -> list[int]:
    return model(x).masked_fill(~legal, float("-inf")).argmax(1).tolist()


def certify_policy(graph: GameGraph, states: list[State], choices: list[int]) -> dict:
    """Compare each chosen move's guaranteed mover payoff with the best legal payoff."""
    required = {s for s in graph.states if graph.children[s]}
    if len(choices) != len(states) or len(states) != len(required) or set(states) != required:
        raise ValueError("Certification requires one choice for every reachable decision board.")
    gaps = []
    for state, move in zip(states, choices):
        child = graph.children[state][move]
        gaps.append(state.to_move * (minimax_value(state) - minimax_value(child)))
    return dict(boards=len(gaps), optimal_moves=gaps.count(0), mean_payoff_forgone=sum(gaps)/len(gaps),
                worst_payoff_forgone=max(gaps),
                same_minimax_move=sum(a == minimax_action(s) for s, a in zip(states, choices)))


def train_source(graph: GameGraph, states: list[State], seed: int) -> tuple[nn.Module, dict, dict]:
    torch.manual_seed(seed)
    x, legal = policy_tensors(states)
    labels = torch.tensor([minimax_action(s) for s in states])
    model = network(10, 9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for epoch in range(1, 301):
        adjust_rate(optimizer, epoch)
        for i in torch.randperm(len(states)).split(256):
            scores = model(x[i]).masked_fill(~legal[i], float("-inf"))
            optimizer_step(nn.functional.cross_entropy(scores, labels[i]), optimizer)
        if epoch % 10 == 0:
            moves = policy_choices(model, x, legal)
            if moves == labels.tolist():
                break
    certificate = certify_policy(graph, states, moves)
    if certificate["worst_payoff_forgone"] != 0:
        raise ValueError(f"Optimal neural control required before value comparison: {certificate}")
    model.eval().requires_grad_(False)
    return model, {s: {a: Fraction(1)} for s, a in zip(states, moves)}, dict(certificate, epoch=epoch, seed=seed)


def fit_values(x: Tensor, train: Tensor, labels: Tensor, validation: Tensor,
               truth: Tensor, seed: int, full: bool, log_loss: bool = False) -> tuple[nn.Module, dict]:
    """Fit the conditional frequency of a legal move using bounded scalar predictions."""
    torch.manual_seed(seed)
    model = network(x.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    order = torch.Generator().manual_seed(seed)
    training_x = x[train]
    target = labels
    best = float("inf")
    history = []
    for epoch in range(1, (300 if full else 120) + 1):
        adjust_rate(optimizer, epoch)
        for i in torch.randperm(len(train), generator=order).split(256):
            outputs = model(training_x[i])
            if log_loss:
                loss = nn.functional.binary_cross_entropy_with_logits(outputs[:, 0], target[i])
            else:
                loss = nn.functional.mse_loss(outputs[:, 0].sigmoid(), target[i])
            optimizer_step(loss, optimizer)
        if epoch == 1 or epoch % 10 == 0:
            with torch.no_grad():
                train_mse = (model(training_x)[:, 0].sigmoid() - target).square().mean().item()
                selection_mse = train_mse if full else (
                    model(x[validation])[:, 0].sigmoid() - truth[validation]
                ).square().mean().item()
            history.append([epoch, train_mse, selection_mse])
            if selection_mse < best:
                best = selection_mse
                selected = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
    model.load_state_dict(selected)
    return model, dict(epoch=best_epoch, training_count=len(train), selection_count=len(train) if full else len(validation),
                       curve_epoch_training_selection_mse=history)
