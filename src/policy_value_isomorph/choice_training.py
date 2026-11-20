"""Fit distributions over legal moves to observed conditional visit frequencies."""

from __future__ import annotations

from graphlib import TopologicalSorter

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .neural_game_control import adjust_rate, network, optimizer_step
from .pilot_protocol import GameGraph, State
from .symmetry import canonicalize_state
from .tictactoe import WIN_LINES
from .trajectory_utility import MoveSampler


def line_features(state: State) -> list[float]:
    """Counts of unblocked lines and distinct immediate winning cells, for both sides."""
    features = []
    for player in (state.to_move, -state.to_move):
        lines = [0, 0, 0]
        threats = set()
        for line in WIN_LINES:
            marks = [state.board[i] * player for i in line]
            count = sum(marks)
            if -1 in marks or count == 0:
                continue
            lines[count - 1] += 1
            if count == 2:
                threats.add(line[marks.index(0)])
        features.extend([*lines, len(threats)])
    return features


def board_geometry(graph: GameGraph, sampler: MoveSampler) -> Tensor:
    """Board geometry. Pool possible replies without source choices or payoff labels."""
    local = np.array([line_features(s) for s in graph.states], dtype=np.float32)
    features = [local]
    for _ in range(2):
        previous = features[-1]
        pooled = np.zeros((len(graph.states), 3 * previous.shape[1]), dtype=np.float32)
        for i in np.flatnonzero(sampler.active):
            outcomes = previous[sampler.child[sampler.legal[i, sampler.legal[i] >= 0]]]
            # Summing these short float32 vectors in float64 makes rotations exactly identical.
            pooled[i] = np.concatenate((outcomes.min(0), outcomes.mean(0, dtype=np.float64), outcomes.max(0)))
        features.append(pooled)
    cells = [canonicalize_state(State(tuple(c * s.to_move for c in s.board), 1)).board for s in graph.states]
    return torch.from_numpy(np.column_stack([*features, cells]).astype(np.float32))


def move_log_probabilities(logits: Tensor, legal: Tensor, rule: str) -> Tensor:
    """Log probabilities conditional on a legal selection; padded cells have zero probability."""
    if rule == "three_levels":
        # A move at level k wins if earlier moves are strictly below k and later moves are at most k.
        log_mass = F.log_softmax(logits, dim=-1)
        cdf = torch.logcumsumexp(log_mass, -1).masked_fill(~legal[..., None], 0)
        later = F.pad(cdf[..., 1:, :].flip((-2,)).cumsum(-2).flip((-2,)), (0, 0, 0, 1))
        earlier = F.pad(cdf[..., :-1, :-1].cumsum(-2), (1, 0, 1, 0))
        earlier_legal = F.pad(legal[..., :-1].int().cumsum(-1), (1, 0)) > 0
        earlier[..., 0] = earlier[..., 0].masked_fill(earlier_legal, -torch.inf)
        logits = torch.logsumexp(log_mass + earlier + later, dim=-1)
    elif rule == "first_acceptable":
        # Probability that this cell is acceptable and every earlier legal cell was skipped.
        log_skip = F.logsigmoid(-logits).masked_fill(~legal, 0)
        earlier = F.pad(log_skip[..., :-1].cumsum(-1), (1, 0))
        logits = F.logsigmoid(logits) + earlier
    elif rule != "softmax":
        raise ValueError(f"Unknown legal move probability rule: {rule}")
    return F.log_softmax(logits.masked_fill(~legal, -torch.inf), dim=-1)


def fit_choice(
    x: Tensor,
    edges: Tensor,
    train: Tensor,
    validation: Tensor,
    frequencies: Tensor,
    seed: int,
    rule: str,
    final_pass: bool = False,
    examples_per_pass: int | None = None,
) -> tuple[nn.Module, dict]:
    """Fit the observed move distribution. Use minimum validation loss or the fixed final-pass control."""
    torch.manual_seed(seed)
    model = network(x.shape[-1], 3 if rule == "three_levels" else 1)
    order = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    history, best, selected = [], float("inf"), None
    if not len(train) or (not len(validation) and not final_pass):
        raise ValueError(
            "Training requires observations and checkpoint selection requires validation boards."
        )

    if examples_per_pass is not None and examples_per_pass < 1:
        raise ValueError("A training pass requires a positive number of examples.")

    def loss(boards: Tensor) -> Tensor:
        ids = edges[boards]
        legal = ids >= 0
        logits = model(x[ids.clamp_min(0)]).squeeze(-1)
        logs = move_log_probabilities(logits, legal, rule).masked_fill(~legal, 0)
        targets = frequencies[ids.clamp_min(0)].masked_fill(~legal, 0)
        return -(targets * logs).sum(-1).mean()

    for epoch in range(1, 121):
        adjust_rate(optimizer, epoch)
        sequence = torch.randperm(len(train), generator=order)
        if examples_per_pass is not None and examples_per_pass > len(train):
            repeat = (examples_per_pass - 1) // len(train)
            cycles = [sequence] + [torch.randperm(len(train), generator=order) for _ in range(repeat)]
            sequence = torch.cat(cycles)
        if examples_per_pass is not None:
            sequence = sequence[:examples_per_pass]
        for batch in sequence.split(64):
            optimizer_step(loss(train[batch]), optimizer)
        if epoch == 1 or epoch % 10 == 0:
            with torch.no_grad():
                score = loss(validation).item() if len(validation) else None
                history.append([epoch, loss(train).item(), score])
                if (score is not None and score < best) or (final_pass and epoch == 120):
                    best, best_epoch = score, epoch
                    selected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(selected)
    return model, dict(
        epoch=best_epoch,
        training_boards=len(train),
        validation_boards=len(validation),
        epoch_train_validation_log_loss=history,
    )


def strict_board_ranks(sampler: MoveSampler, counts: np.ndarray) -> np.ndarray:
    """One scalar per resulting board, with every observed chosen child above its siblings."""
    predecessors = {i: set() for i in range(len(sampler.states))}
    for row in sampler.legal[sampler.active]:
        edges = row[row >= 0]
        chosen = edges[counts[edges] > 0]
        if len(chosen) != 1:
            raise ValueError("One observed move per parent is required for this deterministic ranking.")
        predecessors[sampler.child[chosen[0]]].update(sampler.child[edges[edges != chosen[0]]])
    scores = np.zeros(len(sampler.states), dtype=np.int64)
    for child in TopologicalSorter(predecessors).static_order():
        scores[child] = max((scores[p] + 1 for p in predecessors[child]), default=0)
    return scores
