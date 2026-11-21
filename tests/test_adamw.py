import random

import pytest
import torch

from policy_value_isomorph import policy_mlp, q_mlp, value_mlp
from policy_value_isomorph.rollout_value import StateValueTarget
from policy_value_isomorph.sampling import StateActionSample
from policy_value_isomorph.tictactoe import TicTacToeState
from policy_value_isomorph.torch_mlp import TorchMLP
from policy_value_isomorph.training_config import TrainingConfig


@pytest.mark.parametrize("kind", ["policy", "value", "q"])
def test_configured_adamw_matches_native_updates(kind):
    seed = 17
    rng = random.Random(seed)
    cfg = TrainingConfig(optimizer="adamw", dtype="float32", deterministic=True,
                         weight_decay=0.12, beta1=0.8, beta2=0.95, epsilon=1e-6)
    board = TicTacToeState.initial().apply_move(4)
    x = policy_mlp.encode_state(board)
    if kind == "policy":
        initial = policy_mlp._init_model(10, 3, 9, rng)
        train = policy_mlp.train_policy_mlp
        data = [StateActionSample(board, 0)]
    elif kind == "value":
        initial = value_mlp._init_model(10, 3, rng)
        train = value_mlp.train_value_mlp
        data = [StateValueTarget(board, 1, -1.0)]
    else:
        initial = q_mlp._init_model(19, 3, rng)
        train = q_mlp.train_q_mlp
        data = [q_mlp.StateActionValueTarget(board, 0, 1, -1.0)]
        x = q_mlp._encode_state_action(board, 0)
    reference = TorchMLP(initial, dtype=torch.float32)
    native = torch.optim.AdamW(reference.parameters(), lr=0.003, weight_decay=0.12,
                               betas=(0.8, 0.95), eps=1e-6)
    losses = []
    for _ in range(3):
        native.zero_grad()
        scores = reference(torch.tensor(x, dtype=torch.float32))
        if kind == "policy":
            legal = board.legal_moves()
            loss = -scores[legal].log_softmax(0)[legal.index(0)]
        else:
            loss = (scores + 1).square()
        loss.backward()
        native.step()
        losses.append(loss.item())
    old_rng = torch.get_rng_state()
    old_mode = torch.are_deterministic_algorithms_enabled()
    fit = train(data, config=cfg, seed=seed, hidden_dim=3, learning_rate=0.003, epochs=3)
    assert torch.equal(old_rng, torch.get_rng_state())
    assert torch.are_deterministic_algorithms_enabled() == old_mode
    assert fit.training_log.losses == pytest.approx(losses, abs=1e-6)
    for key, weight in reference.weights().items():
        torch.testing.assert_close(torch.tensor(getattr(fit.model, key)), torch.tensor(weight))
