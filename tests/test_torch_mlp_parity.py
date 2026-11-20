"""Software comparisons with outputs recorded by the original Python trainers."""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from policy_value_isomorph.policy_mlp import TinyMLPPolicy, policy_mlp_action, policy_mlp_probabilities, train_policy_mlp
from policy_value_isomorph.q_mlp import StateActionValueTarget, TinyMLPQ, recovered_action_from_q, train_q_mlp
from policy_value_isomorph.rollout_value import StateValueTarget
from policy_value_isomorph.sampling import StateActionSample
from policy_value_isomorph.tictactoe import TicTacToeState
from policy_value_isomorph.value_mlp import train_value_mlp


FIXTURE = json.loads((Path(__file__).parent / "fixtures/manual_mlp_training.json").read_text())
# Absolute error in the original loss/weight units; no relative tolerance.
ATOL = 1e-12


@pytest.mark.parametrize("name", ["policy", "value", "q"])
def test_archived_training_and_resume(name, tmp_path):
    examples = [(TicTacToeState(tuple(e["board"]), e["to_move"]), e["action"], e["value"])
                for e in FIXTURE["examples"]]
    if name == "policy":
        train = train_policy_mlp
        data = [StateActionSample(s, a) for s, a, _ in examples]
    elif name == "value":
        train = train_value_mlp
        data = [StateValueTarget(s, 1, v) for s, _, v in examples]
    else:
        train = train_q_mlp
        data = [StateActionValueTarget(s, a, 1, v) for s, a, v in examples]
    options = FIXTURE["options"]
    uninterrupted = train(data, **options)
    expected = FIXTURE[name]
    assert uninterrupted.training_log.losses == pytest.approx(expected["losses"], abs=ATOL, rel=0)
    for key, actual in asdict(uninterrupted.model).items():
        original = torch.tensor(expected["weights"][key], dtype=torch.float64)
        torch.testing.assert_close(torch.tensor(actual, dtype=torch.float64), original, atol=ATOL, rtol=0)
    train(data, **{**options, "epochs": 1}, checkpoint_dir=tmp_path)
    resumed = train(data, **options, checkpoint_dir=tmp_path, resume_latest=True)
    assert resumed.model == uninterrupted.model
    assert resumed.training_log.losses == uninterrupted.training_log.losses[1:]
    # Original files saved the random generator state but started with a fresh index list.
    legacy = json.loads((tmp_path / name / "step_000003.json").read_text())
    legacy["model"] = expected["weights"]
    del legacy["metadata"]["order"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy))
    continued = train(data, **{**options, "epochs": 4}, resume_checkpoint_path=path)
    assert continued.training_log.losses == pytest.approx([expected["legacy_resume_loss"]], abs=ATOL, rel=0)


def test_public_decoders_resolve_exact_ties_and_ignore_illegal_scores():
    state = TicTacToeState.initial().apply_move(0)
    policy = TinyMLPPolicy([[0.0]] * 10, [0.0], [[0.0] * 9], [999.0] + [0.0] * 8)
    assert policy_mlp_action(state, policy) == 1
    assert policy_mlp_probabilities(state, policy) == [0.0] + [0.125] * 8
    q = TinyMLPQ([[0.0]] * 19, [0.0], [0.0], 0.0)
    for root in (-1, 1):
        assert recovered_action_from_q(state, q, root_player=root) == 1
    # Raising the value of cell 8 benefits the fixed root and harms its opponent.
    q.w1[-1] = [1.0]
    q.w2[0] = 1.0
    assert recovered_action_from_q(state, q, root_player=-1) == 8
    assert recovered_action_from_q(state, q, root_player=1) == 1


def test_replay_all_frozen_choices_and_outcomes(tmp_path):
    report = tmp_path / "replay.json"
    script = Path(__file__).resolve().parents[1] / "scripts/replay_torch_pilot.py"
    subprocess.run([sys.executable, str(script), "--output", str(report)], check=True)
    replay = json.loads(report.read_text())
    original = replay["frozen_players"]
    assert original["boards"] == 6 * 4520 and original["different_public_choices"] == 0
    assert original["different_best_sets"] == 0 and original["max_abs_score_change"] < ATOL
    for name, measures in replay["moves"].items():
        assert measures["boards"] == 5 * 4520 and measures["legal_scores"] == 5 * 16167
        assert measures["different_archived_top_moves"] == measures["different_best_sets"] == 0
        assert measures["max_abs_score_change"] < ATOL
        if name == "policy":
            assert measures["max_abs_probability_change"] < ATOL
        else:
            assert measures["different_signs"] == 0
    assert replay["moves"]["value"]["max_abs_current_board_change"] < ATOL
    assert replay["outcomes"]["rows"] == 900
    for field in ("win", "draw", "loss", "score"):
        assert replay["outcomes"][field] < ATOL
