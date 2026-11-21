import pytest
import torch

from policy_value_isomorph.checkpointing import latest_checkpoint_path, load_checkpoint
from policy_value_isomorph.policy import heuristic_policy_action
from policy_value_isomorph.policy_mlp import train_policy_mlp
from policy_value_isomorph.q_mlp import generate_q_targets, train_q_mlp
from policy_value_isomorph.rollout_value import generate_value_targets
from policy_value_isomorph.sampling import generate_on_policy_dataset
from policy_value_isomorph.training_config import TrainingConfig
from policy_value_isomorph.value_mlp import train_value_mlp


@pytest.mark.parametrize("kind", ["policy", "value", "q"])
@pytest.mark.parametrize("latest", [True, False])
def test_adamw_continuation_restores_state_and_order(tmp_path, kind, latest):
    policy = heuristic_policy_action
    moves = generate_on_policy_dataset(policy, n_episodes=1)[:3]
    states = [m.state for m in moves]
    if kind == "policy":
        train, data = train_policy_mlp, moves
    elif kind == "value":
        train, data = train_value_mlp, generate_value_targets(states, policy, root_player=1, rollout_budgets=[1])
    else:
        train, data = train_q_mlp, generate_q_targets(states, policy, root_player=1, rollout_budgets=[1])
    kwargs = dict(config=TrainingConfig(optimizer="adamw", dtype="float32", deterministic=True),
                  hidden_dim=5, seed=109, learning_rate=0.002)
    whole_dir, split_dir = tmp_path / "whole", tmp_path / "split"
    whole = train(data, **kwargs, epochs=4, checkpoint_dir=whole_dir)
    part = train(data, **kwargs, epochs=2, checkpoint_dir=split_dir)
    file = latest_checkpoint_path(split_dir, kind)
    assert file.suffix == ".pt"
    assert load_checkpoint(file, model_type=type(part.model))["model"] == part.model
    resume = dict(resume_latest=True) if latest else dict(resume_checkpoint_path=file)
    end = train(data, epochs=4, checkpoint_dir=split_dir, **resume)
    assert end.model == whole.model
    assert end.training_log.losses == whole.training_log.losses[2:]
    assert end.training_log.telemetry.seed == 109
    records = end.training_log.telemetry.records
    assert [r.step for r in records] == [3, 4]
    assert records[0].wall_time_seconds >= part.training_log.telemetry.records[-1].wall_time_seconds
    original = torch.load(latest_checkpoint_path(whole_dir, kind), weights_only=True)
    continued = torch.load(latest_checkpoint_path(split_dir, kind), weights_only=True)
    for key in ("model_state", "optimizer_state", "cpu_rng"):
        torch.testing.assert_close(continued[key], original[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="examples"):
        train(data[::-1], epochs=5, resume_checkpoint_path=file)


def test_latest_orders_legacy_and_native_numerically(tmp_path):
    folder = tmp_path / "policy"
    folder.mkdir()
    for name in ("step_999999.json", "step_1000000.json", "step_1000000.pt", "incomplete.tmp"):
        (folder / name).touch()
    assert latest_checkpoint_path(tmp_path, "policy").name == "step_1000000.pt"
