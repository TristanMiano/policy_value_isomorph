import json

from policy_value_isomorph.policy import heuristic_policy_action
from policy_value_isomorph.policy_mlp import train_policy_mlp
from policy_value_isomorph.q_mlp import generate_q_targets, train_q_mlp
from policy_value_isomorph.rollout_value import generate_value_targets
from policy_value_isomorph.sampling import generate_on_policy_dataset
from policy_value_isomorph.value_mlp import train_value_mlp


def test_periodic_checkpoints_for_policy_value_and_q(tmp_path):
    dataset = generate_on_policy_dataset(heuristic_policy_action, n_episodes=6)
    states = [sample.state for sample in dataset[:8]]

    train_policy_mlp(dataset, epochs=4, seed=3, checkpoint_interval=2, checkpoint_dir=tmp_path)
    value_targets = generate_value_targets(states, heuristic_policy_action, root_player=1, rollout_budgets=[1])
    train_value_mlp(value_targets, epochs=4, seed=3, checkpoint_interval=2, checkpoint_dir=tmp_path)
    q_targets = generate_q_targets(states, heuristic_policy_action, root_player=1, rollout_budgets=[1])
    train_q_mlp(q_targets, epochs=4, seed=3, checkpoint_interval=2, checkpoint_dir=tmp_path)

    for run_type in ("policy", "value", "q"):
        step2 = tmp_path / run_type / "step_000002.json"
        step4 = tmp_path / run_type / "step_000004.json"
        assert step2.exists()
        assert step4.exists()
        payload = json.loads(step4.read_text(encoding="utf-8"))
        assert payload["run_type"] == run_type
        assert payload["step"] == 4
        assert payload["metadata"]["epoch"] == 4
