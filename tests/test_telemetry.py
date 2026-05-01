from policy_value_isomorph.policy_mlp import train_policy_mlp
from policy_value_isomorph.q_mlp import generate_q_targets, train_q_mlp
from policy_value_isomorph.rollout_value import frozen_policy_from_mlp, generate_value_targets
from policy_value_isomorph.sampling import generate_off_policy_dataset
from policy_value_isomorph.value_mlp import train_value_mlp


def test_standardized_telemetry_schema_for_policy_value_q() -> None:
    dataset = generate_off_policy_dataset(n_episodes=14, seed=5)
    trained_policy = train_policy_mlp(dataset, hidden_dim=12, epochs=3, seed=5)

    policy_telemetry = trained_policy.training_log.telemetry
    assert policy_telemetry.run_type == "policy"
    assert policy_telemetry.game == "tic_tac_toe"
    assert policy_telemetry.seed == 5
    assert policy_telemetry.checkpoint_step == 3
    assert len(policy_telemetry.records) == 3
    assert policy_telemetry.records[0].step == 1

    frozen = frozen_policy_from_mlp(trained_policy.model)
    states = [s.state for s in dataset[:6]]

    value_targets = generate_value_targets(states, frozen, root_player=1, rollout_budgets=[1])
    trained_value = train_value_mlp(value_targets, hidden_dim=10, epochs=3, seed=7)
    value_telemetry = trained_value.training_log.telemetry
    assert value_telemetry.run_type == "value"
    assert value_telemetry.checkpoint_step == 3
    assert all(r.wall_time_seconds >= 0.0 for r in value_telemetry.records)

    q_targets = generate_q_targets(states, frozen, root_player=1, rollout_budgets=[1])
    trained_q = train_q_mlp(q_targets, hidden_dim=10, epochs=3, seed=9)
    q_telemetry = trained_q.training_log.telemetry
    assert q_telemetry.run_type == "q"
    assert q_telemetry.hyperparameters["epochs"] == 3
    assert [r.step for r in q_telemetry.records] == [1, 2, 3]
