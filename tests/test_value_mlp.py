from policy_value_isomorph.policy import heuristic_policy_action
from policy_value_isomorph.rollout_value import generate_value_targets
from policy_value_isomorph.sampling import generate_on_policy_dataset
from policy_value_isomorph.value_mlp import train_value_mlp, value_mlp_predict


def test_train_value_mlp_loss_decreases():
    policy_ds = generate_on_policy_dataset(heuristic_policy_action, n_episodes=60)
    states = [sample.state for sample in policy_ds[:120]]
    targets = generate_value_targets(
        states,
        heuristic_policy_action,
        root_player=1,
        rollout_budgets=[1],
    )

    trained = train_value_mlp(targets, hidden_dim=24, learning_rate=0.03, epochs=60, seed=13)
    assert trained.training_log.losses[0] > trained.training_log.losses[-1]


def test_value_mlp_predictions_track_rollout_targets_reasonably():
    policy_ds = generate_on_policy_dataset(heuristic_policy_action, n_episodes=70)
    states = [sample.state for sample in policy_ds[:160]]
    targets = generate_value_targets(
        states,
        heuristic_policy_action,
        root_player=1,
        rollout_budgets=[1],
    )

    trained = train_value_mlp(targets, hidden_dim=32, learning_rate=0.03, epochs=80, seed=17)

    mae = 0.0
    for t in targets:
        pred = value_mlp_predict(t.state, trained.model)
        mae += abs(pred - t.value)
    mae /= len(targets)

    assert mae <= 0.40
