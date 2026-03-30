from policy_value_isomorph.policy import heuristic_policy_action
from policy_value_isomorph.policy_mlp import policy_mlp_action, train_policy_mlp
from policy_value_isomorph.sampling import generate_on_policy_dataset


def test_train_policy_mlp_loss_decreases():
    dataset = generate_on_policy_dataset(heuristic_policy_action, n_episodes=50)
    trained = train_policy_mlp(dataset, hidden_dim=24, learning_rate=0.03, epochs=40, seed=7)

    assert trained.training_log.losses[0] > trained.training_log.losses[-1]


def test_policy_mlp_action_is_legal_and_matches_reasonably_well():
    dataset = generate_on_policy_dataset(heuristic_policy_action, n_episodes=80)
    trained = train_policy_mlp(dataset, hidden_dim=24, learning_rate=0.03, epochs=60, seed=11)

    matches = 0
    for sample in dataset[:200]:
        pred = policy_mlp_action(sample.state, trained.model)
        assert pred in sample.state.legal_moves()
        if pred == sample.action:
            matches += 1

    assert matches >= 130
