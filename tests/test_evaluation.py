from policy_value_isomorph.evaluation import (
    action_agreement_rate,
    top_k_agreement_rate,
    value_calibration_curve,
    win_draw_loss_rate,
)
from policy_value_isomorph.policy import heuristic_policy_action
from policy_value_isomorph.policy_mlp import policy_mlp_action, train_policy_mlp
from policy_value_isomorph.rollout_value import generate_value_targets
from policy_value_isomorph.sampling import generate_on_policy_dataset
from policy_value_isomorph.value_mlp import train_value_mlp, value_mlp_predict


def test_action_agreement_and_top_k_are_reasonable_for_trained_policy_mlp():
    dataset = generate_on_policy_dataset(heuristic_policy_action, n_episodes=90)
    trained = train_policy_mlp(dataset, hidden_dim=24, learning_rate=0.03, epochs=70, seed=19)

    states = [sample.state for sample in dataset[:180]]
    agree = action_agreement_rate(
        states,
        policy_a=heuristic_policy_action,
        policy_b=lambda s: policy_mlp_action(s, trained.model),
    )

    assert agree >= 0.62

    def score_by_match(state, move):
        pred = policy_mlp_action(state, trained.model)
        return 1.0 if move == pred else 0.0

    top3 = top_k_agreement_rate(
        states,
        reference_policy=heuristic_policy_action,
        scored_policy=score_by_match,
        k=3,
    )
    assert top3 >= agree


def test_win_draw_loss_rates_sum_to_one_and_are_nontrivial():
    rates = win_draw_loss_rate(heuristic_policy_action, heuristic_policy_action, n_games=20)
    total = rates.win_rate + rates.draw_rate + rates.loss_rate

    assert abs(total - 1.0) < 1e-9
    assert rates.draw_rate > 0.0


def test_value_calibration_curve_has_expected_shape():
    policy_ds = generate_on_policy_dataset(heuristic_policy_action, n_episodes=60)
    states = [sample.state for sample in policy_ds[:120]]
    targets = generate_value_targets(
        states,
        heuristic_policy_action,
        root_player=1,
        rollout_budgets=[1],
    )

    trained_value = train_value_mlp(targets, hidden_dim=24, learning_rate=0.03, epochs=50, seed=23)
    curve = value_calibration_curve(targets, lambda s: value_mlp_predict(s, trained_value.model), n_bins=6)

    assert len(curve.bins) == 6
    assert sum(b.count for b in curve.bins) == len(targets)
