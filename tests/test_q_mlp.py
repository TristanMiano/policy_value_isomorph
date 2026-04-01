from policy_value_isomorph.evaluation import action_agreement_rate
from policy_value_isomorph.policy import heuristic_policy_action
from policy_value_isomorph.q_mlp import (
    estimate_q_pi,
    generate_q_targets,
    recovered_action_from_q,
    train_q_mlp,
)
from policy_value_isomorph.rollout_value import recovered_action_from_v
from policy_value_isomorph.sampling import generate_on_policy_dataset
from policy_value_isomorph.tictactoe import state_from_rows


def test_estimate_q_pi_matches_immediate_forced_win():
    s = state_from_rows(["XX.", "OO.", "..."], to_move=1)
    q_win = estimate_q_pi(s, 2, heuristic_policy_action, root_player=1)
    assert q_win == 1.0


def test_train_q_mlp_loss_decreases():
    policy_ds = generate_on_policy_dataset(heuristic_policy_action, n_episodes=60)
    states = [sample.state for sample in policy_ds[:120]]
    targets = generate_q_targets(
        states,
        heuristic_policy_action,
        root_player=1,
        rollout_budgets=[1],
    )

    trained = train_q_mlp(targets, hidden_dim=32, learning_rate=0.03, epochs=80, seed=31)
    assert trained.training_log.losses[0] > trained.training_log.losses[-1]


def test_direct_q_recovery_is_comparable_to_successor_v_recovery():
    policy_ds = generate_on_policy_dataset(heuristic_policy_action, n_episodes=90)
    states = [sample.state for sample in policy_ds[:180]]

    q_targets = generate_q_targets(
        states,
        heuristic_policy_action,
        root_player=1,
        rollout_budgets=[1],
    )
    trained_q = train_q_mlp(q_targets, hidden_dim=32, learning_rate=0.03, epochs=90, seed=37)

    q_agree = action_agreement_rate(
        states,
        policy_a=heuristic_policy_action,
        policy_b=lambda s: recovered_action_from_q(s, trained_q.model, root_player=1),
    )
    v_agree = action_agreement_rate(
        states,
        policy_a=heuristic_policy_action,
        policy_b=lambda s: recovered_action_from_v(s, heuristic_policy_action, root_player=1, n_rollouts=1),
    )

    assert q_agree >= 0.55
    assert abs(v_agree - q_agree) <= 0.30
