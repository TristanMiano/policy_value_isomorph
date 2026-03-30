from policy_value_isomorph.policy import heuristic_policy_action
from policy_value_isomorph.policy_mlp import policy_mlp_action, train_policy_mlp
from policy_value_isomorph.rollout_value import (
    estimate_v_pi,
    frozen_policy_from_mlp,
    generate_value_targets,
    recovered_action_from_v,
)
from policy_value_isomorph.sampling import generate_on_policy_dataset
from policy_value_isomorph.tictactoe import state_from_rows


def test_rollout_value_sign_root_wins_immediately():
    # X to move can win with move 2.
    s = state_from_rows(["XX.", "OO.", "..."], to_move=1)
    v_x = estimate_v_pi(s, policy=heuristic_policy_action, root_player=1)
    v_o = estimate_v_pi(s, policy=heuristic_policy_action, root_player=-1)
    assert v_x == 1.0
    assert v_o == -1.0


def test_rollout_value_sign_root_loses_immediately_when_opponent_to_move():
    # O to move can win immediately on index 2; root is X.
    s = state_from_rows(["OO.", "XX.", "..."], to_move=-1)
    v_x = estimate_v_pi(s, policy=heuristic_policy_action, root_player=1)
    assert v_x == -1.0


def test_recovered_policy_picks_immediate_win_for_root_turn():
    s = state_from_rows(["XX.", "OO.", "..."], to_move=1)
    mv = recovered_action_from_v(s, policy=heuristic_policy_action, root_player=1)
    assert mv == 2


def test_recovered_policy_minimizes_root_value_on_opponent_turn():
    # O can win now at index 2; root is X, so recovered action should pick index 2 (minimizes X value).
    s = state_from_rows(["OO.", "XX.", "..."], to_move=-1)
    mv = recovered_action_from_v(s, policy=heuristic_policy_action, root_player=1)
    assert mv == 2


def test_recovered_policy_blocks_immediate_threat_when_it_improves_value():
    # X threatens index 2. O should block at 2 under both heuristic and recovered value.
    s = state_from_rows(["XX.", "O..", "..."], to_move=-1)
    heuristic_mv = heuristic_policy_action(s)
    recovered_mv = recovered_action_from_v(s, policy=heuristic_policy_action, root_player=-1)
    assert heuristic_mv == 2
    assert recovered_mv == 2


def test_generate_value_targets_with_configurable_rollout_budgets():
    states = [
        state_from_rows(["XX.", "OO.", "..."], to_move=1),
        state_from_rows(["XO.", "X..", ".O."], to_move=-1),
    ]

    targets = generate_value_targets(
        states,
        heuristic_policy_action,
        root_player=1,
        rollout_budgets=[1, 4],
    )

    assert len(targets) == 4
    budgets = [t.n_rollouts for t in targets]
    assert budgets.count(1) == 2
    assert budgets.count(4) == 2


def test_frozen_policy_from_mlp_matches_direct_mlp_actions():
    dataset = generate_on_policy_dataset(heuristic_policy_action, n_episodes=40)
    trained = train_policy_mlp(dataset, hidden_dim=16, learning_rate=0.03, epochs=40, seed=3)
    frozen_policy = frozen_policy_from_mlp(trained.model)

    for sample in dataset[:30]:
        assert frozen_policy(sample.state) == policy_mlp_action(sample.state, trained.model)
