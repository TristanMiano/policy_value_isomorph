from policy_value_isomorph.connect_four import ConnectFourState
from policy_value_isomorph.connect_four_pipeline import (
    generate_off_policy_dataset,
    generate_on_policy_dataset,
    generate_value_targets,
    random_policy_action,
)


def test_initial_state_and_move_application() -> None:
    s = ConnectFourState.initial()
    assert len(s.board) == 42
    assert len(s.legal_moves()) == 7

    s2 = s.apply_move(3)
    assert s2.board[5 * 7 + 3] == 1
    assert s2.to_move == -1


def test_winner_detection_horizontal() -> None:
    s = ConnectFourState.initial()
    for col in [0, 0, 1, 1, 2, 2, 3]:
        s = s.apply_move(col)
    assert s.winner() == 1
    assert s.is_terminal()


def test_dataset_and_targets_pipeline_shapes() -> None:
    on_data = generate_on_policy_dataset(random_policy_action, n_episodes=2, seed=1)
    off_data = generate_off_policy_dataset(n_episodes=2, seed=2)
    assert len(on_data) > 0
    assert len(off_data) > 0

    states = [sample.state for sample in on_data[:3]]
    targets = generate_value_targets(states, random_policy_action, root_player=1, rollout_budgets=[1, 2], seed=0)
    assert len(targets) == len(states) * 2
    for t in targets:
        assert -1.0 <= t.value <= 1.0
