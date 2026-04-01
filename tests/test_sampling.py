from policy_value_isomorph.policy import heuristic_policy_action
from policy_value_isomorph.sampling import (
    augment_dataset_with_symmetries,
    generate_off_policy_dataset,
    generate_on_policy_dataset,
    reduce_dataset_by_canonical_symmetry,
)


def test_generate_on_policy_dataset_returns_legal_state_action_pairs():
    dataset = generate_on_policy_dataset(heuristic_policy_action, n_episodes=2)

    assert dataset
    for sample in dataset:
        assert sample.action in sample.state.legal_moves()
        assert not sample.state.is_terminal()


def test_generate_off_policy_dataset_is_reproducible_for_seed():
    data_a = generate_off_policy_dataset(n_episodes=3, seed=123)
    data_b = generate_off_policy_dataset(n_episodes=3, seed=123)

    assert data_a == data_b


def test_generate_datasets_validate_episode_count():
    try:
        generate_on_policy_dataset(heuristic_policy_action, n_episodes=0)
        assert False, "expected ValueError"
    except ValueError:
        pass

    try:
        generate_off_policy_dataset(n_episodes=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_augment_dataset_with_symmetries_expands_by_eight():
    dataset = generate_on_policy_dataset(heuristic_policy_action, n_episodes=1)
    augmented = augment_dataset_with_symmetries(dataset)
    assert len(augmented) == 8 * len(dataset)
    for sample in augmented:
        assert sample.action in sample.state.legal_moves()


def test_reduce_dataset_by_canonical_symmetry_collapses_equivalent_samples():
    dataset = generate_on_policy_dataset(heuristic_policy_action, n_episodes=1)
    augmented = augment_dataset_with_symmetries(dataset)
    reduced = reduce_dataset_by_canonical_symmetry(augmented)
    assert len(reduced) < len(augmented)
    assert len(reduce_dataset_by_canonical_symmetry(reduced)) == len(reduced)
