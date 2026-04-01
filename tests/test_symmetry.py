from policy_value_isomorph.symmetry import (
    SYMMETRIES,
    apply_symmetry_to_move,
    apply_symmetry_to_state,
    canonicalize_state,
    canonicalize_state_action,
)
from policy_value_isomorph.tictactoe import state_from_rows


def test_symmetries_are_eight_unique_permutations():
    maps = [sym.index_map for sym in SYMMETRIES]
    assert len(maps) == 8
    assert len(set(maps)) == 8
    for m in maps:
        assert sorted(m) == list(range(9))


def test_canonicalize_state_is_equal_across_symmetric_positions():
    state = state_from_rows(["X..", ".O.", "..X"], to_move=-1)
    rotated = apply_symmetry_to_state(state, SYMMETRIES[1])
    mirrored = apply_symmetry_to_state(state, SYMMETRIES[4])

    assert canonicalize_state(state) == canonicalize_state(rotated)
    assert canonicalize_state(state) == canonicalize_state(mirrored)


def test_canonicalize_state_action_remaps_action_consistently():
    state = state_from_rows(["X..", ".O.", "..."], to_move=1)
    action = 6

    canon_state_a, canon_action_a = canonicalize_state_action(state, action)
    rot_state = apply_symmetry_to_state(state, SYMMETRIES[1])
    rot_action = apply_symmetry_to_move(action, SYMMETRIES[1])
    canon_state_b, canon_action_b = canonicalize_state_action(rot_state, rot_action)

    assert canon_state_a == canon_state_b
    assert canon_action_a == canon_action_b
