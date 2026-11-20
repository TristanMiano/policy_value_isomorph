import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from policy_value_isomorph.coverage_sampling import (
    enumerated_counts,
    random_openings,
    reference_visits,
    sample_games,
)
from policy_value_isomorph.pilot_protocol import GameGraph, State, minimax_action
from policy_value_isomorph.trajectory_utility import MoveSampler


@pytest.fixture(scope="module")
def graph():
    path = Path(__file__).resolve().parents[1] / "experiments/tictactoe_pilot/manifest.json"
    return GameGraph.build(json.loads(path.read_text())["split"])


@pytest.fixture(scope="module")
def sampler(graph):
    laws = {s: {minimax_action(s): 1} for s, moves in graph.children.items() if moves}
    return MoveSampler(graph, laws)


def test_random_prefixes_end_on_reachable_decisions(sampler):
    starts, cost = random_openings(256, np.random.default_rng(123))
    assert len(starts) == 256
    active = {s for i, s in enumerate(sampler.states) if sampler.active[i]}
    assert all(s in active for s in starts)
    assert cost >= sum(9 - s.board.count(0) for s in starts)


@pytest.mark.parametrize("design", ["empty_self", "empty_random", "random_openings", "low_visit_openings"])
def test_only_source_turns_count_against_the_budget(sampler, design):
    pool = np.flatnonzero(sampler.active)
    samples = sample_games(sampler, pool, design, (37, 128), 456)
    for budget, sample in samples.items():
        assert sample.counts.sum() == budget
        for i in np.flatnonzero(sample.counts):
            state, move = sampler.edges[i]
            assert move == minimax_action(state)
        assert 0 <= sample.extra_source_turns_to_finish <= 8
    assert np.all(samples[128].counts >= samples[37].counts)
    if design == "empty_self":
        assert samples[128].random_opponent_turns == 0
        assert np.count_nonzero(samples[128].counts) == 9
    else:
        assert samples[128].random_opponent_turns > 0


def test_empty_board_reach_counts_only_the_source_position(sampler):
    start = sampler.states.index(State.initial())
    own = reference_visits(sampler, False)
    opposed = reference_visits(sampler, True)
    assert own[start] == 1 and own.sum() == 9
    assert opposed[start] == 0.5
    assert np.count_nonzero(opposed) == 546
    assert np.all(opposed[~sampler.active] == 0)


def test_rare_start_tie_break_uses_counts_only_from_prior_games(sampler):
    # These boards can only enter through a restart, and each ends after one source move.
    reach = reference_visits(sampler, True)
    candidates = [
        i
        for i, s in enumerate(sampler.states)
        if sampler.active[i] and s.board.count(0) == 1 and reach[i] == 0
    ]
    pool = np.array(candidates[:2])
    for rule, expected in (("random_openings", [40, 54]), ("low_visit_openings", [47, 47])):
        sample = sample_games(sampler, pool, rule, (256,), 456)[256]
        visits = np.bincount(sampler.parent, weights=sample.counts, minlength=len(sampler.states))
        assert visits[pool].tolist() == expected


def test_one_enumerated_start_preserves_all_fixed_player_labels(graph, sampler):
    once, single_calls = enumerated_counts(sampler, graph, 1)
    repeated, repeated_calls = enumerated_counts(sampler, graph, 32)
    probabilities, denominators = sampler.frequencies(once)
    assert np.array_equal(repeated, 32 * once)
    assert repeated_calls == 32 * single_calls
    assert single_calls > once.sum()
    assert np.all(denominators >= 1)
    assert np.array_equal(probabilities, [a == minimax_action(s) for s, a in sampler.edges])
