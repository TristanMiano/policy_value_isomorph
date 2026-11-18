from fractions import Fraction

import pytest

from policy_value_isomorph.pilot_exact import decode_behavior, evaluate_source, play_match
from policy_value_isomorph.pilot_protocol import GameGraph, State, minimax_action, minimax_value, uniform_policy
from policy_value_isomorph.pilot_recovery import training_inputs
from policy_value_isomorph.tictactoe import state_from_rows


@pytest.fixture(scope="module")
def graph() -> GameGraph:
    return GameGraph.build({"salt": "ttt-pilot-v1", "train_end": 65, "validation_end": 80})


def test_minimax_source_recovers_every_fixed_tie_choice(graph: GameGraph) -> None:
    policy = {s: {minimax_action(s): Fraction(1)} for s in graph.states if graph.children[s]}
    exact = evaluate_source(graph, policy)
    assert all(exact.value[s] == minimax_value(s) for s in graph.states)
    assert all(exact.best[s][0] == next(iter(law)) for s, law in policy.items())
    assert exact.best[State.initial()] == tuple(range(9))
    assert play_match(graph, policy, policy) == (0, 1, 0)


def test_rational_continuation_and_full_game_seats(graph: GameGraph) -> None:
    uniform = {s: uniform_policy(s) for s in graph.states if graph.children[s]}
    exact = evaluate_source(graph, uniform)
    state = state_from_rows(["XXO", "OO.", "X.X"], to_move=-1)
    assert exact.action_value[state] == {5: -1, 7: 0}
    assert exact.value[state] == Fraction(-1, 2)
    assert exact.best[state] == (5,)
    assert decode_behavior({state: uniform[state]})[state] == {5: Fraction(1, 2), 7: Fraction(1, 2)}
    terminal = [s for s in graph.states if not graph.children[s]]
    outcomes = [sum(exact.visitation[s] for s in terminal if s.terminal_return(1) == u) for u in (1, 0, -1)]
    assert sum(outcomes) == 1
    assert outcomes[0] - outcomes[2] == exact.value[State.initial()]
    assert play_match(graph, uniform, uniform) == pytest.approx([float(p) for p in outcomes])


def test_held_out_successor_inputs_are_excluded(graph: GameGraph) -> None:
    decisions, v_inputs = training_inputs(graph)
    for s in decisions + v_inputs:
        assert graph.split[s] == "train"
    evaluation_queries = {c for s in graph.states if graph.split[s] != "train" for c in graph.children[s].values()}
    assert set(v_inputs).isdisjoint(evaluation_queries)
    assert len(decisions) == 2980
