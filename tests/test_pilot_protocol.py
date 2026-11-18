import json
from fractions import Fraction
from pathlib import Path

from policy_value_isomorph.pilot_protocol import GameGraph, State, frozen_policies, miss_win_action
from policy_value_isomorph.symmetry import canonicalize_state
from policy_value_isomorph.tictactoe import state_from_rows

PILOT = Path(__file__).resolve().parents[1] / "experiments/tictactoe_pilot"
MANIFEST = json.loads((PILOT / "manifest.json").read_text())


def test_graph_population_and_orbit_split() -> None:
    graph = GameGraph.build(MANIFEST["split"])
    assert len(graph.states) == 5478
    assert sum(bool(moves) for moves in graph.children.values()) == 4520
    assert sum(map(len, graph.children.values())) == 16167
    assert graph.split[State.initial()] == "test"
    assert all(graph.split[s] == graph.split[canonicalize_state(s)] for s in graph.states)


def test_frozen_sources_opponents_and_reserves() -> None:
    policies = frozen_policies(MANIFEST, PILOT / "weights.json")
    state = State.initial()
    assert policies["stochastic_heuristic"](state)[4] == Fraction(3, 4) + Fraction(1, 36)
    assert all(sum(policy(state).values()) == 1 for policy in policies.values())
    assert len([p for p in policies if p.startswith("random_net_")]) == 5
    pilot_seeds = {MANIFEST["source_training"]["seed"], *MANIFEST["random_opponents"]["seeds"]}
    reserved = MANIFEST["reserved_confirmation"]
    assert pilot_seeds.isdisjoint(reserved["source_network_seeds"] + reserved["opponent_network_seeds"])
    winning = state_from_rows(["XX.", "OO.", "..."], to_move=1)
    assert miss_win_action(winning) == 5
