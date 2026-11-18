import csv
import gzip
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
from policy_value_isomorph.neural_game_control import certify_policy, network, policy_choices, policy_tensors
from policy_value_isomorph.pilot_protocol import GameGraph, State, minimax_action, point_policy
from policy_value_isomorph.policy_mlp import encode_state
from policy_value_isomorph.trajectory_utility import MoveSampler, trajectory_scores

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
RESULTS = EXPERIMENTS / "neural_tictactoe_control"


@pytest.fixture(scope="module")
def graph():
    settings = json.loads((EXPERIMENTS / "tictactoe_pilot/manifest.json").read_text(encoding="utf-8"))
    return GameGraph.build(settings["split"])


def test_sampled_complete_trajectories_and_normalized_counts(graph):
    player = point_policy(minimax_action)
    laws = {s: player(s) for s in graph.states if graph.children[s]}
    sampler = MoveSampler(graph, laws)
    counts = sampler.visits(np.array([graph.states.index(State.initial())]), 16, 4)
    freq, totals = sampler.frequencies(counts)
    assert counts.sum() == 16 * 9
    assert np.isnan(freq[totals == 0]).all()
    assert np.all(freq[counts > 0] == 1)
    paths = sampler.rollout_paths(1, 7)
    local = np.arange(len(sampler.edges), dtype=float) % 11 - 5
    utilities = trajectory_scores(paths, local)
    ids = {edge: i for i, edge in enumerate(sampler.edges)}
    # Independently play actual games, starting with each possible opening move.
    for a in range(9):
        s = State.initial()
        edge = ids[s, a]
        trajectory = [local[edge]]
        s = s.apply_move(a)
        while not s.is_terminal():
            move = minimax_action(s)
            trajectory.append(local[ids[s, move]])
            s = s.apply_move(move)
        assert utilities["vnm_sum"][0][edge] == sum(trajectory)
        assert utilities["vnm_decay"][0][edge] == pytest.approx(sum(0.9**i*v for i, v in enumerate(trajectory)))
        assert utilities["vnm_terminal"][0][edge] == trajectory[-1]


def test_sampled_stochastic_branch_uses_continuation_probabilities(graph):
    from fractions import Fraction
    laws = {s: dict.fromkeys(moves, Fraction(1, len(moves))) for s, moves in graph.children.items() if moves}
    sampler = MoveSampler(graph, laws)
    paths = sampler.rollout_paths(64, 11)
    opening = next(i for i, (s, a) in enumerate(sampler.edges) if s == State.initial() and a == 4)
    assert np.all(paths[0][opening] == opening)
    assert len(np.unique(paths[1][opening])) == 8
    for step in paths[1:]:
        for edge in step[opening][step[opening] >= 0]:
            s, action = sampler.edges[edge]
            assert action in s.legal_moves()
    # X utility is positive for an X choice and negative for an O choice at the same estimated frequency.
    local = sampler.sign * np.arange(len(sampler.edges)) / len(sampler.edges)
    sum_samples = sum(np.where(step[opening] >= 0, local[step[opening]], 0) for step in paths)
    means, variances = trajectory_scores(paths, local)["vnm_sum"]
    assert means[opening] == pytest.approx(sum_samples.mean())
    assert variances[opening] == pytest.approx(sum_samples.var(ddof=1) / 64)


def test_published_neural_policy_and_paired_comparisons(graph):
    torch.set_num_threads(1)
    weights = torch.load(RESULTS / "weights.pt", map_location="cpu", weights_only=True)
    source = network(10, 9)
    source.load_state_dict(weights["optimal_source"])
    states = [s for s in graph.states if graph.children[s]]
    moves = policy_choices(source, *policy_tensors(states))
    actual = certify_policy(graph, states, moves)
    saved = json.loads((RESULTS / "training.json").read_text())["certificate"]
    assert actual == {k: saved[k] for k in actual}
    assert actual["optimal_moves"] == actual["boards"] == 4520
    with (RESULTS / "move_comparisons.csv").open() as f:
        rows = {(r["source"], r["method"]): r for r in csv.DictReader(f)
                if r["split"] == "test" and r["weighting"] == "uniform"}
    for player in {s for s, _ in rows}:
        for family in ("value_mlp", "transition_mlp", "q_mlp"):
            for kind in ("direct", "vnm_terminal", "vnm_sum", "vnm_decay",
                         "vnm_terminal_copies", "vnm_sum_copies", "vnm_decay_copies"):
                assert (player, family + "/held_out/7201/" + kind) in rows
    assert float(rows["optimal_net", "visit_table/direct"]["agreement"]) == 1
    assert float(rows["optimal_net", "q_mlp_log_loss/full_coverage/7201/direct"]["agreement"]) == 1
    recovered = network(19, 1)
    recovered.load_state_dict(weights["optimal_net/q_mlp_log_loss/full_coverage/7201"])
    pairs = [(s, a) for s in states for a in graph.children[s]]
    inputs = torch.tensor([encode_state(s) + [float(b == a) for b in range(9)] for s, a in pairs])
    with torch.no_grad():
        scores = dict(zip(pairs, recovered(inputs)[:, 0].sigmoid().tolist()))
    assert [max(graph.children[s], key=lambda a: scores[s, a]) for s in states] == moves
    with gzip.open(RESULTS / "visits.csv.gz", "rt") as stream:
        for row in csv.DictReader(stream):
            assert float(row["frequency"]) == pytest.approx(int(row["count"]) / int(row["parent_visits"]))
