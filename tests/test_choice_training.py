import csv
import json
from itertools import product
from math import prod
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
from policy_value_isomorph.choice_training import board_geometry, move_log_probabilities, strict_board_ranks
from policy_value_isomorph.pilot_protocol import GameGraph, minimax_action
from policy_value_isomorph.symmetry import canonicalize_state
from policy_value_isomorph.trajectory_utility import MoveSampler


def test_three_learned_levels_integrate_all_outcomes_and_break_ties_by_cell():
    logits = torch.tensor(
        [[1.0, 0.5, -0.4], [20, 1, 4], [0.0, 2, -1], [1.5, 0.4, 2]], dtype=torch.float64, requires_grad=True
    )
    legal = torch.tensor([True, False, True, True])
    log_p = move_log_probabilities(logits, legal, "three_levels")
    mass = logits.detach().softmax(-1)
    expected = torch.zeros(4, dtype=torch.float64)
    for levels in product(range(3), repeat=3):
        cells = [0, 2, 3]
        winner = max(range(3), key=lambda i: levels[i])
        expected[cells[winner]] += prod(mass[a, k] for a, k in zip(cells, levels))
    torch.testing.assert_close(log_p.exp(), expected)
    (-log_p[2]).backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[1].abs().sum() == 0


def test_first_acceptable_conditions_on_at_least_one_legal_acceptance():
    p = move_log_probabilities(
        torch.zeros((1, 4)), torch.tensor([[True, False, True, True]]), "first_acceptable"
    ).exp()[0]
    torch.testing.assert_close(p, torch.tensor([4, 0, 2, 1]) / 7)


def test_geometry_is_identical_under_rotation_and_full_observed_board_ranks_are_strict():
    root = Path(__file__).resolve().parents[1] / "experiments"
    manifest = json.loads((root / "tictactoe_pilot/manifest.json").read_text())
    graph = GameGraph.build(manifest["split"])
    sampler = MoveSampler(graph, {s: {minimax_action(s): 1} for s in graph.states if graph.children[s]})
    indices = {s: i for i, s in enumerate(graph.states)}
    features = board_geometry(graph, sampler)
    assert torch.equal(features, features[[indices[canonicalize_state(s)] for s in graph.states]])
    with np.load(root / "tictactoe_sampling/counts.npz") as archive:
        counts = archive["optimal_net/known_once"]
    ranks = strict_board_ranks(sampler, counts)
    for row in sampler.legal[sampler.active]:
        edges = row[row >= 0]
        wins = edges[counts[edges] > 0]
        assert np.all(ranks[sampler.child[wins]] > ranks[sampler.child[edges[counts[edges] == 0]]])


def test_reported_losses_are_selected_training_objectives_and_counts_use_recorded_exposure():
    results = Path(__file__).resolve().parents[1] / "experiments/tictactoe_choice_training"
    fits = json.loads((results / "training.json").read_text())["fits"]
    with (results / "losses.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["objective"] == "joint_log":
                fit = fits["/".join(row[k] for k in ("design", "model", "seed"))]
                selected = next(r for r in fit["epoch_train_validation_log_loss"] if r[0] == fit["epoch"])
                assert float(row["loss"]) == selected[1 if row["group"] == "train" else 2]
    with (results / "disagreements.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {r["group"] for r in rows} == {"train", "test"}
    for group in ("train", "test"):
        candidates = [r for r in rows if r["group"] == group]
        assert sum(int(r["audit_count"]) for r in candidates) == 32
        assert (
            max(candidates, key=lambda r: float(r["Q_probability"]))["move"]
            != max(candidates, key=lambda r: int(r["audit_count"]))["move"]
        )
        for model in ("Q", "V"):
            assert sum(float(r[f"{model}_probability"]) for r in candidates) == pytest.approx(1.0)
            for r in candidates:
                for sample in ("original", "audit"):
                    assert float(r[f"{model}_expected_{sample}_count"]) == pytest.approx(
                        int(r[f"{sample}_visits"]) * float(r[f"{model}_probability"])
                    )


def test_nested_samples_keep_the_test_population_and_reproduce_the_full_training_fit():
    results = Path(__file__).resolve().parents[1] / "experiments/tictactoe_choice_training"
    with (
        np.load(results / "data_size/choices.npz") as curve,
        np.load(results / "final_pass/choices.npz") as full,
    ):
        assert np.array_equal(curve["decision_board"], full["decision_board"])
        for seed in (7201, 7202, 7203):
            previous = np.zeros(len(curve["decision_board"]), dtype=bool)
            for eighths in (1, 2, 4, 6, 8):
                current = curve[f"training_{eighths}_eighths/{seed}/sampled_train"]
                assert np.all(~previous | current)
                assert current.sum() > previous.sum()
                previous = current
            assert previous.sum() == 2980
            for model in ("V_three_levels", "Q_three_levels"):
                assert np.array_equal(
                    curve[f"training_8_eighths/{model}/{seed}/direct"],
                    full[f"all_training_starts/{model}/{seed}/direct"],
                )
