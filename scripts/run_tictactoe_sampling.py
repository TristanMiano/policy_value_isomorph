"""Compare fixed sampling budgets, then recover the certified neural player from its own observed moves."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_neural_tictactoe_control import write_table
from policy_value_isomorph.coverage_sampling import (
    enumerated_counts,
    random_openings,
    reference_visits,
    sample_games,
)
from policy_value_isomorph.neural_game_control import (
    certify_policy,
    fit_values,
    network,
    policy_choices,
    policy_tensors,
)
from policy_value_isomorph.pilot_metrics import matchup_rows
from policy_value_isomorph.pilot_protocol import GameGraph, State, board_id, frozen_policies
from policy_value_isomorph.policy_mlp import encode_state
from policy_value_isomorph.trajectory_utility import MoveSampler, trajectory_scores

PILOT = ROOT / "experiments/tictactoe_pilot"
DESIGNS = ("empty_self", "empty_random", "random_openings", "low_visit_openings")
BUDGETS = (512, 2048, 8192)
SEEDS = range(9101, 9111)


def summarize_counts(
    sampler: MoveSampler,
    counts: np.ndarray,
    truth: np.ndarray,
    eligible: np.ndarray,
    self_visits: np.ndarray,
    random_visits: np.ndarray,
) -> dict:
    freq, denominator = sampler.frequencies(counts)
    board_visits = np.bincount(sampler.parent, weights=counts, minlength=len(eligible))
    observed = board_visits > 0
    eligible_edges = eligible[sampler.parent]
    visits = board_visits[eligible]
    error_sum = 0.5 * np.nansum(np.abs(freq - truth)[eligible_edges])
    observed_n = np.sum(eligible & observed)

    def covered_mass(weights: np.ndarray) -> float:
        return float(weights[eligible & observed].sum() / weights[eligible].sum())

    return dict(
        boards=int(eligible.sum()),
        observed=int(observed_n),
        observations=int(visits.sum()),
        min_visits=int(visits.min()),
        median_visits=float(np.median(visits)),
        max_visits=int(visits.max()),
        uniform_coverage=float(observed_n / eligible.sum()),
        self_visit_coverage=covered_mass(self_visits),
        opposed_visit_coverage=covered_mass(random_visits),
        observed_mean_tv=float(error_sum / observed_n) if observed_n else None,
        unobserved_possible_moves=int(
            ((counts == 0) & (truth > 0) & eligible_edges & (denominator > 0)).sum()
        ),
    )


def compare_randomized_frequencies(
    sampler: MoveSampler,
    truth: np.ndarray,
    training: np.ndarray,
    counts_archive: dict[str, np.ndarray],
    source: str,
) -> list[dict]:
    """Compare empirical probabilities on the same training boards under both restart rules."""
    legal_moves = (sampler.legal >= 0).sum(axis=1)
    uniform = 1 / legal_moves[sampler.parent]
    rows = []
    for seed in SEEDS:
        observed, distances = {}, {}
        for design in ("random_openings", "low_visit_openings"):
            counts = counts_archive[f"{source}/{seed}/{design}/{BUDGETS[-1]}"]
            freq, totals = sampler.frequencies(counts)
            estimated = np.where(totals > 0, freq, uniform)
            observed[design] = np.bincount(sampler.parent, weights=counts, minlength=len(training)) > 0
            distances[design] = np.bincount(
                sampler.parent, weights=0.5 * np.abs(estimated - truth), minlength=len(training)
            )
        common = training & observed["random_openings"] & observed["low_visit_openings"]
        for population, mask in (("observed_by_both", common), ("all_training_uniform_fallback", training)):
            for design, distance in distances.items():
                rows.append(
                    dict(
                        source=source,
                        seed=seed,
                        budget=BUDGETS[-1],
                        design=design,
                        population=population,
                        boards=int(mask.sum()),
                        observed=int((mask & observed[design]).sum()),
                        mean_tv=float(distance[mask].mean()) if mask.any() else None,
                    )
                )
    return rows


def fit_and_score(
    sampler: MoveSampler,
    graph: GameGraph,
    modelspec: Sequence[tuple[str, torch.Tensor, np.ndarray, np.ndarray]],
    counts: np.ndarray,
    validation_truth: torch.Tensor,
    exact_moves: np.ndarray,
    groups: dict[str, np.ndarray],
    paths: list[np.ndarray],
    seed: int,
) -> tuple[list[dict], dict, dict]:
    freq, denom = sampler.frequencies(counts)
    visits = np.bincount(sampler.parent, weights=counts, minlength=len(graph.states))
    trained = {}
    choices = {}
    metrics = []
    by_count = {
        "train_no_visit": visits == 0,
        "train_one_visit": visits == 1,
        "train_2_to_7": (visits >= 2) & (visits <= 7),
        "train_8_plus": visits >= 8,
    }
    populations = {**groups, **{k: mask & groups["training"] for k, mask in by_count.items()}}
    for family, x, train, val in modelspec:
        observed = np.flatnonzero(train & (denom > 0))
        indices = torch.from_numpy(observed)
        # Equal number of training updates and equal weight per observed candidate in every design.
        model, fit = fit_values(
            x,
            indices,
            torch.tensor(freq[observed], dtype=torch.float32),
            torch.from_numpy(np.flatnonzero(val)),
            validation_truth,
            seed,
            False,
            examples_per_pass=int(train.sum()),
        )
        with torch.no_grad():
            estimates = model(x)[:, 0].sigmoid().double().numpy()
        signed = sampler.sign * estimates
        discounted, _ = trajectory_scores(paths, signed)["vnm_decay"]
        for decoder, values in (("direct", signed), ("discounted", discounted)):
            grid = np.full(sampler.legal.shape, -np.inf)
            grid[sampler.parent, [a for _, a in sampler.edges]] = sampler.sign * values
            chosen = grid.argmax(axis=1)
            agreement = chosen == exact_moves
            method = family + "/" + decoder
            choices[method] = {
                s: {int(chosen[i]): 1} for i, s in enumerate(graph.states) if sampler.active[i]
            }
            for name, weights in populations.items():
                total = weights.sum()
                matched = weights @ agreement.astype(float)
                metrics.append(
                    dict(
                        method=method,
                        group=name,
                        boards=int(np.count_nonzero(weights)),
                        weight_sum=float(total),
                        matching_weight=float(matched),
                        agreement=float(matched / total) if total else None,
                        training_edges=len(observed),
                        selected_epoch=fit["epoch"],
                    )
                )
        trained[family] = model.state_dict()
    return metrics, choices, trained


def run(out: Path) -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((PILOT / "manifest.json").read_text(encoding="utf-8"))
    graph = GameGraph.build(manifest["split"])
    decisions = [s for s in graph.states if graph.children[s]]
    idx = {s: i for i, s in enumerate(graph.states)}
    model = network(10, 9)
    archive = ROOT / "experiments/neural_tictactoe_control/weights.pt"
    model.load_state_dict(torch.load(archive, map_location="cpu", weights_only=True)["optimal_source"])
    model.eval().requires_grad_(False)
    original_moves = policy_choices(model, *policy_tensors(decisions))
    certificate = certify_policy(graph, decisions, original_moves)
    if certificate["worst_payoff_forgone"]:
        raise ValueError(f"Original neural player lost guaranteed payoff: {certificate}")
    print(
        f"The frozen neural player (optimal_net) chooses its highest-scoring legal move. "
        f"It matched the lowest-numbered move preserving the best guaranteed outcome on "
        f"{certificate['same_minimax_move']} of {certificate['boards']} decision boards.",
        flush=True,
    )
    print(manifest["source_definitions"]["heuristic"], flush=True)
    print(
        manifest["source_definitions"]["stochastic_heuristic"] + " Its table name is stochastic_heuristic.",
        flush=True,
    )
    optimal = {s: {a: 1} for s, a in zip(decisions, original_moves)}
    exact_moves = np.full(len(idx), -1)
    exact_moves[[idx[s] for s in decisions]] = original_moves
    policies = frozen_policies(manifest, PILOT / "weights.json")
    opponents = {n: {s: player(s) for s in decisions} for n, player in policies.items()}
    split = np.array([graph.split[s] for s in graph.states])
    sampled, metrics, games, references, pool_stats = [], [], [], [], []
    counts_archive, checkpoints = {}, {}
    pools = {}
    for seed in SEEDS:
        starts, work = random_openings(16384, np.random.default_rng(seed + 1000))
        pool = np.array([idx[s] for s in starts if graph.split[s] == "train"])
        pools[seed] = pool
        pool_stats.append(
            dict(
                seed=seed,
                proposals=len(starts),
                prefix_moves=work,
                eligible_proposals=len(pool),
                distinct_training_positions=len(np.unique(pool)),
            )
        )
    for name, laws in (("optimal_net", optimal), ("stochastic_heuristic", opponents["stochastic_heuristic"])):
        sampler = MoveSampler(graph, laws)
        true_probabilities = np.array([float(laws[s].get(a, 0)) for s, a in sampler.edges])
        self_visits = reference_visits(sampler, False)
        random_visits = reference_visits(sampler, True)
        for s in decisions:
            i = idx[s]
            references.append(
                dict(
                    source=name,
                    board=board_id(s),
                    split=split[i],
                    self_reach=self_visits[i],
                    random_opponent_reach=random_visits[i],
                )
            )
        opening = sampler.visits(np.array([idx[State.initial()]]), 256, 8100)
        census, repeated_calls = enumerated_counts(sampler, graph, 32)
        once, single_calls = enumerated_counts(sampler, graph, 1)
        if not sampler.randomized and not np.array_equal(census, 32 * once):
            raise ArithmeticError("Repeated deterministic games changed the move labels.")
        counts_archive[f"{name}/known_once"] = once
        for tag, observed, calls, starts in (
            ("published_empty", opening, int(opening.sum()), 256),
            ("known_once", once, single_calls, len(decisions)),
            ("published_census", census, repeated_calls, 32 * len(decisions)),
        ):
            for g, pop in (("all", sampler.active), ("train", sampler.active & (split == "train"))):
                sampled.append(
                    dict(
                        source=name,
                        design=tag,
                        seed="",
                        budget=calls,
                        group=g,
                        games_started=starts,
                        random_opponent_turns=0,
                        extra_source_turns_to_finish=0,
                        **summarize_counts(
                            sampler, observed, true_probabilities, pop, self_visits, random_visits
                        ),
                    )
                )
        tensor = torch.tensor([encode_state(s) for s in graph.states])
        cur, after = tensor[sampler.parent], tensor[sampler.child]
        actions = torch.nn.functional.one_hot(torch.tensor([a for _, a in sampler.edges]), 9).float()
        train = split[sampler.parent] == "train"
        val = split[sampler.parent] == "validation"
        test = split[sampler.parent] == "test"
        specs = [
            (
                "V",
                after,
                train & ~np.isin(sampler.child, sampler.child[~train]),
                val & ~np.isin(sampler.child, sampler.child[test]),
            ),
            ("transition", torch.cat((cur, after), 1), train, val),
            ("Q", torch.cat((cur, actions), 1), train, val),
        ]
        test_states = sampler.active & (split == "test")
        positive = test_states & (random_visits > 0)
        median = np.median(random_visits[positive])
        groups = dict(
            test_uniform=test_states,
            test_self=self_visits * test_states,
            test_random=random_visits * test_states,
            test_unreachable=test_states & (random_visits == 0),
            test_less_frequent=positive & (random_visits <= median),
            test_more_frequent=positive & (random_visits > median),
            training=sampler.active & (split == "train"),
        )
        if name == "optimal_net":
            paths = sampler.rollout_paths(1, 8201)
            targets = torch.tensor(sampler.frequencies(once)[0], dtype=torch.float32)
        for rep, seed in enumerate(SEEDS):
            for design in (*DESIGNS, "census"):
                if design != "census":
                    snapshots = sample_games(sampler, pools[seed], design, BUDGETS, seed)
                    for budget, snap in snapshots.items():
                        counts_archive[f"{name}/{seed}/{design}/{budget}"] = snap.counts.astype(
                            np.min_scalar_type(budget)
                        )
                        for g, pop in (
                            ("all", sampler.active),
                            ("train", sampler.active & (split == "train")),
                        ):
                            sampled.append(
                                dict(
                                    source=name,
                                    design=design,
                                    seed=seed,
                                    budget=budget,
                                    group=g,
                                    games_started=snap.games_started,
                                    random_opponent_turns=snap.random_opponent_turns,
                                    extra_source_turns_to_finish=snap.extra_source_turns_to_finish,
                                    **summarize_counts(
                                        sampler,
                                        snap.counts,
                                        true_probabilities,
                                        pop,
                                        self_visits,
                                        random_visits,
                                    ),
                                )
                            )
                    counts = snapshots[BUDGETS[-1]].counts
                else:
                    counts = once
                if name != "optimal_net" or rep >= 5:
                    continue
                fit_seed = 7201 + rep
                result, choices, fitted = fit_and_score(
                    sampler, graph, specs, counts, targets, exact_moves, groups, paths, fit_seed
                )
                for row in result:
                    metrics.append(dict(design=design, sampling_seed=seed, fit_seed=fit_seed, **row))
                for row in matchup_rows(name, graph, {"source": optimal, **choices}, opponents, manifest):
                    games.append(dict(design=design, sampling_seed=seed, fit_seed=fit_seed, **row))
                if design == "low_visit_openings":
                    checkpoints.update({f"{fit_seed}/{family}": state for family, state in fitted.items()})
            print(f"Completed sampling repetition {rep + 1} of {len(SEEDS)} for {name}.", flush=True)
        if sampler.randomized:
            comparisons = compare_randomized_frequencies(
                sampler, true_probabilities, sampler.active & (split == "train"), counts_archive, name
            )
            write_table(out / "randomized_frequencies.csv", comparisons)
    counts_archive["edge_parent_board"] = np.array([board_id(s) for s, a in sampler.edges])
    counts_archive["edge_move"] = np.array([a for s, a in sampler.edges])
    np.savez_compressed(out / "counts.npz", **counts_archive)
    torch.save(checkpoints, out / "low_visit_weights.pt")
    for rows in (metrics, games):
        for row in rows:
            rule = row["method"].rsplit("/", 1)[-1]
            row["selection"] = {"discounted": "VNM discounted sum", "direct": "Direct", "source": "Source"}[rule]
    for file, rows in (
        ("coverage", sampled),
        ("agreement", metrics),
        ("games", games),
        ("reference", references),
        ("openings", pool_stats),
    ):
        write_table(out / (file + ".csv"), rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/tictactoe_sampling")
    output = parser.parse_args().output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "results"
        stage.mkdir()
        run(stage)
        if output.exists():
            for f in stage.iterdir():
                if f.read_bytes() != (output / f.name).read_bytes():
                    raise ValueError(f"Replay differs at {f.name}. Use a fresh output directory.")
        else:
            stage.rename(output)
