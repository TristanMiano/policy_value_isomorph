"""Compare visit-frequency training methods on the frozen neural tic-tac-toe source."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from itertools import permutations
from pathlib import Path

from run_neural_tictactoe_control import PILOT, ROOT, write_table

import numpy as np
import torch

from policy_value_isomorph.choice_training import (
    board_geometry,
    fit_choice,
    move_log_probabilities,
    strict_board_ranks,
)
from policy_value_isomorph.neural_game_control import (
    certify_policy,
    fit_values,
    network,
    policy_choices,
    policy_tensors,
)
from policy_value_isomorph.pilot_metrics import matchup_rows
from policy_value_isomorph.pilot_protocol import GameGraph, board_id, frozen_policies, minimax_value
from policy_value_isomorph.policy_mlp import encode_state
from policy_value_isomorph.symmetry import canonicalize_state
from policy_value_isomorph.trajectory_utility import MoveSampler, trajectory_scores


def choices(sampler: MoveSampler, mover_scores: np.ndarray) -> np.ndarray:
    """Highest mover score, breaking ties by the lowest legal index."""
    grid = np.full(sampler.legal.shape, -np.inf)
    legal = sampler.legal >= 0
    grid[legal] = mover_scores[sampler.legal[legal]]
    return grid.argmax(1)


def disagreement_examples(
    sampler: MoveSampler,
    split: np.ndarray,
    counts: np.ndarray,
    predictions: dict,
    v_training_edges: np.ndarray,
) -> list[dict]:
    """Compare two observed count vectors with predicted counts at precisely the same exposure."""
    rows = []
    visits = np.bincount(sampler.parent, weights=counts, minlength=len(split))
    for group in ("train", "test"):
        for seed in sorted({seed for seed, _ in predictions}):
            q = predictions[seed, "Q_three_levels"]
            v = predictions[seed, "V_three_levels"]
            wrong = sampler.active & (split == group) & (choices(sampler, q) != sampler.fixed_move)
            if not wrong.any():
                continue
            i = min(np.flatnonzero(wrong), key=lambda j: (-visits[j], board_id(sampler.states[j])))
            # Count the first source move in each of 32 complete games started at exactly this board.
            audit = sampler.visits(np.array([i]), 32, 7501)
            ids = sampler.legal[i][sampler.legal[i] >= 0]
            exposure = int(audit[ids].sum())
            for edge in ids:
                state, move = sampler.edges[edge]
                rows.append(
                    dict(
                        group=group,
                        seed=seed,
                        board=board_id(state),
                        to_move=state.to_move,
                        move=move,
                        original_visits=int(visits[i]),
                        original_count=int(counts[edge]),
                        source_probability=counts[edge] / visits[i],
                        audit_visits=exposure,
                        audit_count=int(audit[edge]),
                        Q_probability=q[edge],
                        V_probability=v[edge],
                        Q_expected_original_count=visits[i] * q[edge],
                        V_expected_original_count=visits[i] * v[edge],
                        Q_expected_audit_count=exposure * q[edge],
                        V_expected_audit_count=exposure * v[edge],
                        V_candidate_used_in_fit=bool(v_training_edges[edge]),
                    )
                )
            break
    return rows


def ranked_terminal_values(
    graph: GameGraph, utilities: tuple[int, ...], depth: int | None = None
) -> np.ndarray:
    """Propagate fitted X utilities of O win/draw/X win through alternating best replies."""
    if depth is not None:
        # At a still-unfinished frontier, use the fitted value of a draw.
        ids = {s: i for i, s in enumerate(graph.states)}
        frontier = np.array(
            [
                utilities[(s.winner() or 0) + 1] if not graph.children[s] else utilities[1]
                for s in graph.states
            ]
        )
        for _ in range(depth):
            next_values = frontier.copy()
            for s, children in graph.children.items():
                if children:
                    choose = max if s.to_move == 1 else min
                    next_values[ids[s]] = choose(frontier[ids[t]] for t in children.values())
            frontier = next_values
        return frontier
    values = {}
    for s in reversed(graph.states):
        children = graph.children[s]
        if children:
            choose = max if s.to_move == 1 else min
            values[s] = choose(values[t] for t in children.values())
        else:
            values[s] = utilities[(s.winner() or 0) + 1]
    return np.array([values[s] for s in graph.states])


def study(
    directory: Path,
    alternate: bool = False,
    final_pass: bool = False,
    data_size: bool = False,
    refit_nontest: bool = False,
) -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((PILOT / "manifest.json").read_text())
    split_spec = manifest["split"].copy()
    if alternate:
        split_spec["salt"] = "ttt-choice-robustness-v1"
    graph = GameGraph.build(split_spec)
    control = ROOT / "experiments/neural_tictactoe_control"
    sampling = ROOT / "experiments/tictactoe_sampling"
    saved = torch.load(control / "weights.pt", weights_only=True, map_location="cpu")
    old_fits = json.loads((control / "training.json").read_text())["fits"]
    with (sampling / "agreement.csv").open(newline="") as stream:
        rare_epochs = {
            (r["method"].split("/")[0], int(r["fit_seed"])): int(r["selected_epoch"])
            for r in csv.DictReader(stream)
            if r["design"] == "low_visit_openings" and r["group"] == "test_uniform"
        }
    old_rare = torch.load(sampling / "low_visit_weights.pt", weights_only=True, map_location="cpu")
    source = network(10, 9)
    source.load_state_dict(saved["optimal_source"])
    source.eval().requires_grad_(False)
    decisions = [s for s in graph.states if graph.children[s]]
    original = policy_choices(source, *policy_tensors(decisions))
    certificate = certify_policy(graph, decisions, original)
    if certificate["worst_payoff_forgone"]:
        raise ValueError("The source gave up guaranteed payoff.")
    laws = {s: {a: 1} for s, a in zip(decisions, original)}
    sampler = MoveSampler(graph, laws)
    players = frozen_policies(manifest, PILOT / "weights.json")
    opponents = {k: {s: p(s) for s in decisions} for k, p in players.items()}
    reference_paths = sampler.rollout_paths(1, 8201)
    split = np.array([graph.split[s] for s in graph.states])
    original_split = split.copy()
    if refit_nontest:
        split[split == "validation"] = "train"
    parents = sampler.parent
    edges = torch.tensor(sampler.legal)
    cells = torch.tensor([a for _, a in sampler.edges])
    positions = torch.tensor([encode_state(s) for s in graph.states])
    raw_q = torch.cat((positions[parents], torch.nn.functional.one_hot(cells, 9)), 1)
    raw_v = positions[sampler.child]
    geometry = board_geometry(graph, sampler)
    q_geometry = torch.cat((geometry[parents], geometry[sampler.child], cells[:, None].float() / 8), 1)
    # Orthonormal columns preserve all raw information and distances. Match the geometric input width.
    generator = torch.Generator().manual_seed(7139)
    rotation, _ = torch.linalg.qr(torch.randn((q_geometry.shape[1], raw_q.shape[1]), generator=generator))
    expanded_raw = raw_q @ rotation.T
    # From a common parent, placing a mark at cell a increases this board feature by a/8.
    occupied_index = (positions[:, :9].abs() * torch.arange(9)).sum(1, keepdim=True) / 8
    v_geometry = torch.cat((geometry, occupied_index), 1)[sampler.child]
    guarantee = np.array([minimax_value(s) for s in graph.states])
    action_gaps = sampler.sign * (guarantee[parents] - guarantee[sampler.child])
    counts_data = np.load(sampling / "counts.npz")
    if not (
        np.array_equal(counts_data["edge_parent_board"], [board_id(s) for s, _ in sampler.edges])
        and np.array_equal(counts_data["edge_move"], cells.numpy())
    ):
        raise ValueError("Count archive and game graph disagree about edge order.")
    census = counts_data["optimal_net/known_once"]
    truth, _ = sampler.frequencies(census)
    if not np.array_equal(truth, cells.numpy() == sampler.fixed_move[parents]):
        raise ValueError("Archived counts disagree with the frozen neural policy.")
    targets = torch.tensor(truth, dtype=torch.float32)
    scores, outcomes, fits, checks, orders = [], [], {}, {}, []
    saved_choices, saved_models = {}, {}
    losses, diagnostic_p = [], {}
    coverage, sampled_groups = [], {}

    def independent_loss(
        name: str,
        design: str,
        seed: int,
        epoch: int,
        objective: str,
        model: torch.nn.Module,
        x: torch.Tensor,
        labels: torch.Tensor,
        group: str,
        ids: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            logits = model(x[ids])[:, 0]
            y = labels[ids]
            value = (
                torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
                if objective == "binary_log"
                else (logits.sigmoid() - y).square().mean()
            )
        losses.append(
            dict(
                design=design,
                model=name,
                seed=seed,
                selected_pass=epoch,
                group=group,
                objective=objective,
                boards=len(np.unique(parents[ids.numpy()])),
                candidates=len(ids),
                loss=value.item(),
            )
        )

    def evaluate(
        name: str,
        design: str,
        seed: int,
        scores_for_x: np.ndarray,
        utility: str,
        probabilities: np.ndarray | None = None,
    ) -> None:
        direct = choices(sampler, sampler.sign * scores_for_x)

        def table(actions: np.ndarray) -> dict:
            return {s: {int(actions[i]): 1} for i, s in enumerate(graph.states) if sampler.active[i]}

        methods = {"source": laws}
        x_values = {"direct": scores_for_x}
        copy_paths = MoveSampler(graph, table(direct)).rollout_paths(1, 8201)
        for continuation, paths in (("source", reference_paths), ("copies", copy_paths)):
            x_values["VNM_" + utility + "_" + continuation] = trajectory_scores(paths, scores_for_x)[
                "vnm_" + utility
            ][0]
        for decoder, values in x_values.items():
            selected = choices(sampler, sampler.sign * values)
            methods[decoder] = table(selected)
            saved_choices[f"{design}/{name}/{seed}/{decoder}"] = selected[sampler.active].astype(np.uint8)
            populations = {g: sampler.active & (split == g) for g in ("train", "validation", "test")}
            populations["all"] = sampler.active
            for group, mask in (populations | sampled_groups).items():
                if not mask.any():
                    continue
                match = selected[mask] == sampler.fixed_move[mask]
                gaps = action_gaps[sampler.legal[mask, selected[mask]]]
                edge_mask = mask[parents]
                scores.append(
                    dict(
                        design=design,
                        model=name,
                        seed=seed,
                        selection=decoder,
                        group=group,
                        boards=int(mask.sum()),
                        matches=int(match.sum()),
                        agreement=float(match.mean()),
                        direct_errors_corrected=int(
                            (match & (direct[mask] != sampler.fixed_move[mask])).sum()
                        ),
                        direct_matches_lost=int(
                            ((~match) & (direct[mask] == sampler.fixed_move[mask])).sum()
                        ),
                        other_optimal=int(((~match) & (gaps == 0)).sum()),
                        worse_moves=int((gaps > 0).sum()),
                        mean_payoff_forgone=float(gaps.mean()),
                        candidates=int(edge_mask.sum()),
                        direct_frequency_mse=(
                            float(np.square(probabilities[edge_mask] - truth[edge_mask]).mean())
                            if probabilities is not None
                            else None
                        ),
                    )
                )
        for row in matchup_rows("optimal_net", graph, methods, opponents, manifest):
            outcomes.append(dict(design=design, model=name, seed=seed, **row))

    train_parent = split[parents] == "train"
    val_parent = split[parents] == "validation"
    test_parent = split[parents] == "test"
    val_boards = torch.from_numpy(np.flatnonzero(sampler.active & (split == "validation")))
    designs = ("alternate_training_starts",) if alternate else ("all_training_starts", "rare_starts")
    if final_pass:
        designs = designs[:1]
    if refit_nontest:
        designs = ("all_nontest_fixed_duration",)
    if data_size:
        portions = {f"training_{n}_eighths": n for n in (1, 2, 4, 6, 8)}
        designs = tuple(portions)
    if data_size or refit_nontest:
        train_pool = sampler.active & (original_split == "train")
        # Randomize entire rotation/reflection groups; no move labels enter subset selection.
        orbit = np.array(
            [board_id(canonicalize_state(s)) if train_pool[i] else "" for i, s in enumerate(graph.states)]
        )
        all_orbits = np.unique(orbit[train_pool])
        original_train = original_split[parents] == "train"
        eligible = original_train & ~np.isin(sampler.child, sampler.child[~original_train])
        kept = edges.masked_fill(~torch.tensor(eligible)[edges.clamp_min(0)], -1)
        v_pool_size = int(
            (((kept >= 0) & (targets[kept.clamp_min(0)] > 0)).any(1) & ((kept >= 0).sum(1) >= 2)).sum()
        )
    seeds = (7301, 7302, 7303) if alternate else (7201, 7202, 7203)
    for design in designs:
        for rep, seed in enumerate(seeds):
            counts = (
                census
                if design != "rare_starts"
                else counts_data[f"optimal_net/{9101 + rep}/low_visit_openings/8192"]
            )
            if data_size:
                priority = np.random.default_rng(9401 + rep).permutation(all_orbits)
                n = len(priority) * portions[design] // 8
                included = train_pool & np.isin(orbit, priority[:n])
                counts = counts.copy()
                counts[train_parent & ~included[parents]] = 0
                sampled_groups = {"sampled_train": included, "omitted_train": train_pool & ~included}
                saved_choices[f"{design}/{seed}/sampled_train"] = included[sampler.active]
                coverage.append(
                    dict(
                        design=design,
                        fit_seed=seed,
                        subset_seed=9401 + rep,
                        symmetry_groups=n,
                        total_symmetry_groups=len(priority),
                        boards=int(included.sum()),
                        total_training_boards=int(train_pool.sum()),
                        candidates=int(included[parents].sum()),
                        retained_source_turns=int(counts[train_parent].sum()),
                    )
                )
            frequencies, total = sampler.frequencies(counts)
            observed = total > 0
            train_boards = torch.from_numpy(np.unique(parents[train_parent & observed]))
            # Only observed training groups enter the fit. Separate census validation labels select duration.
            fit_targets = torch.full_like(targets, torch.nan)
            observed_train = train_parent & observed
            fit_targets[observed_train] = torch.tensor(frequencies[observed_train], dtype=torch.float32)
            fit_targets[val_parent] = targets[val_parent]
            for family, x in (("V", raw_v), ("Q", raw_q)):
                if final_pass or data_size or refit_nontest:
                    continue
                eligible = train_parent.copy()
                valid = val_parent.copy()
                if family == "V":
                    eligible &= ~np.isin(sampler.child, sampler.child[~train_parent])
                    valid &= ~np.isin(sampler.child, sampler.child[test_parent])
                for objective in ("squared",) if alternate else ("squared", "binary_log"):
                    key = f"{design}/{family}_{objective}/{seed}"
                    ids = torch.from_numpy(np.flatnonzero(eligible & observed))
                    validation_ids = torch.from_numpy(np.flatnonzero(valid))
                    if objective == "squared" and not alternate:
                        model = network(x.shape[1], 1)
                        old_family = "q_mlp" if family == "Q" else "value_mlp"
                        state = (
                            saved[f"optimal_net/{old_family}/held_out/{seed}"]
                            if design == "all_training_starts"
                            else old_rare[f"{seed}/{family}"]
                        )
                        model.load_state_dict(state)
                        epoch = (
                            old_fits[f"optimal_net/{old_family}/held_out/{seed}"]["epoch"]
                            if design == "all_training_starts"
                            else rare_epochs[family, seed]
                        )
                    else:
                        model, fits[key] = fit_values(
                            x,
                            ids,
                            fit_targets[ids],
                            validation_ids,
                            fit_targets,
                            seed,
                            False,
                            log_loss=(objective == "binary_log"),
                            examples_per_pass=int(eligible.sum()),
                        )
                        epoch = fits[key]["epoch"]
                    for group, support in (("train", ids), ("validation", validation_ids)):
                        independent_loss(
                            f"{family}_{objective}",
                            design,
                            seed,
                            epoch,
                            objective,
                            model,
                            x,
                            fit_targets,
                            group,
                            support,
                        )
                    with torch.no_grad():
                        p = model(x)[:, 0].sigmoid().double().numpy()
                    evaluate(f"{family}_{objective}", design, seed, sampler.sign * p, "decay", p)
            # For V, a training or validation pair must only use input boards allowed in that split.
            # A choice provides an ordering if the selected move and a competitor both survive.
            eligible_v = (train_parent & observed & ~np.isin(sampler.child, sampler.child[~train_parent])) | (
                val_parent & ~np.isin(sampler.child, sampler.child[test_parent])
            )
            v_edges = edges.masked_fill(~torch.tensor(eligible_v)[edges.clamp_min(0)], -1)
            valid_v = v_edges >= 0
            supervised = (valid_v & (fit_targets[v_edges.clamp_min(0)] > 0)).any(1) & (valid_v.sum(1) >= 2)
            v_train = train_boards[supervised[train_boards]]
            v_val = val_boards[supervised[val_boards]]
            for name, x, rule in (
                ("Q_raw_choice", raw_q, "softmax"),
                ("Q_equal_width_raw", expanded_raw, "softmax"),
                ("Q_geometry_choice", q_geometry, "softmax"),
                ("Q_geometry_ordered", q_geometry, "first_acceptable"),
                ("Q_symmetric_ordered", q_geometry[:, :-1], "first_acceptable"),
                ("Q_three_levels", q_geometry[:, :-1], "three_levels"),
                ("V_geometry_choice", v_geometry, "softmax"),
                ("V_three_levels", geometry[sampler.child], "three_levels"),
            ):
                if alternate and name not in ("Q_geometry_ordered", "Q_three_levels", "V_three_levels"):
                    continue
                if (final_pass or data_size or refit_nontest) and name not in (
                    "Q_three_levels",
                    "V_three_levels",
                ):
                    continue
                key = f"{design}/{name}/{seed}"
                groups = (
                    (v_edges, v_train, v_val) if name.startswith("V_") else (edges, train_boards, val_boards)
                )
                budget = (
                    (v_pool_size if name.startswith("V_") else int(train_pool.sum()))
                    if data_size or refit_nontest
                    else None
                )
                model, fits[key] = fit_choice(
                    x, *groups, fit_targets, seed, rule, final_pass or data_size or refit_nontest, budget
                )
                info = fits[key]
                _, train_loss, val_loss = next(
                    r for r in info["epoch_train_validation_log_loss"] if r[0] == info["epoch"]
                )
                for group, support, loss in (
                    ("train", groups[1], train_loss),
                    ("validation", groups[2], val_loss),
                ):
                    if not len(support):
                        continue
                    losses.append(
                        dict(
                            design=design,
                            model=name,
                            seed=seed,
                            selected_pass=info["epoch"],
                            group=group,
                            objective="joint_log",
                            boards=len(support),
                            candidates=int((groups[0][support] >= 0).sum()),
                            loss=loss,
                        )
                    )
                if not data_size and name in (
                    "Q_geometry_ordered",
                    "Q_symmetric_ordered",
                    "Q_three_levels",
                    "V_three_levels",
                ):
                    saved_models[key] = model.state_dict()
                with torch.no_grad():
                    legal = edges[sampler.active] >= 0
                    logits = model(x).squeeze(-1)[edges[sampler.active].clamp_min(0)]
                    p = move_log_probabilities(logits, legal, rule).double().exp()[legal].numpy()
                evaluate(name, design, seed, sampler.sign * p, "decay", p)
                if design == "all_training_starts" and name in ("Q_three_levels", "V_three_levels"):
                    diagnostic_p[seed, name] = p
                    if name == "V_three_levels":
                        diagnostic_v_edges = np.zeros(len(census), dtype=bool)
                        used = v_edges[v_train]
                        diagnostic_v_edges[used[used >= 0].numpy()] = True
            print(f"Evaluated training design {design}, paired fit {rep + 1}/3.", flush=True)

    if not alternate and not final_pass and not data_size and not refit_nontest:
        # Existence control using only full-board comparisons. The resulting score is ordinal.
        rank = strict_board_ranks(sampler, census)
        checks["rank_levels"] = int(rank.max() + 1)
        checks["strict_comparisons"] = len(census) - int(sampler.active.sum())
        evaluate("V_board_rank", "all_boards_observed", 0, sampler.sign * rank[sampler.child], "decay")
        saved_choices["board_rank_by_state"] = rank
        for name, key in (("Q_full_squared", "q_mlp"), ("Q_full_binary_log", "q_mlp_log_loss")):
            model = network(raw_q.shape[1], 1)
            model.load_state_dict(saved[f"optimal_net/{key}/full_coverage/7201"])
            with torch.no_grad():
                p = model(raw_q)[:, 0].sigmoid().double().numpy()
            evaluate(name, "all_boards_observed", 7201, sampler.sign * p, "decay", p)
            archive_fit = old_fits[f"optimal_net/{key}/full_coverage/7201"]
            independent_loss(
                name,
                "all_boards_observed",
                7201,
                archive_fit["epoch"],
                "binary_log" if key.endswith("log_loss") else "squared",
                model,
                raw_q,
                targets,
                "train",
                torch.arange(len(census)),
            )

        # Fit a terminal utility ordering to observed choices under a stated best-reply assumption.
        train_ids = np.flatnonzero(sampler.active & (split == "train"))
        for order in permutations((-1, 0, 1)):
            backed = ranked_terminal_values(graph, order)
            picked = choices(sampler, sampler.sign * backed[sampler.child])
            match = int(truth[sampler.legal[train_ids, picked[train_ids]]].sum())
            orders.append(
                dict(o_win=order[0], draw=order[1], x_win=order[2], boards=len(train_ids), matches=match)
            )
        best = max(orders, key=lambda r: r["matches"])
        for depth in (0, 2, 4, None):
            values = ranked_terminal_values(graph, (best["o_win"], best["draw"], best["x_win"]), depth)
            name = "V_terminal_order" if depth is None else f"V_terminal_order_depth{depth}"
            evaluate(name, "rules_and_training_starts", 0, values[sampler.child], "terminal")
    saved_choices["decision_board"] = np.array([board_id(s) for s in decisions])
    saved_choices["state_board"] = np.array([board_id(s) for s in graph.states])
    np.savez_compressed(directory / "choices.npz", **saved_choices)
    if saved_models:
        torch.save(saved_models, directory / "ordered_weights.pt")
    if coverage:
        write_table(directory / "coverage.csv", coverage)
    if diagnostic_p and final_pass:
        np.savez_compressed(
            directory / "probabilities.npz",
            **{f"{seed}/{name}": p for (seed, name), p in diagnostic_p.items()},
            edge_parent_board=np.array([board_id(s) for s, _ in sampler.edges]),
            edge_move=cells.numpy(),
        )
    elif diagnostic_p:
        examples = disagreement_examples(sampler, split, census, diagnostic_p, diagnostic_v_edges)
        if examples:
            write_table(directory / "disagreements.csv", examples)
    for file, rows in (
        ("agreement", scores),
        ("games", outcomes),
        ("utility_orders", orders),
        ("losses", losses),
    ):
        if rows:
            write_table(directory / (file + ".csv"), rows)
    (directory / "training.json").write_bytes(
        (
            json.dumps(
                dict(
                    certificate=certificate,
                    checks=checks,
                    split=split_spec,
                    seeds=seeds,
                    q_geometry_inputs=q_geometry.shape[1],
                    v_geometry_inputs=v_geometry.shape[1],
                    fits=fits,
                ),
                indent=2,
            )
            + "\n"
        ).encode()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("original", "alternate"), default="original")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--final-pass", action="store_true", help="Evaluate the three-level models after all 120 passes."
    )
    mode.add_argument(
        "--data-size",
        action="store_true",
        help="Vary unique training positions at fixed update budget and test set.",
    )
    mode.add_argument(
        "--refit-nontest",
        action="store_true",
        help="Refit with the former validation boards at the fixed update budget.",
    )
    args = parser.parse_args()
    default = ROOT / "experiments/tictactoe_choice_training"
    if args.split == "alternate":
        default = default / "alternate"
    if args.refit_nontest:
        default = default / "nontest_refit"
    elif args.data_size:
        default = default / "data_size"
    elif args.final_pass:
        default = default / "final_pass"
    out = (args.output_dir or default).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as working:
        stage = Path(working) / "results"
        stage.mkdir()
        study(stage, args.split == "alternate", args.final_pass, args.data_size, args.refit_nontest)
        if out.exists():
            for file in stage.iterdir():
                if (out / file.name).read_bytes() != file.read_bytes():
                    raise ValueError(f"Replay differs at {file.name}. Supply a fresh output directory.")
        else:
            stage.rename(out)
