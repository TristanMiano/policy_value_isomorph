"""Replay the frozen tic-tac-toe measurements through the Phase III research interfaces."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from dataclasses import replace
from fractions import Fraction
from itertools import groupby
from pathlib import Path
from random import Random
from statistics import mean, stdev
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policy_value_isomorph.candidate_sets import legal_candidates
from policy_value_isomorph.comparison_suite import GameMethod, calibration, compare_game
from policy_value_isomorph.consequence_utility import (
    ExactLotteries,
    Utility,
    UtilityObservation,
    expectation,
    fit_utility,
)
from policy_value_isomorph.distillation import fit_scores, observe
from policy_value_isomorph.neural_game_control import (
    certify_policy,
    network,
    policy_choices,
    policy_tensors,
)
from policy_value_isomorph.pilot_exact import evaluate_source
from policy_value_isomorph.pilot_protocol import (
    GameGraph,
    State,
    board_id,
    fingerprint,
    frozen_policies,
    minimax_value,
)
from policy_value_isomorph.pilot_recovery import Recovery, predict
from policy_value_isomorph.policy_adapter import GameAdapter
from policy_value_isomorph.policy_mlp import TinyMLPPolicy, encode_state
from policy_value_isomorph.preference_probes import probe
from policy_value_isomorph.q_mlp import TinyMLPQ
from policy_value_isomorph.recovery_contract import Continuation, RecoveryContract, ValueArtifact
from policy_value_isomorph.recovery_matrix import FACTORS, RunResult, factor_seeds, run_matrix
from policy_value_isomorph.score_alignment import align, cardinal_mse
from policy_value_isomorph.score_encoding import encode_probabilities, sample_probabilities
from policy_value_isomorph.torch_mlp import TorchMLP
from policy_value_isomorph.trajectory_utility import MoveSampler, trajectory_scores
from policy_value_isomorph.value_atlas import AtlasRecovery, ValueAtlas
from policy_value_isomorph.value_mlp import TinyMLPValue

PILOT = ROOT / "experiments/tictactoe_pilot"
VISITS = ROOT / "experiments/neural_tictactoe_control"
PANEL = "experiments/tictactoe_pilot/manifest.json#opponent_families"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def indexed_csv(path, fields):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {tuple(r[k] for k in fields): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate rows in {path}")
    return indexed


def base_contract(source):
    return RecoveryContract(
        domain="tic-tac-toe",
        source=source,
        semantics="behavior",
        collection_access="frozen source",
        deployment_access="original player",
        population="reachable decisions or source self-play visits",
        candidates="all legal cells",
        decoder="source policy",
        ties="lowest cell for equal scores",
        normalization="source probabilities",
        units="probability",
        split="ttt-pilot-v1",
        scope="all reachable orientations, games from the empty board",
        game=True,
        opponent_panel=PANEL,
    )


def method(contract, name, scores, laws, provenance, **settings):
    artifact = ValueArtifact(
        replace(contract, **settings),
        name,
        provenance,
        "enumerated conditional on the identified weights and decision rule",
        {"reference": provenance, "rule": name, "source": contract.source},
    )
    if ValueArtifact.from_json(artifact.to_json()) != artifact:
        raise AssertionError("artifact failed round trip")
    return GameMethod(artifact, laws, scores)


def greedy(rows):
    return {s: {max(sorted(row), key=row.__getitem__): 1} for s, row in rows.items()}


def source_method(name, laws, graph, provenance):
    scores = {s: {a: float(p.get(a, 0)) for a in graph.children[s]} for s, p in laws.items()}
    return method(base_contract(name), "source", scores, laws, provenance)


def continuation(name, utility, discount=1, draws=None):
    return Continuation(
        utility,
        "tic-tac-toe",
        name,
        name,
        "first terminal position",
        "X positive, O negative, prefer mover's score",
        discount,
        draws,
    )


def frozen_methods(name, graph, states, laws, weights, board_rows):
    contract = base_contract(name)
    recovered = weights[name]
    random_player = name == "stochastic_heuristic"
    saved = Recovery(
        TinyMLPPolicy(**recovered["policy"]),
        TinyMLPValue(**recovered["value"]),
        TinyMLPQ(**recovered["q"]),
        random_player,
        {},
    )
    fitted = predict(graph, saved)
    logits = TorchMLP(saved.policy).predict([encode_state(s) for s in states]).tolist()
    policy_scores = {s: {a: row[a] for a in graph.children[s]} for s, row in zip(states, logits)}
    adapter = GameAdapter(name, law=laws.__getitem__)
    lotteries = ExactLotteries(laws.__getitem__, laws.__getitem__)
    encoded, log_scores, expected, v, q = {}, {}, {}, {}, {}
    observed = set()
    max_probability_error = 0.0
    payoff_pairs = {key: [] for key in ("state_v", "successor_v", "direct_q")}
    by_id = {board_id(s): s for s in states}
    for row in board_rows:
        s = by_id[row["board"]]
        if s in observed:
            raise AssertionError("duplicate decision board")
        observed.add(s)
        menu = legal_candidates(adapter, s)
        legal = menu.values
        original = adapter.probabilities(s)
        archive_law = tuple(map(Fraction, json.loads(row["source_probs"])))
        if (
            list(legal) != json.loads(row["legal_moves"])
            or menu.coverage(original)["omitted_probability"]
            or tuple(original[a] for a in legal) != archive_law
            or row["split"] != graph.split[s]
        ):
            raise AssertionError("source, menu or board split changed")
        encoding = encode_probabilities(archive_law, [1] * len(legal))
        if encoding.exact_probabilities() != tuple(map(Fraction, json.loads(row["behavior_exact_probs"]))):
            raise AssertionError("exact copy changed")
        encoded[s] = dict(zip(legal, encoding.exact_probabilities()))
        log_scores[s] = dict(zip(legal, encoding.scores))
        expected[s] = {a: s.to_move * expectation(lotteries.force(s, a), Utility()) for a in legal}
        if [s.to_move * x for x in expected[s].values()] != list(
            map(Fraction, json.loads(row["q_exact_x"]))
        ):
            raise AssertionError("exact payoff changed")
        if max(legal, key=expected[s].__getitem__) != int(row["return_move"]):
            raise AssertionError("exact payoff tie changed")
        v[s] = {a: s.to_move * fitted.v[graph.children[s][a]] for a in legal}
        q[s] = {a: s.to_move * fitted.q[s][a] for a in legal}
        if graph.split[s] == "test":
            payoff_pairs["state_v"].append((fitted.v[s], float(Fraction(row["v_exact_x"]))))
            for a in legal:
                truth = float(s.to_move * expected[s][a])
                payoff_pairs["successor_v"].append((s.to_move * v[s][a], truth))
                payoff_pairs["direct_q"].append((s.to_move * q[s][a], truth))
        if fitted.v_move[s] != int(row["v_move"]) or fitted.q_move[s] != int(row["q_move"]):
            raise AssertionError("saved payoff network changed a move")
        max_probability_error = max(
            max_probability_error,
            *(
                abs(fitted.imitation[s].get(a, 0) - p)
                for a, p in zip(legal, json.loads(row["imitation_probs"]))
            ),
        )
    if observed != set(states) or max_probability_error > 1e-12:
        raise AssertionError("missing decision or archived imitation probabilities changed")
    score_file = "experiments/tictactoe_pilot/results/recovery_weights.json"
    point = continuation(name, "terminal game points +1, 0, -1 for X")
    payoff = dict(
        semantics="return",
        units="game points",
        normalization="orient X points for mover",
        continuation=point,
    )
    return (
        [
            source_method(name, laws, graph, "experiments/tictactoe_pilot/weights.json"),
            method(
                contract,
                "behavior_exact",
                log_scores,
                encoded,
                "Task 61 source law",
                decoder="temperature sampling",
                temperature=1,
                base_weights="equal",
                units="log probability",
                normalization="log of source probabilities",
                deployment_access="complete lookup",
            ),
            method(
                contract,
                "return_exact",
                expected,
                greedy(expected),
                "enumerated terminal lotteries",
                decoder="VNM maximum",
                deployment_access="rules and source for both seats",
                **payoff,
            ),
            method(
                contract,
                "policy_mlp",
                policy_scores,
                fitted.imitation,
                score_file,
                units="logit",
                normalization="raw network scores",
                collection_access="sampled source actions",
                deployment_access="fitted weights",
                decoder="temperature sampling" if random_player else "direct maximum",
                temperature=1 if random_player else None,
                base_weights="equal" if random_player else None,
            ),
            *[
                method(
                    contract,
                    label,
                    rows,
                    greedy(rows),
                    score_file,
                    decoder="direct maximum",
                    collection_access="source-generated game payoffs",
                    deployment_access="fitted weights",
                    **payoff,
                )
                for label, rows in (("value_mlp", v), ("q_mlp", q))
            ],
        ],
        max_probability_error,
        payoff_pairs,
    )


def verify_calibration(source, pairs, archive, result):
    names = ("all", "(-inf,-0.6)", "[-0.6,-0.2)", "[-0.2,0.2)", "[0.2,0.6)", "[0.6,inf)")
    for model, points in pairs.items():
        bins = calibration(points, (-0.6, -0.2, 0.2, 0.6))
        for key, row in zip(names, bins, strict=True):
            archived = archive.pop((source, model, key))
            if row["entries"] != int(archived["n"]):
                raise AssertionError("payoff calibration population changed")
            for field, old in (("prediction", "mean_prediction"), ("target", "mean_exact"), ("mse", "mse")):
                if not archived[old] and row[field] is None:
                    continue
                delta = abs(row[field] - float(archived[old]))
                if delta > 1e-12:
                    raise AssertionError((source, model, key, field, delta))
                result["largest_difference"]["calibration_" + field] = max(
                    result["largest_difference"].get("calibration_" + field, 0.0), delta
                )
            result["calibration_rows_checked"] += 1


def compare_fixture(graph, methods, populations, opponents, archived, result):
    source = methods[0].artifact.contract.source
    names = {m.artifact.method for m in methods}
    expected_f = {
        k for k in archived[0] if k[0] == source and k[1] in names and (k[2], k[3]) in populations
    }
    expected_g = {k for k in archived[1] if k[0] == source and k[1] in names}
    tables = compare_game(graph, methods[0], methods[1:], populations, opponents, include_decisions=False)

    def compare(metric, actual, reference):
        if reference == "":
            if actual is not None:
                raise AssertionError(f"{metric} must be undefined")
            return
        error = abs(actual - float(reference))
        if not math.isclose(actual, float(reference), abs_tol=1e-12, rel_tol=1e-12):
            raise AssertionError((source, metric, actual, reference))
        result["largest_difference"][metric] = max(result["largest_difference"].get(metric, 0.0), error)

    for row in tables["fidelity"]:
        split, weighting = row["population_name"]
        key = (source, row["method"], split, weighting)
        expected_f.remove(key)
        old = archived[0][key]
        for new, prev in (
            ("boards", "boards"),
            ("positive_weight_boards", "positive_weight_boards"),
            ("weight_sum", "weight_sum"),
            ("agreement", "agreement"),
            ("tv", "total_variation"),
        ):
            compare(new, row[new], old[prev])
        result["fidelity_rows_checked"] += 1
        if split == "test" and weighting == "uniform":
            result["held_out"].append(
                {
                    k: row[k]
                    for k in (
                        "source",
                        "method",
                        "artifact_id",
                        "decoder",
                        "continuation",
                        "deployment_access",
                        "boards",
                        "agreement",
                        "tv",
                        "win",
                        "draw",
                        "loss",
                        "points",
                        "delta_win",
                        "delta_draw",
                        "delta_loss",
                        "delta_points",
                    )
                }
            )
            if source == "stochastic_heuristic" and row["method"] == "return_exact":
                result["random_source_fixed_subset"] = {
                    "boards_with_one_possible_choice": row["fixed_choice_weight"],
                    "agreement_on_those_boards": row["fixed_choice_agreement"],
                    "all_withheld_boards": row["boards"],
                    "mean_probability_distance_on_all_boards": row["tv"],
                }
    for row in tables["game_outcomes"]:
        seat = {"first": "X", "second": "O", "balanced": "balanced"}[row["seat"]]
        opponent = "head_to_head" if row["opponent"] == "source head-to-head" else row["opponent"]
        key = (source, row["method"], opponent, seat)
        expected_g.remove(key)
        old = archived[1][key]
        for k in ("win", "draw", "loss", "points", "delta_win", "delta_draw", "delta_loss", "delta_points"):
            compare(k, row[k], old[k.replace("points", "score")])
        result["game_rows_checked"] += 1
    if expected_f or expected_g:
        raise AssertionError("unmatched archived rows")


def board_weights(graph, states, laws, splits):
    visits = evaluate_source(graph, laws).visitation
    return {
        (split, weight): {
            s: 1 if weight == "uniform" else float(visits[s])
            for s in states
            if split == "all" or graph.split[s] == split
        }
        for split in splits
        for weight in ("uniform", "visitation")
    }


def archives(directory, fidelity, matchups):
    return (
        indexed_csv(directory / fidelity, ("source", "method", "split", "weighting")),
        indexed_csv(directory / matchups, ("source", "method", "opponent", "seat")),
    )


def audit_section():
    return {"fidelity_rows_checked": 0, "game_rows_checked": 0, "largest_difference": {}, "held_out": []}


def score_rows(sampler, values):
    rows = {}
    for (s, a), v in zip(sampler.edges, values, strict=True):
        rows.setdefault(s, {})[a] = float(v)
    return rows


def three_fit_inference(changes: list[float]) -> dict:
    """Two-sided inference for this archive's three independent fits on one fixed split."""
    if len(changes) != 3:
        raise ValueError("this archived comparison requires three paired fits")
    difference = mean(changes)
    se = stdev(changes) / math.sqrt(3)
    # For Student's t with two degrees of freedom, P(|T| <= x) = x / sqrt(2 + x²).
    interval = p_value = None
    if se > 0:
        critical = 0.95 * math.sqrt(2 / (1 - 0.95**2))
        interval = [difference - critical * se, difference + critical * se]
        t = abs(difference) / se
        length = math.hypot(t, math.sqrt(2))
        p_value = 2 / (length * (length + t))
    # Conditional on the number of nonzero differences, a fair sign is a coin toss.
    positives = sum(value > 0 for value in changes)
    negatives = sum(value < 0 for value in changes)
    nonzero = positives + negatives
    tail = sum(math.comb(nonzero, k) for k in range(min(positives, negatives) + 1))
    return {
        "paired_mean_ci95": interval,
        "paired_t_p": p_value,
        "sign_test_nonzero_pairs": nonzero,
        "sign_test_p": min(1.0, 2 * tail / 2**nonzero) if nonzero else None,
    }


def summarize_fits(rows):
    """Average each rule across saved fits and pair it with direct selection of the same fit."""
    by_method = {}
    for row in rows:
        if row["method"] == "source":
            continue
        family, _, seed, rule = row["method"].split("/")
        by_method.setdefault((family, rule), {})[seed] = row
    summaries = []
    comparisons = []
    for (family, rule), fits in by_method.items():
        direct = by_method[family, "direct"]
        if len(fits) != 3 or fits.keys() != direct.keys():
            raise AssertionError("incomplete set of paired visit fits")
        summary = {"model": family, "selection_rule": rule, "fits": len(fits), "metrics": {}}
        for field in ("agreement", "win", "draw", "loss", "points"):
            changes = [row[field] - direct[seed][field] for seed, row in fits.items()]
            values = [row[field] for row in fits.values()]
            metric = {
                "mean": mean(values),
                "sample_sd": stdev(values),
                "observed_range": [min(values), max(values)],
                "change_from_direct": mean(changes),
                "standard_error_of_change": stdev(changes) / math.sqrt(len(changes)),
            }
            if rule != "direct":
                metric.update(three_fit_inference(changes))
                comparisons.append(metric)
            summary["metrics"][field] = metric
        summaries.append(summary)
    # Holm adjustment across the entire family, including correlated outcomes.
    ordered = sorted(comparisons, key=lambda m: 1.0 if m["paired_t_p"] is None else m["paired_t_p"])
    adjusted = 0.0
    for rank, metric in enumerate(ordered):
        p = metric["paired_t_p"]
        if p is None:
            metric["holm_p"] = None
            continue
        adjusted = max(adjusted, min(1.0, (len(ordered) - rank) * p))
        metric["holm_p"] = adjusted
    return summaries


def visit_control(graph, states, opponents, refs, result):
    """Reload already trained networks; no source, split, weights or settings are selected here."""
    snapshots = torch.load(VISITS / "weights.pt", map_location="cpu", weights_only=True)
    training = load_json(VISITS / "training.json")
    player = network(10, 9)
    player.load_state_dict(snapshots["optimal_source"])
    inputs, legal = policy_tensors(states)
    moves = policy_choices(player, inputs, legal)
    cert = certify_policy(graph, states, moves)
    if any(cert[k] != training["certificate"][k] for k in cert):
        raise AssertionError("frozen optimal source certificate changed")
    result["source_certificate"] = cert
    name = "optimal_net"
    laws = {s: {a: 1} for s, a in zip(states, moves)}
    contract = base_contract(name)
    source = source_method(
        name, laws, graph, "experiments/neural_tictactoe_control/weights.pt#optimal_source"
    )
    methods = [source]
    sampler = MoveSampler(graph, laws)
    paths = sampler.rollout_paths(1, 8201)
    cells = torch.tensor([a for _, a in sampler.edges])
    boards = torch.tensor([encode_state(s) for s in graph.states])
    x = {
        "value_mlp": boards[sampler.child],
        "q_mlp": torch.cat((boards[sampler.parent], torch.nn.functional.one_hot(cells, 9).float()), 1),
    }
    counts = {}
    with gzip.open(VISITS / "visits.csv.gz", "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["source"] == name:
                counts[row["board"], int(row["move"])] = int(row["count"])
    observed = np.array([counts[board_id(s), a] for s, a in sampler.edges])
    labels, _ = sampler.frequencies(observed)
    if not np.array_equal(labels, [a in laws[s] for s, a in sampler.edges]):
        raise AssertionError("visit archive differs from frozen deterministic source")
    splits = np.array([graph.split[s] for s, _ in sampler.edges])
    result["visits"] = int(observed.sum())
    result["training"] = []
    result["example_predictions"] = []
    for family, features in x.items():
        train = splits == "train"
        if family == "value_mlp":
            train &= ~np.isin(sampler.child, sampler.child[splits != "train"])
        for seed in (7201, 7202, 7203):
            key = f"{family}/held_out/{seed}"
            net = network(features.shape[1], 1)
            net.load_state_dict(snapshots[name + "/" + key])
            with torch.no_grad():
                predictions = net(features)[:, 0].sigmoid().double().numpy()
            loss = float(np.square(predictions[train] - labels[train]).mean())
            fit = training["fits"][name + "/" + key]
            epoch = fit["epoch"]
            saved_loss = next(
                row[1] for row in fit["curve_epoch_training_selection_mse"] if row[0] == epoch
            )
            if (
                not math.isclose(loss, saved_loss, abs_tol=1e-7)
                or int(train.sum()) != fit["training_count"]
            ):
                raise AssertionError("recomputed loss or its denominator changed")
            result["training"].append(
                dict(
                    model=family,
                    seed=seed,
                    pass_number=epoch,
                    legal_moves=int(train.sum()),
                    loss=loss,
                    archived_loss=saved_loss,
                )
            )
            freqs = score_rows(sampler, predictions)
            direct_law = greedy(freqs)
            provenance = "experiments/neural_tictactoe_control/weights.pt#" + name + "/" + key
            direct = method(
                contract,
                key + "/direct",
                freqs,
                direct_law,
                provenance,
                decoder="direct maximum",
                units="visit frequency",
                normalization="independent fraction per candidate",
                collection_access="source games",
                deployment_access="fitted weights",
            )
            methods.append(direct)
            if seed == 7201:
                adapter = GameAdapter(name, law=laws.__getitem__)
                view = AtlasRecovery(
                    direct.artifact, freqs.__getitem__, direct_law.__getitem__, freqs.__getitem__
                )
                mismatches = (s for s in states if graph.split[s] == "test" and laws[s] != direct_law[s])
                for _, s in zip(range(2), mismatches):
                    atlas = ValueAtlas(adapter, adapter.probabilities, [view]).inspect(
                        s, depth=1, visits={a: counts[board_id(s), a] for a in graph.children[s]}
                    )
                    result["example_predictions"].append(
                        dict(
                            model=family,
                            seed=seed,
                            board=board_id(s),
                            split="test",
                            moves=[r["move"] for r in atlas["actions"]],
                            observed=[r["observed"] for r in atlas["actions"]],
                            predictions=[r["predicted_visit_count"] for r in atlas["actions"]],
                            visits=atlas["actions"][0]["visits"],
                        )
                    )
            copies = MoveSampler(graph, direct_law).rollout_paths(1, 8201)
            for suffix, rollout, actor in (("", paths, name), ("_copies", copies, key + "/direct")):
                scores = trajectory_scores(rollout, sampler.sign * predictions)
                for rule, (means, _) in scores.items():
                    desc = {
                        "vnm_sum": "sum of signed fitted frequencies",
                        "vnm_decay": "discounted sum of signed fitted frequencies",
                        "vnm_terminal": "signed fitted frequency on last move",
                    }[rule]
                    rows = score_rows(sampler, sampler.sign * means)
                    methods.append(
                        method(
                            contract,
                            key + "/" + rule + suffix,
                            rows,
                            greedy(rows),
                            provenance,
                            decoder="VNM maximum",
                            units="signed visit score",
                            normalization="X positive, oriented for mover",
                            collection_access="source games",
                            deployment_access="rules and " + actor,
                            continuation=continuation(actor, desc, 0.9 if rule == "vnm_decay" else 1, 1),
                        )
                    )
    compare_fixture(
        graph, methods, board_weights(graph, states, laws, ("all", "test")), opponents, refs, result
    )
    result["paired_fits"] = summarize_fits(result["held_out"])


def interface_checks(laws):
    """Exercise the remaining research interfaces on a frozen stochastic pilot row."""
    s = State.initial()
    adapter = GameAdapter("stochastic_heuristic", law=laws.__getitem__)
    observation = observe(adapter, s, lambda st, a: (float(a == 4),), "probability", Random(0))
    encoded = tuple(range(9))
    if observation.target != tuple(float(laws[s][a]) for a in encoded):
        raise AssertionError("probability supervision changed the pilot row")
    try:
        probe(adapter, s, "empty", strategy="elimination", budget=2, seed=0)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("an archive of full-menu probabilities cannot answer fresh menu queries")

    # A second interface explicitly supplies the perfect-reply rule on any offered
    # subset. These are new queries to a known rule, not labels from the archive.
    def pick(state, menu, rng):
        return max(sorted(menu), key=lambda a: state.to_move * minimax_value(state.apply_move(a)))

    controlled = GameAdapter(
        "best guaranteed payoff, lowest cell ties",
        fixed=lambda state: pick(state, state.legal_moves(), None),
        choose_menu=pick,
    )
    controlled_probe = probe(controlled, s, "empty", strategy="elimination", budget=8, seed=0)
    if [q.choice for q in controlled_probe.history] != list(range(8)):
        raise AssertionError("changed-menu perfect-reply rule changed at the empty board")
    lottery = ExactLotteries(laws.__getitem__, laws.__getitem__)
    alternatives = {a: lottery.force(s, a) for a in encoded}
    fitted_utility = fit_utility(
        [UtilityObservation(alternatives, max(laws[s], key=laws[s].get), 1)], [-1, 0, 1]
    )
    scores = {a: float(laws[s][a]) for a in encoded}

    def anchored(row):
        return align(row, "anchor", units="probability difference", reference="cell zero", anchor=0)

    alignment_error = cardinal_mse(anchored(scores), anchored({a: p + 1 for a, p in scores.items()}))
    calls = []

    def recover(seeds):
        fit = fit_scores([observation], "probability", hidden=3, epochs=2, seed=seeds["initialization"])
        calls.append(fit.loss_nats)
        probabilities = sample_probabilities(fit.scores(observation.features), [1] * 9, 1)
        contract = replace(
            base_contract("stochastic_heuristic"),
            decoder="temperature sampling",
            temperature=1,
            base_weights="equal",
            deployment_access="fitted weights",
        )
        artifact = ValueArtifact(
            contract,
            "two-update interface check",
            "archived opening frequencies",
            "two initializations on one fixed row",
            {
                "loss_nats": fit.loss_nats,
                "seeds": seeds,
                "weights": {k: v.tolist() for k, v in fit.model.state_dict().items()},
            },
        )
        return RunResult(artifact, {"empty": {str(a): p for a, p in zip(encoded, probabilities)}})

    factors = factor_seeds(49, {k: 2 if k == "initialization" else 1 for k in FACTORS})
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "runs.sqlite"
        definition = {
            "fixture": "Task 61 randomized priority probabilities on empty board",
            "method": "probability distillation, two full-batch Adam updates",
        }
        matrix = run_matrix(path, definition, factors, recover)
        if run_matrix(path, definition, factors, recover) != matrix or len(calls) != 2:
            raise AssertionError("completed fits were changed or rerun on resume")
    return {
        "fresh_menu_access": "unavailable from the Task 61 full-menu records",
        "additional_perfect_reply_queries": [
            {"menu": list(q.menu), "response": q.choice} for q in controlled_probe.history
        ],
        "training_choice": "most probable opening move of the randomized priority source",
        "draw_values_and_matches": [[float(u.draw), n] for u, n in fitted_utility.matches],
        "alignment_squared_error": alignment_error,
        "initial_fits": len(calls),
        "extra_fits_on_resume": 0,
        "probability_distillation_loss_nats": calls,
        "pair_distance": matrix["pairwise"][0]["mean_tv"],
    }


def review():
    torch.set_num_threads(1)
    manifest = load_json(PILOT / "manifest.json")
    run = load_json(PILOT / "results/run.json")
    if fingerprint(manifest) != run["protocol_sha256"]:
        raise ValueError("pilot settings differ from the archive")
    for file, digest in run["sha256"].items():
        if hashlib.sha256((PILOT / "results" / file).read_bytes()).hexdigest() != digest:
            raise ValueError(f"archived pilot fixture changed: {file}")
    graph = GameGraph.build(manifest["split"])
    states = [s for s in graph.states if graph.children[s]]
    players = frozen_policies(manifest, PILOT / "weights.json")
    laws = {name: {s: player(s) for s in states} for name, player in players.items()}
    opponents = {}
    for name, weight in manifest["opponent_families"].items():
        members = (
            [f"random_net_{seed}" for seed in manifest["random_opponents"]["seeds"]]
            if name == "random_net"
            else [name]
        )
        opponents.update({m: (weight / len(members), laws[m]) for m in members})
    result = {
        "fixture": manifest["version"],
        "source_definitions": {
            **manifest["source_definitions"],
            "optimal_net": "Frozen neural player choosing the lowest cell with the best outcome guaranteed against perfect replies on all 4,520 decision boards.",
        },
        "opponent_definitions": manifest["opponent_definitions"],
        "method_definitions": {
            "pilot": manifest["method_definitions"],
            "visit_networks": {
                "source": "The frozen optimal neural player, optimal_net.",
                "value_mlp": "Score the resulting board with the fitted frequency of the move reaching it.",
                "q_mlp": "Score the board and candidate move together with their fitted move frequency.",
                "direct": "Choose the legal move with the highest fitted frequency, breaking ties by lowest cell number.",
                "VNM": "Von Neumann–Morgenstern selection chooses the best expected whole-game utility after each candidate. Count fitted frequencies positively on X moves and negatively on O moves. X maximizes and O minimizes, with lowest-cell ties.",
                "vnm_terminal": "VNM with the signed fitted frequency of the last move of the game.",
                "vnm_sum": "VNM with the sum of signed fitted frequencies from the candidate to game end.",
                "vnm_decay": "VNM with the same sum, weighting the candidate by one and each next move by 0.9 times the previous weight.",
                "_copies": "Both players use the fitted network's direct rule to complete each trial game. Without this suffix both use the frozen original player.",
            },
        },
        "table_fields": {
            "held_out": "One equal vote per withheld decision board for imitation. Game probabilities enumerate full games from the empty opening, with half weight per seat and one quarter per opponent family.",
            "agreement": "Fraction of boards reproducing the fixed original move. Null for a randomized source.",
            "tv": "Mean total variation distance, the minimum move probability mass that must be reassigned at a board to reproduce the source. Smaller means closer.",
            "win, draw, loss, points": "Probabilities of each result and mean points per game, with a win worth +1 and a loss -1. Delta subtracts the original player's result on the same opponent panel.",
            "artifact_id": "Content identifier of the score reference and its selection rules. The method field is the original archive key.",
            "VNM maximum": "Choose the best expected utility of a complete play for the mover. Continuation names both players used after forcing a move. Null rollout count denotes exact enumeration.",
            "training": "Final mean squared error per legal training candidate, in squared move-frequency units, evaluated at the saved training pass. Seed identifies the fit.",
            "paired_fits": "Three equally weighted initializations of each recovery network, one board split, frozen source and opponent panel. Each selection rule and continuation is defined on its held_out row. Agreement and outcome rates use fractions. Points are win probability minus loss probability, in points per game.",
            "sample_sd, observed_range": "Sample standard deviation (spread across the three fits, using divisor two) and their actual minimum and maximum. The frozen original player is one exactly enumerated reference.",
            "change_from_direct, standard_error_of_change": "Average difference from direct selection on the same weights and its standard error (sample standard deviation of paired differences divided by the square root of three).",
            "paired_mean_ci95, paired_t_p": "Individual 95% confidence interval for the mean paired difference and two-sided p-value under zero mean difference using Student's t with two degrees of freedom. Assumes independent, normally distributed fit differences on this fixed source, split and panel. Null when the variance estimate is zero.",
            "holm_p": "Two-sided paired t p-value adjusted by Holm's procedure across all 60 comparisons (two networks, six VNM rules, five outcomes). The intervals are individual, without multiplicity adjustment.",
            "sign_test_nonzero_pairs, sign_test_p": "Count of nonzero fit differences and unadjusted two-sided exact probability of a sign imbalance at least as extreme if either direction is equally likely, conditional on that count. Null if all differences are zero. This tests direction rather than the mean.",
            "example_predictions": "Per-cell predictions multiplied by the observed visits to the withheld board, in the listed legal cell order. Observed gives actual source counts at the same denominator.",
            "largest_difference": "Maximum absolute replay minus archived difference across the listed checks, in each field's own units.",
            "interfaces": "Small software checks. Each restricted menu records the cells offered and the answer of the known perfect-reply rule. Each draw-value pair gives the game points assigned to a draw and the number of matching choices on one randomized priority opening. Alignment error compares score differences from cell zero. The two opening fits report mean negative natural log probability per source move, weighted by its true probability; pair distance is the move probability mass reassigned between the two fits.",
        },
        "pilot": audit_section(),
        "visit_networks": audit_section(),
    }
    refs = archives(PILOT / "results", "fidelity.csv", "matchups.csv")
    calibration_ref = indexed_csv(PILOT / "results/calibration.csv", ("source", "prediction", "bin"))
    result["pilot"]["calibration_rows_checked"] = 0
    weights = load_json(PILOT / "results/recovery_weights.json")
    with gzip.open(PILOT / "results/boards.csv.gz", "rt", newline="") as stream:
        seen = []
        for name, rows in groupby(csv.DictReader(stream), key=lambda r: r["source"]):
            seen.append(name)
            methods, error, pairs = frozen_methods(name, graph, states, laws[name], weights, rows)
            verify_calibration(name, pairs, calibration_ref, result["pilot"])
            compare_fixture(
                graph,
                methods,
                board_weights(graph, states, laws[name], ("all", "train", "validation", "test")),
                opponents,
                refs,
                result["pilot"],
            )
            result["pilot"].setdefault("max_imitation_probability_error_by_source", {})[name] = error
        if (
            calibration_ref
            or seen != manifest["sources"]
            or (result["pilot"]["fidelity_rows_checked"], result["pilot"]["game_rows_checked"])
            != tuple(map(len, refs))
        ):
            raise AssertionError("incomplete Task 61 replay")
    result["pilot"]["source_board_cases"] = len(states) * len(seen)
    result["pilot"]["source_move_cases"] = sum(len(graph.children[s]) for s in states) * len(seen)
    visit_control(
        graph,
        states,
        opponents,
        archives(VISITS, "move_comparisons.csv", "games.csv"),
        result["visit_networks"],
    )
    result["interfaces"] = interface_checks(laws["stochastic_heuristic"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the complete replay as JSON")
    args = parser.parse_args()
    data = json.dumps(review(), indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(data, end="")
    else:
        args.output.write_text(data, encoding="utf-8")
