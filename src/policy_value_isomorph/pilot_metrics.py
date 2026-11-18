"""Board-level fidelity, calibration, and whole-game strength on fixed populations."""

from __future__ import annotations

from bisect import bisect_right
from math import fsum
from typing import Iterable, Iterator

from .pilot_exact import ExactReturns, PolicyTable, play_match, total_variation
from .pilot_protocol import GameGraph, State
from .pilot_recovery import Predictions


def fidelity_rows(
    source: str, graph: GameGraph, methods: dict[str, PolicyTable], exact: ExactReturns
) -> Iterator[dict]:
    """Average move differences and points forgone if the source plays both sides after this move."""
    original = methods["source"]
    stochastic = any(len(p) > 1 for p in original.values())
    for method, table in methods.items():
        diagnostics = {}
        for s, p in original.items():
            best = exact.best[s]
            q = table[s]
            fixed_mass = float(p.get(best[0], 0))
            optimal_mass = float(sum(p.get(a, 0) for a in best))
            returns_for_x = exact.action_value[s]
            best_return_for_x = returns_for_x[best[0]]
            # O prefers a smaller X payoff. Express the difference as points forgone by the mover.
            gaps = {move: float(s.to_move * (best_return_for_x - payoff))
                    for move, payoff in returns_for_x.items()}
            diagnostics[s] = (
                total_variation(p, q), fixed_mass, optimal_mass - fixed_mass, 1 - optimal_mass,
                1 - float(q.get(best[0], 0)), 1 - float(sum(q.get(a, 0) for a in best)),
                fsum(float(prob) * gaps[a] for a, prob in p.items()),
                fsum(float(prob) * gaps[a] for a, prob in q.items()),
            )
        for split in ("all", "train", "validation", "test"):
            states = [s for s in original if split == "all" or graph.split[s] == split]
            for weighting in ("uniform", "visitation"):
                weights = [1.0 if weighting == "uniform" else float(exact.visitation[s]) for s in states]
                mass = fsum(weights)
                row = dict(source=source, method=method, split=split, weighting=weighting,
                           boards=len(states), positive_weight_boards=sum(w > 0 for w in weights),
                           weight_sum=mass)
                names = ("total_variation", "source_fixed_match", "source_other_tie", "source_strict_loss",
                         "changed_from_exact_move", "selected_lower_return",
                         "source_mean_payoff_gap", "method_mean_payoff_gap")
                for i, name in enumerate(names):
                    row[name] = fsum(w * diagnostics[s][i] for s, w in zip(states, weights)) / mass if mass else None
                row["agreement"] = 1 - row["total_variation"] if mass and not stochastic else None
                yield row


def calibration_rows(source: str, graph: GameGraph, exact: ExactReturns, pred: Predictions) -> Iterator[dict]:
    """Equal weight per held-out V input or per legal candidate for the action panels."""
    samples: dict[str, list[tuple[float, float]]] = {"state_v": [], "successor_v": [], "direct_q": []}
    for s, actions in exact.action_value.items():
        if graph.split[s] != "test":
            continue
        samples["state_v"].append((pred.v[s], float(exact.value[s])))
        for a, truth in actions.items():
            samples["successor_v"].append((pred.v[graph.children[s][a]], float(truth)))
            samples["direct_q"].append((pred.q[s][a], float(truth)))
    edges = [-0.6, -0.2, 0.2, 0.6]
    labels = ["(-inf,-0.6)", "[-0.6,-0.2)", "[-0.2,0.2)", "[0.2,0.6)", "[0.6,inf)"]
    for model, pairs in samples.items():
        bins: list[list[tuple[float, float]]] = [[] for _ in labels]
        for estimate, target in pairs:
            bins[bisect_right(edges, estimate)].append((estimate, target))
        for label, points in zip(["all", *labels], [pairs, *bins]):
            n = len(points)
            yield dict(source=source, prediction=model, bin=label, n=n,
                       mean_prediction=fsum(p for p, _ in points) / n if n else None,
                       mean_exact=fsum(t for _, t in points) / n if n else None,
                       mse=fsum((p - t) ** 2 for p, t in points) / n if n else None)


WDL = tuple[float, float, float]


def average_wdl(weighted: Iterable[tuple[float, WDL]]) -> WDL:
    rows = list(weighted)
    return tuple(fsum(w * outcome[i] for w, outcome in rows) for i in range(3))


def matchup_rows(
    source: str, graph: GameGraph, methods: dict[str, PolicyTable],
    opponents: dict[str, PolicyTable], manifest: dict,
) -> Iterator[dict]:
    families = manifest["opponent_families"]
    members = {f: [f] for f in families}
    members["random_net"] = [f"random_net_{seed}" for seed in manifest["random_opponents"]["seeds"]]
    family_of = {name: family for family, names in members.items() for name in names}
    panel = {name: opponents[name] for name in family_of}
    panel["head_to_head"] = methods["source"]
    baseline: dict[tuple[str, str], WDL] = {}
    for method, player in methods.items():
        scores = {}
        for name, opponent in panel.items():
            scores[name, "X"] = play_match(graph, player, opponent)
            scores[name, "O"] = play_match(graph, opponent, player)[::-1]
        for seat in ("X", "O"):
            scores["panel", seat] = average_wdl(
                (weight / len(members[f]), scores[name, seat])
                for f, weight in families.items() for name in members[f]
            )
        for name in [*panel, "panel"]:
            scores[name, "balanced"] = average_wdl((0.5, scores[name, seat]) for seat in ("X", "O"))
        if method == "source":
            baseline = scores
        for (name, seat), (win, draw, loss) in scores.items():
            bw, bd, bl = baseline[name, seat]
            yield dict(source=source, method=method, opponent=name, family=family_of.get(name, name),
                       seat=seat, win=win, draw=draw, loss=loss, score=win-loss,
                       delta_win=win-bw, delta_draw=draw-bd, delta_loss=loss-bl,
                       delta_score=(win-loss)-(bw-bl))
