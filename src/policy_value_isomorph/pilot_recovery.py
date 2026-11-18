"""Train the existing three MLPs using only the preregistered pilot inputs."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .pilot_protocol import GameGraph, Law, State
from .policy_mlp import TinyMLPPolicy, policy_mlp_action, policy_mlp_probabilities, train_policy_mlp
from .q_mlp import TinyMLPQ, generate_q_targets, q_mlp_predict, train_q_mlp
from .rollout_value import PolicyFn, generate_value_targets
from .sampling import StateActionSample
from .value_mlp import TinyMLPValue, train_value_mlp, value_mlp_predict


def sampled_policy(laws: dict[State, Law], seed: int) -> PolicyFn:
    rng = random.Random(seed)
    choices = {s: (tuple(law), tuple(map(float, law.values()))) for s, law in laws.items()}

    def sample(state: State) -> int:
        actions, weights = choices[state]
        if len(actions) == 1:
            return actions[0]
        return rng.choices(actions, weights=weights)[0]

    return sample


def training_inputs(graph: GameGraph) -> tuple[list[State], list[State]]:
    """Keep every held-out decision and all its V-query successors out of V fitting."""
    held_out_children = {
        child for s, moves in graph.children.items() if graph.split[s] != "train"
        for child in moves.values()
    }
    train = [s for s in graph.states if graph.split[s] == "train"]
    return [s for s in train if graph.children[s]], [s for s in train if s not in held_out_children]


@dataclass
class Recovery:
    policy: TinyMLPPolicy
    value: TinyMLPValue
    q: TinyMLPQ
    stochastic: bool
    counts: dict[str, int]


def fit_models(graph: GameGraph, laws: dict[State, Law], spec: dict) -> Recovery:
    stochastic = any(len(law) > 1 for law in laws.values())
    repeats = spec["stochastic_action_samples"] if stochastic else 1
    rollouts = spec["stochastic_rollouts"] if stochastic else 1
    decisions, value_inputs = training_inputs(graph)
    choose = sampled_policy(laws, spec["policy_label_seed"])
    actions = [StateActionSample(s, choose(s)) for s in decisions for _ in range(repeats)]
    v_targets = generate_value_targets(
        value_inputs, sampled_policy(laws, spec["v_rollout_seed"]), root_player=1, rollout_budgets=[rollouts]
    )
    q_targets = generate_q_targets(
        decisions, sampled_policy(laws, spec["q_rollout_seed"]), root_player=1, rollout_budgets=[rollouts]
    )
    fit = {key: spec[key] for key in ("hidden_dim", "epochs", "learning_rate")}
    return Recovery(
        train_policy_mlp(actions, seed=spec["policy_seed"], **fit).model,
        train_value_mlp(v_targets, seed=spec["value_seed"], **fit).model,
        train_q_mlp(q_targets, seed=spec["q_seed"], **fit).model,
        stochastic,
        {"policy_boards": len(decisions), "policy_labels": len(actions), "v_inputs": len(v_targets),
         "q_inputs": len(q_targets), "rollouts_per_target": rollouts},
    )


@dataclass
class Predictions:
    imitation: dict[State, dict[int, float]]
    v: dict[State, float]
    q: dict[State, dict[int, float]]
    v_move: dict[State, int]
    q_move: dict[State, int]


def predict(graph: GameGraph, models: Recovery) -> Predictions:
    """Raw fixed-X predictions, maximized by X and minimized by O, including terminal successors."""
    v = {s: value_mlp_predict(s, models.value) for s in graph.states}
    imitation, q, v_move, q_move = {}, {}, {}, {}
    for state in graph.states:
        children = graph.children[state]
        if not children:
            continue
        if models.stochastic:
            probabilities = policy_mlp_probabilities(state, models.policy)
            imitation[state] = {a: probabilities[a] for a in children}
        else:
            imitation[state] = {policy_mlp_action(state, models.policy): 1.0}
        q[state] = {a: q_mlp_predict(state, a, models.q) for a in children}
        v_move[state] = max(children, key=lambda a: (state.to_move * v[children[a]], -a))
        q_move[state] = max(children, key=lambda a: (state.to_move * q[state][a], -a))
    return Predictions(imitation, v, q, v_move, q_move)
