"""Frozen policies and board populations for the first tic-tac-toe pilot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Any, Callable

from .policy import heuristic_policy_action
from .policy_mlp import TinyMLPPolicy, policy_mlp_action, train_policy_mlp, untrained_policy_mlp
from .sampling import StateActionSample, generate_off_policy_dataset
from .symmetry import canonicalize_state
from .tictactoe import TicTacToeState

State = TicTacToeState
Law = dict[int, Fraction]
Policy = Callable[[State], Law]


def board_id(state: State) -> str:
    return "".join("O.X"[cell + 1] for cell in state.board)


def board_split(state: State, spec: dict) -> str:
    if state == State.initial():
        return "test"
    key = spec["salt"] + ":" + board_id(canonicalize_state(state))
    bucket = int.from_bytes(hashlib.sha256(key.encode("ascii")).digest()[:4], "big") % 100
    if bucket < spec["train_end"]:
        return "train"
    return "validation" if bucket < spec["validation_end"] else "test"


@dataclass
class GameGraph:
    """All oriented positions in increasing ply order, stopping at terminal states."""

    states: list[State]
    children: dict[State, dict[int, State]]
    split: dict[State, str]

    @classmethod
    def build(cls, split_spec: dict) -> GameGraph:
        initial = State.initial()
        pending = [initial]
        children: dict[State, dict[int, State]] = {}
        while pending:
            state = pending.pop()
            if state not in children:
                moves = {a: state.apply_move(a) for a in state.legal_moves()}
                children[state] = moves
                pending.extend(moves.values())
        states = sorted(children, key=lambda s: (-s.board.count(0), board_id(s)))
        return cls(states, children, {s: board_split(s, split_spec) for s in states})


@cache
def minimax_value(state: State) -> int:
    """Perfect-play return from fixed X perspective, including terminal boards."""
    if state.is_terminal():
        return state.terminal_return(1)
    choose = max if state.to_move == 1 else min
    return choose(minimax_value(state.apply_move(move)) for move in state.legal_moves())


def minimax_action(state: State) -> int:
    return max(state.legal_moves(), key=lambda a: (state.to_move * minimax_value(state.apply_move(a)), -a))


def miss_win_action(state: State) -> int:
    legal = state.legal_moves()
    alternatives = [a for a in legal if state.apply_move(a).winner() != state.to_move]
    if 0 < len(alternatives) < len(legal):
        return alternatives[0]
    return heuristic_policy_action(state)


def point_policy(action: Callable[[State], int]) -> Policy:
    return lambda state: {action(state): Fraction(1)}


def uniform_policy(state: State) -> Law:
    legal = state.legal_moves()
    return dict.fromkeys(legal, Fraction(1, len(legal)))


def freeze_weights(manifest: dict) -> dict:
    """Rebuild the inference checkpoints without evaluating reconstruction outcomes."""
    spec = manifest["source_training"]
    observations = generate_off_policy_dataset(spec["episodes"], seed=spec["data_seed"])
    states = {sample.state for sample in observations}
    train = sorted((s for s in states if board_split(s, manifest["split"]) == "train"), key=board_id)
    samples = [StateActionSample(s, heuristic_policy_action(s)) for s in train]
    params = {k: spec[k] for k in ("hidden_dim", "learning_rate", "epochs", "seed")}
    model = train_policy_mlp(samples, **params).model
    models = {"trained_net": asdict(model)}
    spec = manifest["random_opponents"]
    for seed in spec["seeds"]:
        models[f"random_net_{seed}"] = asdict(untrained_policy_mlp(spec["hidden_dim"], seed=seed))
    return {"version": manifest["version"], "source_training_boards": len(train), "models": models}


def fingerprint(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def frozen_policies(manifest: dict, weights_path: Path) -> dict[str, Policy]:
    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    if fingerprint(weights) != manifest["weights_sha256"]:
        raise ValueError("frozen policy checkpoint fingerprint differs from the pilot manifest")
    mixture = Fraction(*manifest["heuristic_mixture"])

    def stochastic(state: State) -> Law:
        law = {a: p * (1 - mixture) for a, p in uniform_policy(state).items()}
        law[heuristic_policy_action(state)] += mixture
        return law

    policies = {
        "minimax": point_policy(minimax_action), "miss_win": point_policy(miss_win_action),
        "heuristic": point_policy(heuristic_policy_action), "uniform": uniform_policy,
        "stochastic_heuristic": stochastic,
    }
    for name, params in weights["models"].items():
        model = TinyMLPPolicy(**params)
        policies[name] = point_policy(lambda s, m=model: policy_mlp_action(s, m))
    return policies
