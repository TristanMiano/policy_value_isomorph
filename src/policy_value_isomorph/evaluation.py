from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .rollout_value import StateValueTarget
from .tictactoe import Move, TicTacToeState

PolicyFn = Callable[[TicTacToeState], Move]
ActionScoreFn = Callable[[TicTacToeState, Move], float]
ValuePredictFn = Callable[[TicTacToeState], float]


@dataclass(frozen=True)
class WinDrawLossRate:
    """Result rates from policy-vs-policy evaluation.

    Rates are measured from `player_policy` perspective and sum to 1.0.
    """

    win_rate: float
    draw_rate: float
    loss_rate: float


@dataclass(frozen=True)
class CalibrationBin:
    """One value-calibration bin over predicted V values in [-1, +1]."""

    lower: float
    upper: float
    count: int
    mean_pred: float
    mean_true: float


@dataclass(frozen=True)
class CalibrationCurve:
    """Aggregated binned stats suitable for plotting a calibration curve."""

    bins: list[CalibrationBin]


def action_agreement_rate(states: Sequence[TicTacToeState], policy_a: PolicyFn, policy_b: PolicyFn) -> float:
    """Fraction of states where two policies choose the same action."""
    if not states:
        raise ValueError("states must be non-empty")

    matches = 0
    counted = 0
    for state in states:
        if state.is_terminal():
            continue
        counted += 1
        if policy_a(state) == policy_b(state):
            matches += 1

    if counted == 0:
        raise ValueError("states must include at least one non-terminal state")
    return matches / counted


def top_k_agreement_rate(
    states: Sequence[TicTacToeState],
    *,
    reference_policy: PolicyFn,
    scored_policy: ActionScoreFn,
    k: int,
) -> float:
    """Fraction of states where reference action appears in scored policy top-k.

    `scored_policy(state, move)` should output a score where larger is better.
    """
    if not states:
        raise ValueError("states must be non-empty")
    if k <= 0:
        raise ValueError("k must be >= 1")

    hits = 0
    counted = 0
    for state in states:
        legal = state.legal_moves()
        if not legal:
            continue

        counted += 1
        ranked = sorted(legal, key=lambda mv: (scored_policy(state, mv), -mv), reverse=True)
        top_k = set(ranked[: min(k, len(ranked))])
        if reference_policy(state) in top_k:
            hits += 1

    if counted == 0:
        raise ValueError("states must include at least one non-terminal state")
    return hits / counted


def _play_game(policy_x: PolicyFn, policy_o: PolicyFn) -> int:
    state = TicTacToeState.initial()
    while not state.is_terminal():
        move = policy_x(state) if state.to_move == 1 else policy_o(state)
        state = state.apply_move(move)
    return state.terminal_return(root_player=1)


def win_draw_loss_rate(player_policy: PolicyFn, opponent_policy: PolicyFn, n_games: int = 100) -> WinDrawLossRate:
    """Play both seat assignments and return W/D/L rates for `player_policy`.

    Half the games are run as X, and half as O (difference at most one game).
    """
    if n_games <= 0:
        raise ValueError("n_games must be >= 1")

    wins = 0
    draws = 0
    losses = 0

    for i in range(n_games):
        player_as_x = (i % 2 == 0)
        if player_as_x:
            result_for_x = _play_game(player_policy, opponent_policy)
            result_for_player = result_for_x
        else:
            result_for_x = _play_game(opponent_policy, player_policy)
            result_for_player = -result_for_x

        if result_for_player > 0:
            wins += 1
        elif result_for_player < 0:
            losses += 1
        else:
            draws += 1

    denom = float(n_games)
    return WinDrawLossRate(win_rate=wins / denom, draw_rate=draws / denom, loss_rate=losses / denom)


def value_calibration_curve(
    dataset: Sequence[StateValueTarget],
    predict_value: ValuePredictFn,
    n_bins: int = 8,
) -> CalibrationCurve:
    """Build binned calibration statistics for value predictions.

    Values are assumed to be in roughly [-1,+1]. Bin boundaries are fixed and
    can be plotted as (mean_pred, mean_true) points.
    """
    if not dataset:
        raise ValueError("dataset must be non-empty")
    if n_bins <= 0:
        raise ValueError("n_bins must be >= 1")

    width = 2.0 / n_bins
    pred_sums = [0.0 for _ in range(n_bins)]
    true_sums = [0.0 for _ in range(n_bins)]
    counts = [0 for _ in range(n_bins)]

    for sample in dataset:
        pred = predict_value(sample.state)
        idx = int((pred + 1.0) / width)
        idx = max(0, min(n_bins - 1, idx))

        pred_sums[idx] += pred
        true_sums[idx] += sample.value
        counts[idx] += 1

    bins: list[CalibrationBin] = []
    for idx in range(n_bins):
        lower = -1.0 + idx * width
        upper = lower + width
        if counts[idx] == 0:
            bins.append(CalibrationBin(lower=lower, upper=upper, count=0, mean_pred=0.0, mean_true=0.0))
        else:
            c = counts[idx]
            bins.append(
                CalibrationBin(
                    lower=lower,
                    upper=upper,
                    count=c,
                    mean_pred=pred_sums[idx] / c,
                    mean_true=true_sums[idx] / c,
                )
            )

    return CalibrationCurve(bins=bins)
