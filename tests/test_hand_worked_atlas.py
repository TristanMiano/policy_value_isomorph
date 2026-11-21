"""Verify the hand-worked board's published numbers against legal game transitions."""

from collections.abc import Iterator
from fractions import Fraction
from math import exp
from pathlib import Path
from random import Random
from statistics import mean

import pytest

from policy_value_isomorph import TicTacToeState, estimate_q_pi

CELL_ORDER = (5, 8, 2, 0, 1, 3, 4, 6, 7)
REPORT = Path(__file__).resolve().parents[1] / "docs" / "hand_worked_atlas.md"


def source_move(available: list[int]) -> int:
    return next(move for move in CELL_ORDER if move in available)


def continuations(state: TicTacToeState, path: tuple[int, ...], utility: int,
                  uniform_mass: Fraction, source_mass: int) -> Iterator[tuple[tuple[int, ...], int, int, Fraction, int]]:
    """Enumerate legal paths, keeping X-oriented utility and both continuation laws."""
    if state.is_terminal():
        yield path, state.terminal_return(1), utility, uniform_mass, source_mass
        return
    legal = state.legal_moves()
    preferred = source_move(legal)
    for move in legal:
        chosen = int(move == preferred)
        yield from continuations(state.apply_move(move), path + (move,),
                                 utility + state.to_move * chosen, uniform_mass / len(legal),
                                 source_mass * chosen)


def test_hand_worked_atlas() -> None:
    board = TicTacToeState.initial()
    for move in (0, 3, 1, 4, 6, 7):
        board = board.apply_move(move)
    legal = board.legal_moves()
    assert board.to_move == 1 and legal == [2, 5, 8]
    observed = [source_move(legal) for _ in range(8)]
    counts = [observed.count(move) for move in legal]
    assert counts == [0, 8, 0]
    # Each pairwise victory contributes one rank unit on this three-move menu.
    ranks = [sum(source_move([move, other]) == move for other in legal if other != move)
             for move in legal]
    assert ranks == [0, 2, 1]

    report = REPORT.read_text(encoding="utf-8")
    normalizer = sum(exp(rank) for rank in ranks)
    for move, count, rank in zip(legal, counts, ranks):
        probability = exp(rank) / normalizer
        assert f"| {move} | {count}/8 | {rank} | {rank / 2:g} | {probability:.5f} |" in report
    assert exp(1) / sum(exp(rank / 2) for rank in ranks) == pytest.approx(0.50648, abs=0.000005)

    rng = Random(36)
    exact = {name: [] for name in (
        "VNM game points, both seats use source", "VNM game points, both seats uniform",
        "VNM visit-based utility, both seats use source", "VNM visit-based utility, both seats uniform",
    )}
    samples = []
    for move, count in zip(legal, counts):
        paths = list(continuations(board.apply_move(move), (move,), count // 8, Fraction(1), 1))
        assert sum(row[3] for row in paths) == sum(row[4] for row in paths) == 1
        for path, points, utility, uniform, source in paths:
            path_text = ", ".join(map(str, path))
            outcome = {1: "Win", 0: "Draw", -1: "Loss"}[points]
            line, = (line for line in report.splitlines() if line.startswith(f"| {path_text} | {outcome} |"))
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            assert int(cells[2].replace("−", "-")) == points
            assert int(cells[3].split("=")[-1].strip().replace("−", "-")) == utility
            assert Fraction(cells[4].replace("½", "1/2")) == uniform
            assert int(cells[5]) == source
        for label, quantity, mass in (
            ("VNM game points, both seats use source", 1, 4),
            ("VNM game points, both seats uniform", 1, 3),
            ("VNM visit-based utility, both seats use source", 2, 4),
            ("VNM visit-based utility, both seats uniform", 2, 3),
        ):
            exact[label].append(sum(row[quantity] * row[mass] for row in paths))
        source_payoff = estimate_q_pi(board, move, lambda s: source_move(s.legal_moves()), root_player=1)
        assert source_payoff == exact["VNM game points, both seats use source"][-1]
        # On this board every path conditional on a candidate is equiprobable.
        assert all(row[3] == Fraction(1, len(paths)) for row in paths)
        samples.append(mean(rng.choice(paths)[1] for _ in range(8)))

    for label, values in exact.items():
        cells = [f"{float(value):g}".replace("-", "−") for value in values]
        best, runner_up, _ = sorted(values, reverse=True)
        selected = legal[values.index(best)]
        cells.extend([f"{float(best - runner_up):g}", str(selected), f"{int(selected == 5)}/1"])
        assert f"| {label} | " + " | ".join(cells) + " |" in report
    assert samples == [1, 0.875, -0.5]
