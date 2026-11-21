"""Recount the shared-board result directly from the frozen tic-tac-toe player's visits."""

from collections import defaultdict
from graphlib import TopologicalSorter
from pathlib import Path

import numpy as np


ARCHIVE = Path(__file__).resolve().parents[1] / "experiments/tictactoe_sampling/counts.npz"


def test_minimum_board_score_levels_with_and_without_ties() -> None:
    # Choose the only visited move on each board. Derive successors from the cells
    # themselves, independently of the game's graph and the Phase II score solver.
    rows: dict[str, dict[int, int]] = defaultdict(dict)
    with np.load(ARCHIVE) as data:
        for board, cell, visits in zip(
            data["edge_parent_board"].tolist(),
            data["edge_move"].tolist(),
            data["optimal_net/known_once"].tolist(),
            strict=True,
        ):
            assert board[cell] == "." and cell not in rows[board]
            rows[board][cell] = visits

    parents: dict[str, dict[str, int]] = defaultdict(dict)
    decisions: list[tuple[int, dict[int, str]]] = []
    observed: dict[str, set[bool]] = defaultdict(set)
    for board, counts in rows.items():
        chosen, = (cell for cell, visits in counts.items() if visits > 0)
        mark = "X" if board.count("X") == board.count("O") else "O"
        children = {cell: board[:cell] + mark + board[cell + 1:] for cell in counts}
        selected = children[chosen]
        required = parents[selected]
        for cell, child in children.items():
            observed[child].add(cell == chosen)
            if cell != chosen:
                assert child not in required
                required[child] = int(cell < chosen)
        decisions.append((chosen, children))
    assert len(rows) == 4520
    assert sum(map(len, rows.values())) == 16167
    assert (len(observed), sum(len(labels) == 2 for labels in observed.values())) == (5477, 2026)
    assert sum(map(len, parents.values())) == 11647
    assert sum(sum(rivals.values()) for rivals in parents.values()) == 3275

    # In this archive even the graph with all edges strict has no cycle.
    # Any path with k strict steps needs k+1 levels. Longest-path scores
    # realize that lower bound and are checked by decoding every original row.
    order = tuple(TopologicalSorter(parents).static_order())
    assert set(order) == set(observed)
    for strict_only, minimum, tied_rows in [(True, 7, 0), (False, 3, 2143)]:
        levels: dict[str, int] = {}
        for board in order:
            levels[board] = max(
                (levels[rival] + (1 if strict_only else step)
                 for rival, step in parents.get(board, {}).items()),
                default=0,
            )
        assert sorted(set(levels.values())) == list(range(minimum))
        ties = 0
        for chosen, children in decisions:
            best = max(levels[child] for child in children.values())
            winners = [cell for cell, child in children.items() if levels[child] == best]
            assert min(winners) == chosen
            ties += len(winners) > 1
        assert ties == tied_rows
    # A concrete lower bound with the original lowest-cell priority.
    low, middle, high = "O.XO.X...", "..XO.X..O", "..X..XO.O"
    assert parents[middle][low] == parents[high][middle] == 1
