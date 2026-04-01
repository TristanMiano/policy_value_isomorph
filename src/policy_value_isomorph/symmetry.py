from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

from .tictactoe import Move, TicTacToeState


@dataclass(frozen=True)
class BoardSymmetry:
    """Index permutation for a tic-tac-toe board symmetry.

    `index_map[i] = j` means value at old index `i` moves to new index `j`.
    """

    name: str
    index_map: Tuple[int, ...]



def _index(r: int, c: int) -> int:
    return 3 * r + c



def _build_index_map(transform: Callable[[int, int], Tuple[int, int]]) -> Tuple[int, ...]:
    mapped: List[int] = []
    for i in range(9):
        r, c = divmod(i, 3)
        nr, nc = transform(r, c)
        mapped.append(_index(nr, nc))
    return tuple(mapped)



def _rotation(k: int):
    def transform(r: int, c: int) -> Tuple[int, int]:
        if k == 0:
            return r, c
        if k == 1:
            return c, 2 - r
        if k == 2:
            return 2 - r, 2 - c
        if k == 3:
            return 2 - c, r
        raise ValueError(f"invalid rotation {k}")

    return transform



def _mirror_then_rotate(k: int):
    rot = _rotation(k)

    def transform(r: int, c: int) -> Tuple[int, int]:
        mr, mc = r, 2 - c
        return rot(mr, mc)

    return transform


SYMMETRIES: Tuple[BoardSymmetry, ...] = (
    BoardSymmetry("identity", _build_index_map(_rotation(0))),
    BoardSymmetry("rot90", _build_index_map(_rotation(1))),
    BoardSymmetry("rot180", _build_index_map(_rotation(2))),
    BoardSymmetry("rot270", _build_index_map(_rotation(3))),
    BoardSymmetry("mirror", _build_index_map(_mirror_then_rotate(0))),
    BoardSymmetry("mirror_rot90", _build_index_map(_mirror_then_rotate(1))),
    BoardSymmetry("mirror_rot180", _build_index_map(_mirror_then_rotate(2))),
    BoardSymmetry("mirror_rot270", _build_index_map(_mirror_then_rotate(3))),
)



def apply_symmetry_to_board(board: Sequence[int], symmetry: BoardSymmetry) -> Tuple[int, ...]:
    if len(board) != 9:
        raise ValueError("board must have length 9")

    transformed = [0] * 9
    for old_i, new_i in enumerate(symmetry.index_map):
        transformed[new_i] = board[old_i]
    return tuple(transformed)



def apply_symmetry_to_state(state: TicTacToeState, symmetry: BoardSymmetry) -> TicTacToeState:
    return TicTacToeState(board=apply_symmetry_to_board(state.board, symmetry), to_move=state.to_move)



def apply_symmetry_to_move(move: Move, symmetry: BoardSymmetry) -> Move:
    if move < 0 or move >= 9:
        raise ValueError(f"move out of range: {move}")
    return symmetry.index_map[move]



def symmetric_states(state: TicTacToeState) -> List[TicTacToeState]:
    return [apply_symmetry_to_state(state, sym) for sym in SYMMETRIES]



def canonicalize_state(state: TicTacToeState) -> TicTacToeState:
    """Reduce state to lexicographically minimal board across all 8 symmetries."""
    return min(symmetric_states(state), key=lambda s: s.board)



def canonicalize_state_action(state: TicTacToeState, action: Move) -> Tuple[TicTacToeState, Move]:
    """Canonicalize a state-action pair with consistent move remapping.

    Ties are broken by first matching symmetry order in `SYMMETRIES`.
    """
    best_state: TicTacToeState | None = None
    best_action: Move | None = None

    for sym in SYMMETRIES:
        transformed_state = apply_symmetry_to_state(state, sym)
        transformed_action = apply_symmetry_to_move(action, sym)

        if best_state is None or transformed_state.board < best_state.board:
            best_state = transformed_state
            best_action = transformed_action

    if best_state is None or best_action is None:
        raise ValueError("no symmetries available")
    return best_state, best_action



def unique_canonical_states(states: Iterable[TicTacToeState]) -> List[TicTacToeState]:
    """Return deduplicated canonical states preserving first-seen order."""
    seen: set[Tuple[int, ...]] = set()
    reduced: List[TicTacToeState] = []

    for state in states:
        canonical = canonicalize_state(state)
        if canonical.board not in seen:
            seen.add(canonical.board)
            reduced.append(canonical)
    return reduced
