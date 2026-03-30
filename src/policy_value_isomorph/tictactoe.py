from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

Player = int  # +1 for X, -1 for O
Move = int    # index in [0, 8]

WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def check_winner(board: Sequence[int]) -> Optional[Player]:
    """Return +1 (X) or -1 (O) if a player has a three-in-a-row, else None."""
    for a, b, c in WIN_LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return 1
        if s == -3:
            return -1
    return None


@dataclass(frozen=True)
class TicTacToeState:
    """Canonical tic-tac-toe state.

    board: length-9 tuple with entries in {-1, 0, +1}
      +1 = X, -1 = O, 0 = empty.
    to_move: +1 (X) or -1 (O).
    """

    board: Tuple[int, ...]
    to_move: Player

    @staticmethod
    def initial() -> "TicTacToeState":
        return TicTacToeState(board=(0,) * 9, to_move=1)

    def legal_moves(self) -> List[Move]:
        if self.is_terminal():
            return []
        return [i for i, cell in enumerate(self.board) if cell == 0]

    def apply_move(self, move: Move) -> "TicTacToeState":
        if move < 0 or move >= 9:
            raise ValueError(f"move out of range: {move}")
        if self.board[move] != 0:
            raise ValueError(f"illegal move {move} on occupied square")
        if self.is_terminal():
            raise ValueError("cannot apply move to terminal state")

        mutable = list(self.board)
        mutable[move] = self.to_move
        return TicTacToeState(board=tuple(mutable), to_move=-self.to_move)

    def winner(self) -> Optional[Player]:
        return check_winner(self.board)

    def is_terminal(self) -> bool:
        return self.winner() is not None or all(x != 0 for x in self.board)

    def terminal_return(self, root_player: Player) -> int:
        """Terminal utility from root player's perspective in {-1,0,+1}."""
        if not self.is_terminal():
            raise ValueError("terminal_return called on non-terminal state")
        w = self.winner()
        if w is None:
            return 0
        return 1 if w == root_player else -1

    def as_pretty_string(self) -> str:
        symbol = {1: "X", -1: "O", 0: "."}
        rows = []
        for r in range(3):
            rows.append(" ".join(symbol[self.board[3 * r + c]] for c in range(3)))
        mover = "X" if self.to_move == 1 else "O"
        return "\n".join(rows) + f"\n(to move: {mover})"


def state_from_rows(rows: Iterable[str], to_move: Player) -> TicTacToeState:
    """Helper for tests/demo.

    rows: iterable of 3 strings each with length 3 over {'X','O','.'}.
    """
    text = "".join(rows)
    if len(text) != 9:
        raise ValueError("expected exactly 9 board chars")
    decode = {"X": 1, "O": -1, ".": 0}
    try:
        board = tuple(decode[ch] for ch in text)
    except KeyError as exc:
        raise ValueError(f"invalid board character: {exc}") from exc
    return TicTacToeState(board=board, to_move=to_move)
