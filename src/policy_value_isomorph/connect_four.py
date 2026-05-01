from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Player = int  # +1 for first player, -1 for second player
Move = int    # column index in [0, 6]

_ROWS = 6
_COLS = 7


def _index(row: int, col: int) -> int:
    return row * _COLS + col


def check_winner(board: Sequence[int]) -> Optional[Player]:
    """Return +1 or -1 if a player has four in a row, else None."""
    for row in range(_ROWS):
        for col in range(_COLS):
            token = board[_index(row, col)]
            if token == 0:
                continue
            # Horizontal
            if col <= _COLS - 4 and all(board[_index(row, col + k)] == token for k in range(4)):
                return token
            # Vertical
            if row <= _ROWS - 4 and all(board[_index(row + k, col)] == token for k in range(4)):
                return token
            # Diagonal down-right
            if row <= _ROWS - 4 and col <= _COLS - 4 and all(board[_index(row + k, col + k)] == token for k in range(4)):
                return token
            # Diagonal down-left
            if row <= _ROWS - 4 and col >= 3 and all(board[_index(row + k, col - k)] == token for k in range(4)):
                return token
    return None


@dataclass(frozen=True)
class ConnectFourState:
    """Connect Four state with board entries in {-1, 0, +1} on a 6x7 grid."""

    board: Tuple[int, ...]
    to_move: Player

    @staticmethod
    def initial() -> "ConnectFourState":
        return ConnectFourState(board=(0,) * (_ROWS * _COLS), to_move=1)

    def legal_moves(self) -> List[Move]:
        if self.is_terminal():
            return []
        return [c for c in range(_COLS) if self.board[_index(0, c)] == 0]

    def apply_move(self, move: Move) -> "ConnectFourState":
        if move < 0 or move >= _COLS:
            raise ValueError(f"move out of range: {move}")
        if self.is_terminal():
            raise ValueError("cannot apply move to terminal state")
        if self.board[_index(0, move)] != 0:
            raise ValueError(f"illegal move {move} on full column")

        mutable = list(self.board)
        for row in range(_ROWS - 1, -1, -1):
            idx = _index(row, move)
            if mutable[idx] == 0:
                mutable[idx] = self.to_move
                return ConnectFourState(board=tuple(mutable), to_move=-self.to_move)
        raise RuntimeError("unreachable full-column logic")

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
        for r in range(_ROWS):
            rows.append(" ".join(symbol[self.board[_index(r, c)]] for c in range(_COLS)))
        mover = "X" if self.to_move == 1 else "O"
        rows.append("0 1 2 3 4 5 6")
        return "\n".join(rows) + f"\n(to move: {mover})"
