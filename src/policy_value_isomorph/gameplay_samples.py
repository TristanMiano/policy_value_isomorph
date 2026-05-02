from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .connect_four import ConnectFourState
from .tictactoe import TicTacToeState


@dataclass(frozen=True)
class GameplaySample:
    checkpoint_step: int
    game_index: int
    moves: tuple[int, ...]
    states: tuple[TicTacToeState | ConnectFourState, ...]


def _result_label(state: TicTacToeState | ConnectFourState) -> str:
    winner = state.winner()
    if winner == 1:
        return "winner: X"
    if winner == -1:
        return "winner: O"
    return "draw"


def format_gameplay_sample(sample: GameplaySample) -> str:
    lines = [
        f"checkpoint_step={sample.checkpoint_step}",
        f"game_index={sample.game_index}",
        f"moves={list(sample.moves)}",
    ]
    for i, state in enumerate(sample.states):
        if i == 0:
            lines.append("state[0] (initial):")
        else:
            lines.append(f"state[{i}] (after move {i}: action={sample.moves[i - 1]}):")
        lines.append(state.as_pretty_string())
    if sample.states and sample.states[-1].is_terminal():
        lines.append(_result_label(sample.states[-1]))
    return "\n".join(lines)


def append_gameplay_samples_text(path: str | Path, samples: Sequence[GameplaySample]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for sample in samples:
            f.write(format_gameplay_sample(sample))
            f.write("\n\n---\n\n")
