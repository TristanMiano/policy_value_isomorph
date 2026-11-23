"""Expected utilities from fitted move frequencies with outcome-blind continuing players."""

from dataclasses import dataclass
from itertools import product
import numpy as np

CONTINUATIONS = ("copies", "uniform_reply", "mixed_reply", "uniform_both")
UTILITIES = (
    "signed_last",
    "unsigned_last",
    "signed_sum",
    "signed_90",
    "signed_50",
    "unsigned_sum",
    "unsigned_90",
    "unsigned_50",
    "signed_last_90",
    "own_90",
    "last_plus_sum",
    "signed_last_75",
    "signed_last_97",
    "sqrt_last_90",
    "square_last_90",
)


@dataclass(frozen=True)
class VisitSelection:
    network: str
    continuation: str
    utility: str
    flat_score: bool = False
    flat_moves: bool = False
    paths: int = 8

    def __post_init__(self):
        if (
            self.network not in ("V", "Q")
            or self.continuation not in CONTINUATIONS
            or self.utility not in UTILITIES
            or self.paths < 1
        ):
            raise ValueError("unknown visit-based selection rule")

    @property
    def name(self) -> str:
        suffix = "/constant_all" if self.flat_moves else ("/constant_score" if self.flat_score else "")
        budget = f"/paths{self.paths}" if self.paths != 8 else ""
        return f"{self.network}/{self.continuation}/{self.utility}{budget}{suffix}"

    @classmethod
    def parse(cls, name: str):
        net, continuation, utility, *control = name.split("/")
        budget = next((int(k[5:]) for k in control if k.startswith("paths")), 8)
        flat = "constant_all" in control
        return cls(net, continuation, utility, flat or "constant_score" in control, flat, budget)

    def native_args(self) -> tuple[int, int, bool]:
        return CONTINUATIONS.index(self.continuation), UTILITIES.index(self.utility), self.flat_score


def candidates() -> list[VisitSelection]:
    return [VisitSelection(*r) for r in product(("V", "Q"), CONTINUATIONS, UTILITIES[:11])]


def standard_visit_vnm(network: str = "V") -> VisitSelection:
    """Use the frozen visit-only formula, scoring the resulting board by default."""
    return VisitSelection(network, "uniform_both", "sqrt_last_90", paths=64)


def path_values(frequencies, first_sign: int) -> np.ndarray:
    """Score a played path from the evaluated seat. first_sign is +1 if that seat moves first."""
    raw = np.asarray(frequencies)
    turn = np.arange(len(raw))
    signed = raw * first_sign * np.where(turn % 2, -1, 1)
    slow, fast = 0.9**turn, 0.5**turn
    return np.array(
        [
            signed[-1],
            raw[-1],
            signed.sum(),
            signed @ slow,
            signed @ fast,
            raw.sum(),
            raw @ slow,
            raw @ fast,
            signed[-1] * slow[-1],
            (raw + signed) @ slow / 2,
            signed[-1] + 0.1 * (signed @ slow),
            signed[-1] * 0.75 ** turn[-1],
            signed[-1] * 0.97 ** turn[-1],
            np.sign(signed[-1]) * np.sqrt(raw[-1]) * slow[-1],
            signed[-1] * raw[-1] * slow[-1],
        ]
    )


class FiniteVisitPlanner:
    """Exact expectations on tic-tac-toe. Only legal moves, turns and stopping enter planning."""

    def __init__(self, board, scores):
        self.b = board
        if np.shape(scores) != board.legal.shape or not np.isfinite(scores[board.legal]).all():
            raise ValueError("supply one finite fitted frequency per legal move")
        self.frequencies = np.where(board.legal, np.clip(scores, 0, 1), 0.0)
        self.greedy = board.greedy(self.frequencies)
        self.first = board.greedy(np.zeros_like(self.frequencies))
        self.menus = {}

    def menu(self, mode: str, flat_score: bool = False, flat_moves: bool = False) -> np.ndarray:
        key = (mode, flat_score, flat_moves)
        if key in self.menus:
            return self.menus[key]
        if mode == "mixed_reply":
            result = (
                self.menu("copies", flat_score, flat_moves)
                + self.menu("uniform_reply", flat_score, flat_moves)
            ) / 2
        else:
            b = self.b
            greedy = self.first if flat_moves else self.greedy
            own = b.uniform if mode == "uniform_both" else greedy
            other = greedy if mode == "copies" else b.uniform
            raw = np.full_like(self.frequencies, 0.5) if flat_score else self.frequencies
            result = np.zeros((*b.legal.shape, len(UTILITIES)))
            for seat in (1, -1):
                tail = np.zeros((b.n, len(UTILITIES)))
                for ids in reversed(b.layers):
                    children = b.child[ids]
                    ending = b.ends[children]
                    value = tail[children].copy()
                    own_turn = b.side[ids[0]] == seat
                    f = raw[ids]
                    signed = f if own_turn else -f
                    value[:, :, 0] = np.where(ending, signed, value[:, :, 0])
                    value[:, :, 1] = np.where(ending, f, value[:, :, 1])
                    for j, discount in enumerate((1.0, 0.9, 0.5)):
                        value[:, :, 2 + j] = signed + discount * value[:, :, 2 + j]
                        value[:, :, 5 + j] = f + discount * value[:, :, 5 + j]
                    value[:, :, 8] = np.where(ending, signed, 0.9 * value[:, :, 8])
                    value[:, :, 9] = (value[:, :, 6] + value[:, :, 3]) / 2
                    value[:, :, 10] = value[:, :, 0] + 0.1 * value[:, :, 3]
                    for j, discount in ((11, 0.75), (12, 0.97)):
                        value[:, :, j] = np.where(ending, signed, discount * value[:, :, j])
                    root = np.sqrt(f) if own_turn else -np.sqrt(f)
                    value[:, :, 13] = np.where(ending, root, 0.9 * value[:, :, 13])
                    value[:, :, 14] = np.where(ending, signed * f, 0.9 * value[:, :, 14])
                    law = own if own_turn else other
                    tail[ids] = np.einsum("ij,ijk->ik", law[ids], value)
                    if own_turn:
                        result[ids] = value
        self.menus[key] = result
        return result

    def law(self, rule: VisitSelection | None = None) -> np.ndarray:
        rule = standard_visit_vnm() if rule is None else rule
        menu = self.menu(rule.continuation, rule.flat_score, rule.flat_moves)
        return self.b.greedy(np.rint(menu[:, :, UTILITIES.index(rule.utility)] * 1e12))


def visit_game_player(engine, seed: int, rule: VisitSelection | None = None):
    """Bind the outcome-blind larger-game player to an installed V or Q fit."""
    from .phase_v import native

    rule = standard_visit_vnm() if rule is None else rule
    planner = native().VisitPlanner(
        engine, 1 if rule.network == "V" else 2, seed, rule.paths, rule.flat_moves
    )
    return lambda state: planner.choose(state, *rule.native_args())
