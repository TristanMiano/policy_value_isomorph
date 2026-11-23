"""Verify learned-only complete-game scores without giving the planner game results."""

from types import SimpleNamespace
import json
from pathlib import Path
import sys
import numpy as np
import pytest

from policy_value_isomorph.visit_selection import (
    FiniteVisitPlanner,
    VisitSelection,
    UTILITIES,
    path_values,
    standard_visit_vnm,
)


def test_exact_scores_match_complete_paths_without_payoff_access():
    from policy_value_isomorph.phase_iv_game import game

    b = game()
    graph = SimpleNamespace(
        **{k: getattr(b, k) for k in ("legal", "child", "side", "ends", "n", "uniform", "greedy", "layers")}
    )
    scores = np.random.default_rng(392).uniform(size=b.legal.shape)
    p = FiniteVisitPlanner(graph, scores)
    lock = Path(__file__).resolve().parents[1] / "experiments/visit_utility/cells/refinement/selection.json"
    assert standard_visit_vnm("Q").name == json.loads(lock.read_text())["winner"]
    assert standard_visit_vnm().network == "V"
    np.testing.assert_array_equal(p.law(), p.law(standard_visit_vnm("V")))
    for depth in (5, 6):
        root = b.layers[depth][23]
        seat = b.side[root]
        for flat_score, flat_moves in ((False, False), (True, False), (True, True)):
            greedy = p.first if flat_moves else p.greedy

            def enumerate_paths(state, history, reply):
                if b.ends[state]:
                    return path_values(history, 1)
                law = greedy if b.side[state] == seat else reply
                return sum(
                    law[state, a]
                    * enumerate_paths(
                        b.child[state, a], [*history, 0.5 if flat_score else scores[state, a]], reply
                    )
                    for a in np.flatnonzero(law[state])
                )

            menu = p.menu("mixed_reply", flat_score, flat_moves)
            for a in np.flatnonzero(b.legal[root]):
                initial = [0.5 if flat_score else scores[root, a]]
                expect = (
                    enumerate_paths(b.child[root, a], initial, greedy)
                    + enumerate_paths(b.child[root, a], initial, b.uniform)
                ) / 2
                np.testing.assert_allclose(menu[root, a], expect, atol=1e-13)


@pytest.mark.skipif(
    sys.platform == "win32", reason="larger-game simulation uses the Linux/macOS C++ toolchain"
)
def test_native_fitted_and_constant_utilities_from_explicit_paths():
    from policy_value_isomorph.phase_v import native, prefixes, SIZE, spec

    rng = np.random.default_rng(92)
    for rev in (False, True):
        e = native().Engine(rev)
        e.set(1, False, 2, rng.normal(0.03, 0.08, SIZE))
        planner = native().VisitPlanner(e, 1, 56, 8, False)
        for state in prefixes(e, 18, rng):
            legal = [a for a in range(17) if e.legal(state) & (1 << a)]
            if len(legal) < 2:
                continue
            paths = []
            for candidate in legal:
                s, move, path = state, candidate, []
                while e.legal(s):
                    values = e.profiles([s], spec(1), False)[0, :-2]
                    a = move if not path else int(values.argmax())
                    path.append(values[a])
                    s = e.next(s, a)
                paths.append(path)
            for flat in (False, True):
                expected = np.stack([path_values([0.5] * len(path) if flat else path, 1) for path in paths])
                for j, u in enumerate(UTILITIES):
                    r = VisitSelection("V", "copies", u, flat)
                    a = legal[np.rint(expected[:, j] * 1e12).argmax()]
                    assert planner.choose(state, *r.native_args()) == a


@pytest.mark.skipif(
    sys.platform == "win32", reason="native full-game simulator uses the Linux/macOS toolchain"
)
def test_sampled_standard_matches_independent_64_path_average():
    from policy_value_isomorph.phase_v import SIZE, native, spec

    mask64 = (1 << 64) - 1

    def scramble(x):
        x &= mask64
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask64
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask64
        return x ^ (x >> 31)

    for rev in (False, True):
        e = native().Engine(rev)
        rule = standard_visit_vnm()
        salt = 72
        e.set(1, False, 2, np.random.default_rng(27).normal(0.035, 0.07, SIZE))
        planner = native().VisitPlanner(e, 1, salt, rule.paths, False)
        first_board = e.start()
        second_board = e.next(first_board, (e.legal(first_board) & -e.legal(first_board)).bit_length() - 1)
        for root in (first_board, second_board):
            means = {}
            for candidate in range(17):
                if not (e.legal(root) & (1 << candidate)):
                    continue
                utilities = []
                for k in range(64):
                    s, a, age = root, candidate, 0
                    while True:
                        child = e.next(s, a)
                        legal = e.legal(child)
                        if not legal:
                            final = e.profiles([s], spec(1), False)[0, a]
                            utilities.append((-1 if age % 2 else 1) * 0.9**age * np.sqrt(final))
                            break
                        key = scramble(child[0] + scramble(child[1] + child[2]))
                        draw = (scramble(key ^ scramble(salt + k)) >> 11) / 2**53
                        choices = [j for j in range(17) if legal & (1 << j)]
                        a = choices[int(len(choices) * draw)]
                        s, age = child, age + 1
                means[candidate] = np.rint(np.mean(utilities) * 1e12)
            assert planner.choose(root, *rule.native_args()) == max(means, key=means.get)
