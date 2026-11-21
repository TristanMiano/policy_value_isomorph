"""Finite theorem examples and exhaustive checks, runnable without third-party packages."""

import doctest
import importlib.util
import unittest
from collections.abc import Sequence
from fractions import Fraction as F
from itertools import permutations, product
from math import exp, isclose, log
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# Load the standalone mathematical module without initializing the package's neural trainers.
spec = importlib.util.spec_from_file_location(
    "finite_formal", ROOT / "src" / "policy_value_isomorph" / "finite_formal.py"
)
assert spec is not None and spec.loader is not None
formal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formal)


def distance(left: Sequence[F | int], right: Sequence[F | int]) -> F:
    """Total variation for finite probability rows on the same menu."""
    return sum((abs(a - b) for a, b in zip(left, right, strict=True)), F()) / 2


def quarter_laws() -> list[tuple[F, ...]]:
    """All three-move laws whose masses are integer multiples of one quarter."""
    return [tuple(F(n, 4) for n in row) for row in product(range(5), repeat=3) if sum(row) == 4]


class FormalExamples(unittest.TestCase):
    def test_published_examples(self):
        for task in range(37, 48):
            path, = (ROOT / "formalism").glob(f"{task}_*.md")
            with self.subTest(proof=path.name):
                result = doctest.testfile(str(path), module_relative=False, encoding="utf-8")
                self.assertGreater(result.attempted, 0)
                self.assertEqual(result.failed, 0)

    def test_score_and_probability_encodings(self):
        for law in quarter_laws():
            for base in ((1, 1, 1), (1, 2, 4)):
                # Exact exponential weights for T log(p/base), with zero weight for exclusion.
                ratios = [p / w for p, w in zip(law, base)]
                for common_factor in (F(1, 3), 1, 5):
                    decoded = formal.normalized_weights(
                        [c * w * common_factor for c, w in zip(ratios, base)]
                    )
                    self.assertEqual(decoded, law)
        for invalid in ([], [0, 0], [-1, 2]):
            with self.assertRaises(ValueError):
                formal.normalized_weights(invalid)

    def test_shared_successor_feasibility_against_enumeration(self):
        vertices = (0, 1, 2)
        pairs = tuple(permutations(vertices, 2))
        integer_tables = tuple(product(vertices, repeat=3))
        feasible = 0
        for kinds in product((None, False, True), repeat=len(pairs)):
            edges = [(i, j, strict) for (i, j), strict in zip(pairs, kinds) if strict is not None]

            def satisfies(table):
                return all(table[j] >= table[i] + int(strict) for i, j, strict in edges)

            witness = formal.successor_levels(vertices, edges)
            # Three ordered values suffice to realize any weak ordering on three labels.
            self.assertEqual(witness is not None, any(satisfies(t) for t in integer_tables))
            if witness is not None:
                feasible += 1
                self.assertTrue(satisfies(witness))
        self.assertEqual(feasible, 178)
        self.assertEqual(formal.successor_levels([], []), {})
        self.assertEqual(formal.successor_levels(["isolated"], []), {"isolated": 0})

    def test_choice_constraints_respect_ties_duplicates_and_minimizers(self):
        for row in product(range(3), repeat=3):
            for maximize in (False, True):
                choose = max if maximize else min
                for selected in range(len(row)):
                    edges = formal.choice_constraints(row, selected, maximize=maximize)
                    for table in product(range(3), repeat=3):
                        winner = choose(range(len(row)), key=lambda i: table[row[i]])
                        self.assertEqual(
                            all(table[b] >= table[a] + int(strict) for a, b, strict in edges),
                            winner == selected,
                        )
        with self.assertRaises(ValueError):
            formal.choice_constraints([], 0)

    def test_metric_grid_and_ranking_queries(self):
        grid = [F(i, 4) for i in range(-4, 5)]
        for target in [F(i, 8) for i in range(-8, 9)]:
            best = max(grid, key=lambda a: -abs(a - target))
            self.assertLessEqual(abs(best - target), F(1, 8))
        for a, b, c in permutations("ABC"):
            # Reversing the strict preference on the A/B/C cycle closes a strict loop.
            chain = [(c, b, True), (b, a, True)]
            self.assertIsNotNone(formal.successor_levels("ABC", chain + [(c, a, True)]))
            self.assertIsNone(formal.successor_levels("ABC", chain + [(a, c, True)]))

    def test_consequence_lotteries_and_optimal_tie_mixing(self):
        laws = quarter_laws()
        for source in laws:
            for rewards in product(range(2), repeat=3):
                best = max(rewards)
                outside = sum(p for p, r in zip(source, rewards) if r != best)
                witness = formal.nearest_maximizer_law(source, rewards)
                self.assertEqual(formal.expected_utility(witness, rewards), best)
                self.assertEqual(distance(source, witness), outside)
                attainable = [distance(source, q) for q in laws if formal.expected_utility(q, rewards) == best]
                self.assertEqual(min(attainable), outside)
        with self.assertRaises(ValueError):
            formal.expected_utility((F(1, 2), 0), (1, 0))
        with self.assertRaises(ValueError):
            formal.expected_utility((1, 0), (1,))
        with self.assertRaises(ValueError):
            formal.nearest_maximizer_law((1, 0), (1,))

    def test_decision_stability_under_signed_errors(self):
        for scores in product(range(3), repeat=3):
            best, second, _ = sorted(scores, reverse=True)
            for errors in product((-1, 0, 1), repeat=3):
                fitted = [s + d for s, d in zip(scores, errors)]
                choice = fitted.index(max(fitted))
                spread = max(errors) - min(errors)
                self.assertLessEqual(best - scores[choice], spread)
                if best - second > spread:
                    self.assertEqual(choice, scores.index(best))

    def test_base_weighted_tied_maxima_and_menu_odds(self):
        base, scores = (2, 1, 7), (3, 3, 2)
        for temperature in (10, 1, 0.1, 0.01):
            masses = [b * exp((s - max(scores)) / temperature) for b, s in zip(base, scores)]
            law = [m / sum(masses) for m in masses]
            self.assertAlmostEqual(law[0] / law[1], 2)
            adjusted_gap = temperature * (log(law[1] / law[2]) - log(base[1] / base[2]))
            self.assertAlmostEqual(adjusted_gap, 1)
        self.assertTrue(isclose(law[0], 2 / 3))
        self.assertTrue(isclose(law[1], 1 / 3))
        hot = [b * exp(s / 1e12) for b, s in zip(base, scores)]
        for w, b in zip(hot, base):
            self.assertAlmostEqual(w / sum(hot), b / sum(base))
        self.assertAlmostEqual(1 / (1 + exp(-1 / 0.01)), 1)
        # A common factor in one pair's exponential weights is an additive score offset.
        weights = (1, 2, 3, 4)
        shifted = (1, 2, 15, 20)
        for i, j in [(0, 1), (2, 3)]:
            self.assertEqual(F(weights[i], weights[j]), F(shifted[i], shifted[j]))
        self.assertNotEqual(F(weights[1], weights[2]), F(shifted[1], shifted[2]))

    def test_restricted_consequence_features_and_random_utility(self):
        # Both choices end in a draw. Only the expanded consequence can distinguish duration.
        coarse = [(1,), (1,)]
        detailed = [(1, 0), (0, 1)]
        coarse_scores = [formal.expected_utility(row, (4,)) for row in coarse]
        detailed_scores = [formal.expected_utility(row, (0, 1)) for row in detailed]
        self.assertEqual(coarse_scores.index(max(coarse_scores)), 0)
        self.assertEqual(detailed_scores.index(max(detailed_scores)), 1)
        types = [((1, 0), F(1, 4)), ((0, 1), F(3, 4))]
        mixture = tuple(sum(w for u, w in types if u.index(max(u)) == i) for i in (0, 1))
        mean_scores = [sum(w * u[i] for u, w in types) for i in (0, 1)]
        self.assertEqual(mixture, (F(1, 4), F(3, 4)))
        self.assertEqual(mean_scores.index(max(mean_scores)), 1)

    def test_probability_stability_in_exact_arithmetic(self):
        for weights in product((1, 2, 4), repeat=3):
            p = formal.normalized_weights(weights)
            for factors in product((1, 2, 4), repeat=3):
                q = formal.normalized_weights([x * f for x, f in zip(weights, factors)])
                tv = distance(p, q)
                # Equivalent rational form of TV <= (sqrt(k)-1)/(sqrt(k)+1).
                k = F(max(factors), min(factors))
                self.assertLessEqual((1 + tv)**2, k * (1 - tv)**2)
        p = formal.normalized_weights([1, 2])
        q = formal.normalized_weights([4, 2])
        self.assertEqual(distance(p, q), F(1, 3))


if __name__ == "__main__":
    unittest.main()
