from fractions import Fraction as F

import pytest

from policy_value_isomorph.score_encoding import (distance_energies, encode_probabilities,
    indicator_scores, sample_probabilities, select)


def test_round_trip_and_restricted_grid():
    for a in (2,0,7):
        assert select(indicator_scores([2,0,7], a, gap=3)) == a
    p = (F(1,3), F(2,3), 0)
    enc = encode_probabilities(p, [2,1,4], 0.25)
    assert enc.exact_probabilities() == p
    assert sample_probabilities(enc.scores, enc.base, enc.temperature) == pytest.approx(tuple(map(float,p)))
    assert sample_probabilities([10000,10000], [1,3], 0.01) == pytest.approx((0.25,0.75))
    for grid, expected in [([0,0.375,1],0.375), ([0,0.5,1],0.5)]:
        assert select(distance_energies(grid, 0.375, lambda x,y: abs(x-y)), minimize=True) == expected
    with pytest.raises(ValueError):
        encode_probabilities([1,0], [0,1])
