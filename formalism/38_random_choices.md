# 38 · A score for a probability distribution

**In plain language.** A player that sometimes takes its second choice can still be copied exactly if its move probabilities are known. Use probabilities relative to the sampler's starting weights, take their logarithms, and scale them to the sampler's temperature. A move with exactly zero chance needs an explicit exclusion or a score of negative infinity. [Common notation](00_notation.md)

**Theorem.** Fix one context, a finite or countable menu and a positive finite base weight μ(a) on each eligible move. The temperature T is a positive number in score units. Given scores S(a), assign mass proportional to μ(a) exp(S(a)/T). The sum of these masses, Z, must be positive and finite. For any probability law p on the eligible moves, all scores that yield p are
`S(a) = T log(p(a)/μ(a)) + c` on its positive support (the moves of positive probability), and −∞ elsewhere. Here c is an arbitrary finite constant shared by that row. Energy, when used, is −S. If a base weight is zero, the sampler always excludes that move, its score there is unobservable, and a positive source probability there cannot be reproduced. Apply the statement separately to each context.

**Proof.** Substitution gives unnormalized masses p(a) exp(c/T), whose sum is exp(c/T), including on a countable menu since p sums to 1. Conversely, for a decoded law p, rearrange p(a)Z = μ(a) exp(S(a)/T) and take logarithms on positive support. This forces the stated formula with c = T log Z. With a positive base weight, zero mass forces the excluded score. Choosing Z = 1 fixes a unique representative on the eligible moves. For countably infinite menus arbitrary supplied scores can have a divergent sum, in which case they do not define this decoder.

**Finite-score repair.** On a finite menu with zero-probability eligible moves, assign an aggregate mass δ strictly between 0 and 1 to those moves, positive on each of them, and multiply all original probabilities by 1−δ. All scores are now finite. The **total variation distance**, half the sum of the absolute probability differences, is exactly δ: the positive support loses δ and the excluded set gains δ. If the base measure forbids source mass m, every allowable approximation is at least m away by considering that forbidden set.

```pycon
>>> from math import exp, log, isclose, inf
>>> from fractions import Fraction as F
>>> p, base, T = (0.75, 0.25, 0.0), (1, 2, 4), 2
>>> s = [T * log(x / w) if x else -inf for x, w in zip(p, base)]
>>> for c in (-3, 0, 2):
...     masses = [w * exp((v + c) / T) for w, v in zip(base, s)]
...     assert all(isclose(m / sum(masses), x) for m, x in zip(masses, p))
>>> target = (F(3, 4), F(1, 4), F(0))
>>> approx = (F(27, 40), F(9, 40), F(1, 10))
>>> sum(abs(a-b) for a, b in zip(target, approx)) / 2
Fraction(1, 10)

```

**Self-review.** Allowing exclusion gives an exact constructive result. Requiring every legal move to have a finite score changes the attainable distributions and yields a tunable approximation. Neither construction determines unknown probabilities from a finite sample. [Repeated-choice inference](44_probability_ratios.md)
