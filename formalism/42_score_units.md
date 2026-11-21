# 42 · Comparing scores measured in different units

**In plain language.** Rankings survive any strictly increasing relabeling. Sampling probabilities also depend on the spacing between scores. To preserve them when doubling every gap, double the sampling temperature. A score intended to be averaged over uncertain outcomes has a further constraint: a nonlinear change of units can reverse which lottery looks better. [Common notation](00_notation.md)

**Normalization theorem.** In a finite row with finite scores, subtract a common anchor, such as the maximum or mean. This preserves differences, ties, first-maximum choice and exponential sampling at any fixed temperature and base weights. For S′ = b + αS with α > 0, all pairwise differences are multiplied by α. Exponential sampling gives the same probabilities when T′ = αT. Conversely, on common positive support with fixed positive base weights, identical probabilities imply `S′/T′ − S/T` is constant across moves. Divide two probabilities and take logarithms to obtain this condition for every pair. In particular, at fixed T only offsets are invisible on the support. These assertions follow row by row. Scaling rows separately requires a corresponding temperature per row to preserve all distributions in general.

**Rank and quantile coordinates.** Sort the distinct scores and number their levels from zero, giving equal scores the same level. Alternatively use the fraction of entries at or below the score as its empirical quantile. Both transformations preserve strict order among the distinct observed levels and keep ties. Quantile binning that merges distinct levels can create new ties. Rank or quantile gaps need not be proportional to the original gaps, so the sampling probabilities can change even after adjusting a single temperature. With scores (0,1,3), adjacent gaps have ratio 1:2. Equally spaced ranks have ratio 1:1. No shared temperature compensates for both ratios.

**Lottery units.** A lottery is a probability distribution over consequences and U is their common utility. Replacing U by αU+b with α>0 preserves comparisons of expected utility because the same transformation takes place after taking the mean. Conversely on an interval of attainable utilities, if a relabeling preserves order and all indifferences between a sure value and an arbitrary two-point lottery with that same mean, it must be positive affine on every segment. A point dividing a segment in fraction λ is indifferent to a lottery on its endpoints with that mixing fraction. Its new value must divide the transformed endpoints in the same fraction. Overlapping segments give one common affine rule. Restrictions on available lotteries can leave more changes unobservable. A sure 2 beats an equal mixture of 0 and 3 under the identity utility. Squaring these nonnegative values reverses that preference.

**Comparison protocol.** Specify which responses a score should preserve. For fixed-temperature sampling, compare differences in temperature units with a common anchor and record support. For pure order compare tied ranks and pairwise reversals. To compare expected utility, use the same two consequence anchors across decisions or disclose how the units were fixed. Store a constant finite row as zeros after centering, with its tie flag. Exclude forbidden actions from finite-range arithmetic and publish their mass separately. Compute squared score error only on quantities in compatible units. A fixed threshold retains its numerical meaning only if its cutoff is transformed consistently or all scores stay on their original side.

```pycon
>>> from fractions import Fraction as F
>>> from math import exp, isclose
>>> def probabilities(row, t):
...     w = [exp((v-max(row))/t) for v in row]
...     return [x/sum(w) for x in w]
>>> s = (0,1,3)
>>> assert all(isclose(a,b) for a,b in zip(probabilities(s,2), probabilities([4+3*x for x in s],6)))
>>> assert probabilities(s,1) != probabilities([0,1,2],1)
>>> q = [sum(y <= x for y in [0,0,3])/3 for x in [0,0,3]]
>>> q[0] == q[1] < q[2], [v-max([4,4]) for v in [4,4]]
(True, [0, 0])
>>> 2 > F(0+3,2), 2**2 < F(0**2+3**2,2)
(True, True)

```

**Self-review.** Normalization can remove arbitrary units or destroy real spacing, depending on the question. Keeping both original and aligned scores permits a check against each declared consumer. How stable the aligned scores are across fitted networks requires independent fits.
