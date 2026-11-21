# 46 · How score errors change decisions

**In plain language.** Close scores leave room for a small error to change the leading move. A large error shared by every move leaves their order intact. Knowing the average training error alone cannot tell you how many choices will change. For sampled choices we can also bound *how much probability* can move. [Common notation](00_notation.md)

**Deterministic stability theorem.** On one finite legal menu let S be the reference scores oriented for the mover, Ŝ the fitted scores, and δ(a)=Ŝ(a)−S(a). Let c be the unique best reference move, g its gap above the runner-up, and e the largest absolute score error. If `g > max δ − min δ`, the fitted winner is c. In particular g>2e suffices. For every rival b, its fitted deficit behind c is `S(c)−S(b)+δ(c)−δ(b)`, giving the claim directly. At equality a new tie can change the choice under the fixed priority rule. With a true top tie, arbitrarily small opposing errors can switch between its members, even with no reference payoff lost. A gap greater than the error range to the best *outside* a tied set keeps the new winner inside that set. A forced move is always preserved on the same menu.

For any fitted maximizer b, the reference score given up satisfies `0 ≤ S(c)−S(b) ≤ δ(b)−δ(c) ≤ 2e`, by fitted maximality. Across weighted boards, disagreement with the exact decoder is bounded by the total normalized weight of multi-move boards with gap ≤ error range, with top ties included at gap zero. Retain the actual counts, gaps, errors and changes on those boards. The bound is about the reference decoder. The absolute change in agreement with a deterministic source is at most the fraction of changed decisions, since only those boards can gain or lose a match. For distributions, the triangle inequality gives the analogous bound on the change in distance to the source.

**Same squared error, different behavior.** Take two moves in their tie priority A, B, with true scores (0,1). Fitted scores (1,2) and (1,0) each have mean squared error 1 over two actions. The first agrees on 1/1 decisions and gives up zero score. The second agrees on 0/1 and gives up one score unit. At (0.5,0.5) the error is 0.5 per action and the exact gap is twice that largest error. The fixed tie rule chooses A. These are controlled cases for the planned Task 67 comparison of matched errors and action gaps. They make no assumption about errors following neural training.

**Stochastic stability theorem.** For exponential sampling from S and Ŝ with the same positive temperature T and base weights on fixed common support, let p and q be the two distributions. Write r = (max δ−min δ)/T over that support. Then their total variation distance (half their absolute probability difference summed over moves) satisfies
`TV(p,q) ≤ tanh(r/4) ≤ tanh(e/(2T))`,
where `tanh(x)=(exp(2x)−1)/(exp(2x)+1)`. The bound is sharp. Excluded moves that remain excluded can be removed before computing r. Changing the support or base weights requires a separate comparison.

**Proof.** For any event A (a set of moves) with positive mass on both sides, the ratio of its new odds to its old odds is the p-weighted mean of exp(δ/T) inside A divided by its p-weighted mean outside A. It lies between exp(−r) and exp(r). For initial probability x the largest possible increase at this odds ratio is `kx/(1−x+kx)−x`, where k=exp(r). Differentiating gives its maximum at x=1/(1+√k) for k>1, a difference (√k−1)/(√k+1). At k=1 it is zero. Use the complement for decreases. For a finite menu TV equals the maximum absolute change in event probability: collect exactly the moves with positive mass differences to attain it. For two moves with unit temperature, scores (−e,0) and errors (+e,−e) realize equality for every e≥0 with equal base weights. This example also shows how the bound changes with temperature by expressing e in temperature units.

```pycon
>>> from fractions import Fraction as F
>>> from math import exp, tanh, isclose
>>> s = (0,1); first = lambda row: row.index(max(row))
>>> for fitted, match, lost in [((1,2),1,0), ((1,0),0,1)]:
...     assert sum((x-y)**2 for x,y in zip(s,fitted))/2 == 1
...     assert int(first(fitted)==first(s)) == match
...     assert max(s)-s[first(fitted)] == lost
>>> assert first((F(1,2),F(1,2))) == 0 and first((0,1)) == 1
>>> assert first((1,1)) != first((1,1.000001))
>>> for e in (0,0.1,1,3):
...     low = 1/(1+exp(e))
...     assert isclose(1-2*low, tanh(e/2), abs_tol=1e-15)

```

**Self-review.** The bound allows a move to change near a tie. A common shift can preserve exactly the same move and all probabilities even there. Reporting the measured changes on the vulnerable boards preserves the distinction between a possible error and one that actually occurred.
