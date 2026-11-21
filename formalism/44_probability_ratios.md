# 44 · What repeated choices reveal about score differences

**In plain language.** How often a player picks one move over another tells you their numerical score gap only after you specify how scores produce probabilities. The same observed odds can come from doubling the gap and doubling the sampling temperature. Finite counts add uncertainty, especially for rare moves. [Common notation](00_notation.md)

**Theorem.** In a fixed context suppose each eligible move a has positive base weight μ(a) and the probability p(a) is proportional to μ(a) exp(S(a)/T) for one positive temperature T. For any two positive-probability moves i,j,
`(S(i)−S(j))/T = log(p(i)/p(j)) − log(μ(i)/μ(j))`.
Divide the two model probabilities to cancel the normalizing constant and take logarithms. The sign of the adjusted log odds identifies their ordering. The magnitude gives a gap in temperature units. Multiplying by a known T gives the gap in score units. When base weights differ, raw choice frequency can even order moves differently from the scores.

Suppose several menus at the same history share these scores, temperature and known base-weight ratios. Join two moves by an edge whenever their odds are observed. In each connected component, summing oriented gaps along a path from an anchor determines all scores relative to it in temperature units. Such scores exist precisely when every directed traversal of a cycle sums to zero, counting a reversed edge with reversed sign. Necessity follows by cancellation. For sufficiency the cycle condition makes all path sums between fixed endpoints equal, so the assigned differences realize every observation. Separate components retain independent offsets. Fresh menus connecting components test and narrow that ambiguity. This result applies within positive support; a zero known probability is handled by a mask or extended score, and a zero base weight makes the score observationally inaccessible. [Discrete sampling theorem](38_random_choices.md)

**Repeated draws.** Given n independent pair choices at unchanged odds, let k select i. For interior counts replace p(i)/p(j) by k/(n−k). This estimates the score gap and need not be unbiased. In a larger menu the same calculation conditions on the n draws that selected either i or j. If n=0 there is no pair information. For extremes k=0 or n record the counts and a one-sided or unbounded log-gap interval. Assigning a pseudocount is a modeling assumption.

Here is a conservative finite-sample interval with a short proof. Let b denote the true probability of i conditional on the pair. The sample fraction has variance b(1−b)/n ≤ 1/(4n). For any α between 0 and 1, the chance that its squared error exceeds 1/(4nα) is at most α, since the expected squared error bounds that chance times the stated squared error. Intersect the sample fraction plus or minus √(1/(4nα)) with [0,1]. Map its endpoints through the increasing function `T log((b/(1−b)) μ(j)/μ(i))`, using infinite limits at 0 and 1. This gives at least 1−α coverage over independent repetitions of this pair experiment. Simultaneous inference for many pairs requires joint uncertainty accounting.

**Example.** A simulated player picks A with probability 3/4 and B otherwise, independently. With base weights (2,1) and temperature 2 its A-minus-B score gap is 0.81093. From 400 draws with seed 44, 311 chose A and 89 B. The estimated gap is 1.11602, an error of +0.30509 score units. The conservative 95% interval above is [−0.00872, 2.78100]. This is one simulated collection at known odds, not an estimate of how a particular game player behaves.

```pycon
>>> from random import Random
>>> from math import log, sqrt, isclose
>>> gap = lambda b: 2 * log(b / (1-b) / 2)
>>> rng = Random(44)
>>> k = sum(rng.random() < 0.75 for _ in range(400))
>>> b, radius = k/400, sqrt(1/(4*400*0.05))
>>> k, *(round(gap(x), 5) for x in (0.75, b, b-radius, b+radius))
(311, 0.81093, 1.11602, -0.00872, 2.781)
>>> assert isclose(log(2) + log(3), log(6))
>>> assert log(0.6/0.4) - log(3) < 0

```

**Self-review.** A stable sampling model explains how gaps can be recovered. Unknown base weights, temperature or menu effects change what the same odds establish. Probe cross-menu consistency and calibrate sampling variation before interpreting numerical gaps as preference intensity.
