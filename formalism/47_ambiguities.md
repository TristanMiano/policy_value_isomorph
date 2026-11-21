# 47 · Six ways different explanations produce the same observations

Two explanations are **indistinguishable under an observation rule** when they produce the same answers, or the same distribution of answers, for every observation the rule allows. Each pair below exhibits this equality and a further question on which the explanations differ. Any inference based only on the shared observations must give the same answer for both worlds. It cannot recover two different hidden answers with certainty. What *is* recoverable changes when the permitted observations change. [Common notation](00_notation.md) · [Observable classes](41_response_equivalence.md)

| Observations allowed | Two constructions and surviving result | Missing distinction and possible repair |
| --- | --- | --- |
| Highest-scoring move, including all restricted menus | On moves A,B,C in that tie priority, scores (3,2,0) and (9,4,0) always choose the same first-ranked available move. The complete ranking is recoverable. | Top-to-second gaps are 1 and 5. Even all subset choices fail to distinguish these gap sizes. A stated sampling model at a known temperature, with observed probabilities, could distinguish them. |
| Full sampled distribution, unknown temperature | On A,B use equal starting weights and scores (0,1) at temperature 1, or (0,2) at temperature 2. Both assign B probability exp(1)/(1+exp(1)). The probability law and the gap in temperature units are determined. | Score gaps 1 and 2 differ. Fix the temperature in common units, or supply two score anchors that fix the difference scale. |
| Only the winning move on the full menu | One player takes the first available in A,B,C. Another takes the first available in A,C,B. Both choose A every time all are offered. | Removing A yields B for one and C for the other. A new restricted-menu query distinguishes the runners-up. Their full-menu decoder scores can still be exact. |
| Behavior induced by accumulated rewards | At the opening choose left or right, then take the only available move to finish. Original two-step rewards are (0,1) for left and (0,0) for right. Both the original and transformed rewards below choose left and retain the same one-point advantage. | Per-step rewards and total points differ. Observing the intermediate rewards or fixing the allowed reward features and units adds information. One anchored total can still leave intermediate redistribution undetermined. |
| One observation combines two distinguishable histories | A source takes A after history x and B after history y. Mapping both histories to the same board leaves one board row. With population weights 3/4 and 1/4, the observed conditional distribution on that board is (3/4,1/4), also obtainable from a player randomly choosing A with probability 3/4 independently of the hidden history on each independent draw. | The best deterministic board-only rule copies the first source on 3/4 of decisions. Observing the missing history permits 2/2 exact conditional choices. The coarse distribution alone cannot distinguish hidden context from a random decision. |
| Only histories reached by the observation policy | From an opening both sources terminate by choosing stop. If forced instead to continue to a new context, one then chooses A and the other B. Every ordinary observed play has the same law. | They disagree on 1/1 of the unvisited decisions. A legal forced start or a direct query at that context distinguishes them. For any proposed probability of A there, its distances from the two point-mass sources sum to 1, so the larger error is at least 1/2. |

**Why the reward example generalizes.** A potential Φ assigns a number to each state. For a play of length n with discount γ, change each reward from r to `r + γΦ(next) − Φ(current)`. In the discounted sum all intermediate terms cancel, leaving the original return minus Φ(start) plus γⁿΦ(end). With zero terminal potentials, the change is common to all moves from a fixed start. The same holds for an infinite discounted play with bounded potential and 0≤γ<1. This preserves relative expected payoffs and their maximizing moves under these conditions. In the table take γ=1 and potentials 2 at the start, 3 after left, −1 after right and 0 at termination. The new reward sequences are (1,−2) and (−3,1). As a limiting case, a freely varying terminal potential can reverse the preferred move. Giving only the right terminal state potential 2 instead of zero makes right better by one point. [Reward-transformation source and scope](../REFERENCES.md#decisions-payoffs-and-observations)

```pycon
>>> from itertools import combinations
>>> from fractions import Fraction as F
>>> moves = (0,1,2)
>>> menus = [m for n in (1,2,3) for m in combinations(moves,n)]
>>> assert all(max(m,key=(3,2,0).__getitem__) == max(m,key=(9,4,0).__getitem__) for m in menus)
>>> assert (1-0)/1 == (2-0)/2
>>> def choose(order,menu): return next(x for x in order if x in menu)
>>> assert [choose(o,"ABC") for o in ("ABC","ACB")] == ["A","A"]
>>> assert [choose(o,"BC") for o in ("ABC","ACB")] == ["B","C"]
>>> def shaped(terminal): return [(3-2,1+0-3), (-1-2,0+terminal+1)]
>>> shaped(0), list(map(sum,shaped(0))), list(map(sum,shaped(2)))
([(1, -2), (-3, 1)], [-1, -2], [-1, 0])
>>> max(F(3,4), F(1,4))
Fraction(3, 4)
>>> assert all(max(p,1-p)>=F(1,2) for p in [F(0),F(1,4),F(1,2),F(1)])

```

**Self-review.** Counting counterexamples would give little guidance about which representation to use. Each pair retains a specific positive result, from exact move copying to invariant return differences. Each repair names new data or assumptions, which themselves need checking before attributing a hidden reward or internal cause to a player.
