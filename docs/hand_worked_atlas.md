# One board, three legal moves, several meanings of value

A **policy** is a rule for choosing among offered moves. Our teaching example uses a player that picks the first available cell in the fixed order **5, 8, 2, 0, 1, 3, 4, 6, 7**, in either seat and on every offered subset. Number cells 0 through 8 in row order. X moves next on the board below, reachable by playing 0, 3, 1, 4, 6, 7 from the empty opening.

```text
X X .
O O .
X O .
```

The legal moves are 2, 5 and 8. The source chooses **5**. Move 2 immediately wins for X. This is a worked example of a known rule. Its counts can be obtained by querying the rule directly, so there is no neural training loss or inference about how often disagreements occur on other boards. [Shared notation](../formalism/00_notation.md)

## What the decisions reveal

A **visit frequency** is the count for a move divided by the number of observed choices at its parent board. Eight calls on the original menu give counts (0, 8, 0), hence frequencies (0, 1, 0), in legal-cell order (2, 5, 8). This gives 5 a score gap of 1 over either rival. It leaves 2 and 8 tied. When 5 is removed, a fresh query picks 8 over 2. The other pair queries choose 5 from {2,5} and 5 from {5,8}. Under the fixed-order assumption the **revealed order** is 5 above 8 above 2. The queries identify the order. The numerical spacing below is a choice of units.

| Move | Original count / visits | Revealed rank score (lowest = 0) | Rank normalized to [0,1] | Sampling probability at temperature 1 using the rank score |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0/8 | 0 | 0 | 0.09003 |
| 5 | 8/8 | 2 | 1 | 0.66524 |
| 8 | 0/8 | 1 | 0.5 | 0.24473 |

For **softmax**, sample a legal move with probability proportional to the exponential of its score divided by a positive **temperature**. Here each move has equal base weight. The normalizing denominator is `1 + exp(2) + exp(1)`. Subtracting 2 from every rank leaves these probabilities unchanged. Halving all ranks and the temperature also preserves them. With normalized ranks at temperature 1 the probability of cell 5 is only 0.50648. The source itself chooses 5 with probability 1. The first sampling rule therefore differs from the source by 0.33476 in total variation distance, half the sum of absolute probability differences. Under largest-score selection either rank scale chooses 5 with certainty.

## Complete plays after forcing each move

A **rollout** forces a candidate first and then plays to the end under a specified continuation. Write a play as its sequence of remaining cell numbers. X receives +1 game point for a win, 0 for a draw and −1 for a loss. In addition define a **visit-based utility** for the same play. Query the source once at every subsequent decision board in this example. Its chosen move has frequency 1 and the other legal moves frequency 0. Sum these move frequencies over the remaining play, adding on X's turns and subtracting on O's, starting with the forced X move. There is no time decay. One such utility is applied to every candidate and every continuation.

The **uniform continuation** chooses each available cell with equal probability at each later decision in both seats. The **source continuation** uses the fixed cell order in both seats. Its only path for each forced move is marked in the final column.

| Remaining moves | Result for X | X game points | Visit-based utility calculation | Probability under uniform continuation | Probability under source continuation |
| --- | --- | ---: | --- | ---: | ---: |
| 2 | Win | +1 | 0 | 1 | 1 |
| 5, 2, 8 | Draw | 0 | 1 − 0 + 1 = 2 | ½ | 0 |
| 5, 8, 2 | Win | +1 | 1 − 1 + 1 = 1 | ½ | 1 |
| 8, 2, 5 | Draw | 0 | 0 − 0 + 1 = 1 | ½ | 0 |
| 8, 5 | Loss | −1 | 0 − 1 = −1 | ½ | 1 |

A **VNM** (von Neumann–Morgenstern) score is the probability-weighted average of a declared complete-play utility after the forced move. X chooses the largest score, resolving ties by lowest cell number. For example, forcing 5 under uniform continuation yields ½ game point and 1½ units of visit-based utility. The game has deterministic transitions, so the expected final points of the board immediately after a forced move equal those of the board-and-move pair under the same continuation.

| Score or selection rule | Move 2 | Move 5 | Move 8 | Largest minus next largest | Selected move | Matches source on this board |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct visit frequency | 0 | 1 | 0 | 1 | 5 | 1/1 |
| Revealed rank | 0 | 2 | 1 | 1 | 5 | 1/1 |
| VNM game points, both seats use source | 1 | 1 | −1 | 0 | 2 | 0/1 |
| VNM game points, both seats uniform | 1 | 0.5 | −0.5 | 0.5 | 2 | 0/1 |
| VNM visit-based utility, both seats use source | 0 | 1 | −1 | 1 | 5 | 1/1 |
| VNM visit-based utility, both seats uniform | 0 | 1.5 | 0 | 1.5 | 5 | 1/1 |

With source continuation, moves 2 and 5 both win with probability 1. Choosing 2 instead of 5 gives up **0 game points** while changing the move. With uniform continuation the choices have different outcomes. Move 2 wins with probability 1. Move 5 wins with probability ½ and draws with probability ½. Move 8 draws with probability ½ and loses with probability ½. These are conditional outcomes from this board. The value of deploying a recovered player for an entire game from the opening requires its rule on other boards. The VNM source-continuation rows use fresh source calls after the forced move. Uniform continuation needs only the environment once the frequency table is available.

**Sampled rollouts.** The path table also gives exact answers against which to check finite samples. Draw eight complete continuations with replacement per candidate. Use Python's `Random(36).choice` on the path rows for each move in their displayed order, including the singleton for cell 2, with one random stream across candidates. Cells 2, 5 and 8 give sample mean game points of 1, 0.875 and −0.5. Their exact means are 1, 0.5 and −0.5. For each candidate the standard error of the sample mean under its known path distribution is the per-draw standard deviation divided by √8, giving 0, 0.17678 and 0.17678 points. Enumerating every path or taking the sole deterministic source continuation has zero simulation error. [Executable check of paths, counts, probabilities and sampled means](../tests/test_hand_worked_atlas.py)

**Self-review.** Using only source continuation would have hidden the cost of choosing move 5 when later replies are random. Using game points alone would have hidden an exact visit-based explanation of that same choice. The restricted-menu query exposes an ordering missed by the direct frequency table. What remains open is how well a compact learned utility predicts such choices on unobserved boards. [Evaluation populations and rankings](evaluation_contract.md)
