# Comparing two weight-update rules on a fixed tic-tac-toe problem

**AdamW** updates each weight using running averages of its gradient and squared gradient, with separate weight shrinkage. **Muon** transforms the update to a matrix as a whole. Here it updates the input-to-hidden weight matrix, while AdamW updates biases and the output layer. The original player takes an immediate win, blocks an immediate loss, takes the center, takes a corner or takes any legal move, in that order with lowest-cell ties. A game starts empty against an opponent that selects every legal cell with equal probability. [Frozen settings and measurement definitions](protocol.json) · [PyTorch Muon implementation](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html)

The **policy** learns to copy the original player's action on 64 training boards. The **board-value network (V)** learns the final payoff for X on 64 eligible positions. The **board-and-move network (Q)** learns that payoff for all 231 legal moves on the 64 training boards. For each payoff label both sides follow the original player to game end. A win for X is +1, a draw 0 and a loss −1. The policy takes its largest legal score. X maximizes and O minimizes the payoff predicted by V or Q, breaking ties by lowest cell. These are game-payoff fits. The experiment holds the data, starting weights and example order fixed within each pair. It uses 16 hidden units, six passes, 32-bit precision and one CPU thread per fit. Each of four paired starting-weight seeds receives equal weight.

**Move agreement** counts the withheld boards where the fitted player picks the original move. **Win/draw/loss** probabilities enumerate play from the opening and give equal weight to each seat. The original player's win/draw/loss rates were 90.82% / 8.54% / 0.63%. Each difference below subtracts AdamW from Muon. The ± quantity is one standard error of that difference across four paired fits (sample standard deviation divided by two). Each pair uses the same 780 equally weighted test boards.

| Network | Mean matches out of 780, AdamW → Muon | Agreement change, percentage points | Win / draw / loss %, AdamW → Muon | Win change, percentage points |
| --- | ---: | ---: | --- | ---: |
| Policy | 364.25 → 376.50 | +1.57 ± 0.18 | 43.24 / 17.82 / 38.95 → 44.21 / 17.87 / 37.92 | +0.98 ± 0.87 |
| V | 373.00 → 389.25 | +2.08 ± 0.80 | 56.55 / 11.13 / 32.32 → 62.50 / 10.93 / 26.58 | +5.94 ± 5.01 |
| Q | 308.50 → 286.00 | −2.88 ± 0.53 | 43.13 / 12.51 / 44.36 → 45.44 / 14.21 / 40.35 | +2.30 ± 4.48 |

Training loss is the mean across equally weighted examples during a pass. For the policy it measures minus the natural logarithm of the observed move's probability. For V and Q it measures squared error in game points. Elapsed time includes model setup and all six training passes. The timings describe this machine and may vary with other processes.

| Network | Final training loss, AdamW → Muon | Loss difference | Seconds per fit, AdamW → Muon | Additional seconds |
| --- | ---: | ---: | ---: | ---: |
| Policy | 0.74162 → 0.74542 | +0.00380 ± 0.00193 | 0.0841 → 0.1154 | 0.0312 ± 0.0048 |
| V | 0.51738 → 0.49131 | −0.02607 ± 0.01310 | 0.0716 → 0.1031 | 0.0315 ± 0.0026 |
| Q | 0.51457 → 0.42880 | −0.08577 ± 0.00730 | 0.2457 → 0.5655 | 0.3198 ± 0.0034 |

**Implementation decision.** Retain Muon as an opt-in benchmark. It improved agreement for two networks and lowered it for Q even as Q's training loss fell. Promoting it to the normal trainers would require supporting two sets of optimizer state per fit and a PyTorch build that implements Muon. The comparison reuses all three existing trainers and adds no runtime dependency. These estimates describe one learning rate and one training budget.

Run `python scripts/compare_optimizers.py --compare-muon --output comparison.json` on a PyTorch build with Muon. The recorded run used Ubuntu and PyTorch 2.11.0. All six pass losses for every fit, per-seed game outcomes and paired uncertainties are saved. [Measurements](results.json) · [Runnable comparison](../../scripts/compare_optimizers.py)
