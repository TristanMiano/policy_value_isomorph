# Phase III · What can be recovered from a player's choices?

A **policy** is the original player's rule for choosing a move or giving each legal move a probability. A **recovery** uses specified observations of that player to produce scores. A **decoder** turns those scores into moves. Tasks 49–59 implemented a common way to query players, fit and compare recoveries, repeat an experiment, and inspect individual decisions. Checkpoint D then compared those tools with the original pilot and the visit-count study. [Review and corrections](CHECKPOINT_D.md)

## A result ordinary move agreement concealed

The example starts from the following reachable tic-tac-toe position. X moves next. Cells are numbered 0 through 8 across rows. The source always chooses the first available cell in the order **5, 8, 2, 0, 1, 3, 4, 6, 7**, for either seat and any offered menu. We fixed this rule before collecting training games.

```text
X X .
O O .
X O .
```

There are six reachable decision boards from here. Starting eight games at each one and letting two source copies finish every game gave 48 complete games and 72 observed moves. At each board the training label for each legal move was its observed count divided by all source visits there. Every board was fitted. The experiment makes a separate prediction about a changed menu that was withheld from training. [Worked position and exact continuations](docs/hand_worked_atlas.md)

A **board network (V)** scores the resulting board and the side to move there, ten inputs. A **board-and-move network (Q)** scores the original board, its side to move and nine indicators identifying the proposed cell, nineteen inputs. Each network has one hidden layer of twelve units with the hyperbolic tangent activation and one scalar output per candidate. Exponentiating and normalizing those outputs over the legal cells gives predicted visit probabilities. We fitted all six boards together for 80 updates with Adam, an optimizer that scales updates using past gradients, at learning rate 0.03. Both network types used weight initialization seeds 13 and 29. The **training loss** averages the negative natural logarithm of the probability assigned to each observed choice, weighted by the within-board frequencies and then equally across boards. Its units are nats per board, where one nat is one natural-log unit.

**Direct selection** chooses the highest fitted visit probability. The **von Neumann–Morgenstern (VNM) expected-utility rule** forces each move and averages one number assigned to the complete remaining play. Here that number sums the same fitted visit probabilities, including the forced move, adding on X turns and subtracting on O turns. There is no time discount. Both seats use the original source to finish each rollout. One rollout per candidate yields its exact utility because the continuation and game rules are deterministic. X maximizes and O minimizes the resulting utility. All four selection rules use the lowest cell number to settle ties. Direct deployment uses the fitted weights and legal moves. VNM deployment here also queries the source during continuation.

| Player and selection rule | Training loss at the final update, seeds 13 / 29 | Exact moves matched, seeds 13 / 29 | Win | Draw | Loss | Predicted cell after removing 5, seeds 13 / 29 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Original fixed-order player | — | 6/6 by definition | 41.67% | 25.00% | 33.33% | 8 on a fresh query |
| V direct | 0.0002674 / 0.0002798 | 6/6 / 6/6 | 41.67% | 25.00% | 33.33% | 2 / 2 |
| V VNM, signed sum, source continues both seats | 0.0002674 / 0.0002798 | 6/6 / 6/6 | 41.67% | 25.00% | 33.33% | 2 / 2 |
| Q direct | 0.0003180 / 0.0002770 | 6/6 / 6/6 | 41.67% | 25.00% | 33.33% | 8 / 8 |
| Q VNM, signed sum, source continues both seats | 0.0003180 / 0.0002770 | 6/6 / 6/6 | 41.67% | 25.00% | 33.33% | 2 / 2 |

For game outcomes the evaluated player takes every turn in its seat against an opponent choosing each legal move with equal probability. All continuations from the displayed position were enumerated. Each seat receives half the weight. Each rule and both fits give the same unrounded win, draw and loss probabilities as the source, with a difference of zero percentage points for each outcome and zero points per game (win +1, draw 0, loss −1). These are exact results conditional on this position, opponent and fits. The two seeds expose variation from weight initialization on identical training data.

The original eight choices on the displayed board give counts **(0, 8, 0)** for cells **(2, 5, 8)**. At eight visits the V network predicts **(0.00527, 7.99314, 0.00160)** and the Q network **(0.00013, 7.99281, 0.00706)**, averaging the two fits. Both put nearly all probability on 5. The small remaining probabilities rank the other two cells in opposite orders. We froze each rule's scores, restricted the final choice to cells 2 and 8, then asked the source. It chose **8**. Q direct predicted this **on 1/1 changed menus in each fit**. The other three methods predicted it **on 0/1 in each fit**. The denominator is one intervention, reused across fits. The source's observed counts on the original menu alone allowed either answer. [Reproduce the training, predictions and fresh queries](scripts/demo_value_atlas.py)

## Utilities, probability and fit error

A **consequence lottery** records how likely each terminal outcome and duration is after a forced move. The framework can enumerate this distribution, sample complete plays, or simulate a prefix and supply a model for the remaining outcome distribution. Applying a new utility to the same lottery reuses the simulated plays. For a separate game-payoff check, we forced cell 5 on the displayed board and let both seats choose uniformly until the end. X receives +1, 0 or −1 for a win, draw or loss. Its exact mean payoff is **0.5 points**. In 32 independent rollouts with seed 71 the mean was **0.3125**, or **0.1875 points below** the exact answer. The estimated standard error of this mean was **0.08325 points**, conditional on the board and continuation players. The tests also compare enumerated means with the earlier exact tic-tac-toe payoff calculation.

The fitted utility can use terminal results and duration with win and loss anchored at +1 and −1 before the duration adjustment. In a constructed choice the source prefers a sure draw worth ½ point to an equal chance of winning or losing. Of four candidate draw values (−1, 0, ½, 1), **½ and 1 both reproduce the choice** under a fixed lowest-numbered tie rule. Adding a second choice where two moves have identical consequences and the source selects the later move leaves a best agreement of **1/2** over the two decisions. Giving the utility distinct durations can distinguish equal-result moves. The implementation retains every best value on the declared parameter grid. [Utility implementation](src/policy_value_isomorph/consequence_utility.py) · [Constructed choices and reference checks](tests/test_consequence_utility.py)

## What is available

| Tasks | Entry points and what they measure |
| --- | --- |
| 49–50 | A policy adapter exposes legal moves, original choices and supported source information for tic-tac-toe and Connect Four. A recovery contract records what was observed, what the deployed player may access, score units, game rules and the evaluation population. Saved artifacts carry that contract and an identifier derived from their contents. [Adapters](src/policy_value_isomorph/policy_adapter.py) · [Contracts](src/policy_value_isomorph/recovery_contract.py) |
| 51–52 | Encode known discrete choices, known move probabilities or distances to a continuous action. Fit one neural score per candidate from four distinct observations: sampled moves, probability vectors, raw scores used to determine probabilities, or fresh choices within a pair. Evaluate fitted scores on common legal menus. [Exact constructions](src/policy_value_isomorph/score_encoding.py) · [Fitting and loss](src/policy_value_isomorph/distillation.py) |
| 53–55 | Evaluate lotteries, fit a common complete-play utility and record actual changed-menu queries. Reproduce candidate generation for legal moves, nearby positions, continuous actions or proposed next tokens. Measure omitted choice probability when a reference distribution is available and distance to a requested continuous action. [Utilities](src/policy_value_isomorph/consequence_utility.py) · [Preference queries](src/policy_value_isomorph/preference_probes.py) · [Candidate coverage](src/policy_value_isomorph/candidate_sets.py) |
| 56–57 | Align scores to named units before comparing numerical magnitudes. Compare rankings, probability errors, game outcomes in both seats, changes in score differences across neighboring positions, prediction against realized returns, and fresh responses under changed conditions. Replay saved fits against fixed training targets and held-out exact targets and report move gaps and game outcomes at the same training step. [Alignment rules](src/policy_value_isomorph/score_alignment.py) · [Comparison suite and metric definitions](src/policy_value_isomorph/comparison_suite.py) |
| 58–59 | Independently control the random numbers for data, rollouts, initialization, training order, architecture and candidate generation. Resume a saved matrix of runs and compare every pair of recoveries and their mean move probabilities. A **value atlas** shows a position's candidate scores, source and reconstructed choices, visit predictions, gaps, variation across fits and legal neighbors. [Repeated runs](src/policy_value_isomorph/recovery_matrix.py) · [Atlas](src/policy_value_isomorph/value_atlas.py) |

## Reproduction and scope

The new querying interface was exercised with both tic-tac-toe and Connect Four. The exhaustive game-outcome and training results above concern tic-tac-toe. Checkpoint D replayed the full frozen pilot. Larger-game performance remains open. [Pilot replay](CHECKPOINT_D.md)

At the end of Task 59, the suite passed all 124 tests on Ubuntu. Windows passed 123 and skipped the graphics-processor test on its processor-only installation. Both platforms reproduced the four changed-menu predictions in the table. The three-board demo also printed the sample states, payoffs and moves.

Run `python scripts/demo_value_atlas.py` for the worked comparison or add `--json` for the per-fit move probabilities, sample standard deviations, originally observed counts, predicted counts, rollout definitions and neighborhood records. Run `python -m pytest -q` for the repository suite. The original three-board demo remains available as `python scripts/demo_tictactoe.py`.

**Self-review.** Choosing a small, fully observed tree lets us check every game outcome and isolate disagreement about the runner-up. It also makes ordinary imitation easier. Only the changed menu was withheld. Which method best predicts a larger population of changed menus or unseen boards, and how much playing strength varies against other opponents, remain open. [Roadmap and next checkpoint](TODO.md#resume-here)
