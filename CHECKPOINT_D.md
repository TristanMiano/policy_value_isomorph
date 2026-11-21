# Checkpoint D · What changed when we replayed the frozen tic-tac-toe studies?

The **source** is the original player whose choices we want to explain. Checkpoint D asked how closely the new Phase III tools reproduce the saved measurements, and whether one table can show both copying the source's moves and winning games. The tools reproduced all 240 original move-comparison rows, 900 game-comparison rows and 90 comparisons of predicted and actual payoff. The largest difference in a reported rate was about 0.00000000000000022. The review also found and corrected two ways the new tools could misrepresent a result. [Replayed measurements](experiments/checkpoint_d.json) · [Reproduction script](scripts/review_checkpoint_d.py)

## Copying a move and winning a game

The original neural player chooses the lowest-numbered move that preserves the best outcome it can guarantee against perfect replies. Its saved network chose that move on **4,520 of 4,520** reachable boards where a player has a decision. Its average and largest guaranteed payoff given up were both zero, with a win worth +1, a draw 0 and a loss −1.

Two networks learned from the moves this player actually made. The **board network (V)** scores the board resulting from a candidate move. The **board-and-move network (Q)** scores the current board together with the candidate move. **Direct selection** takes the highest immediate fitted score. **Von Neumann–Morgenstern (VNM) expected-utility selection** tries a candidate move, plays to the end according to a stated rule for both players, and chooses the move giving the best average whole-game score, or *utility*, for the mover. Ties go to the lowest-numbered legal cell.

In the table, the utility is the fitted frequency of the **last move** of the game, counted positively if X makes it and negatively if O makes it. X maximizes that number and O minimizes it. A pair of **source copies** means the frozen player makes every subsequent move for both sides. A pair of **fitted copies** means each side takes its direct network's highest-scoring move thereafter. One continuation per candidate gives its exact utility here because these players always choose a fixed move. The source-copy rows require access to the original player during selection.

**Move agreement** is the fraction of 780 equally weighted boards withheld from network fitting on which the chosen move matches the source. **Wins, draws and losses** are probabilities for complete games from the empty board, calculated through every possible continuation. The evaluated rule plays every one of its own turns. Each seat receives half the weight. The four equally weighted opponent groups are a player choosing uniformly among legal moves, five equally weighted untrained networks each choosing its highest-scoring legal move, a player prioritizing an immediate win then a block then center then corner then any move, and the player choosing the best guaranteed outcome against perfect replies. All fixed rules break ties by lowest cell number. Each network row averages three fits with different starting weights and training orders, all using the same boards.

For each fitted network the entries below give the **mean ± sample standard deviation** across its three training runs, in percent. The standard deviation measures how much the individual fits vary. The original player's row gives exact percentages for that fixed player, board population and opponent panel.

| Move-selection rule | Who finishes each trial game? | Move agreement (%) | Wins (%) | Draws (%) | Losses (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Original neural player (exact, rounded) | — | 100.00 | 42.51 | 57.49 | 0.00 |
| V direct | — | 71.11 ± 0.15 | 39.77 ± 2.99 | 41.28 ± 18.04 | 18.94 ± 19.18 |
| V VNM, final move | Source copies | 79.02 ± 1.04 | 44.28 ± 2.67 | 55.49 ± 2.61 | 0.23 ± 0.10 |
| V VNM, final move | Fitted copies | 71.92 ± 0.51 | 42.09 ± 1.71 | 38.42 ± 7.86 | 19.49 ± 6.40 |
| Q direct | — | 83.72 ± 0.56 | 43.75 ± 1.33 | 31.53 ± 13.19 | 24.72 ± 14.52 |
| Q VNM, final move | Source copies | 77.61 ± 1.09 | 42.98 ± 2.49 | 53.40 ± 0.41 | 3.62 ± 2.49 |
| Q VNM, final move | Fitted copies | 75.21 ± 0.20 | 38.86 ± 1.44 | 42.79 ± 11.55 | 18.35 ± 11.85 |

To compare rules, subtract direct selection from VNM **within each fit**, preserving the pairing. Below, the change and its interval are in percentage points. The **95% confidence interval** estimates the mean change across repeated training on this fixed split. The **two-sided p-value** gives the probability, under a model with zero mean change, of a test statistic at least as extreme in either direction. We used the paired Student's t procedure, which assumes independent, normally distributed fit differences. Three observations give little information about the shape of that distribution. [NIST guide to paired comparisons](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm)

The intervals apply to each comparison individually. The last column applies **Holm's adjustment**, which accounts for examining the full set of 60 comparisons together, including the other utilities. The set is two network types × six VNM rules × five measures (agreement, wins, draws, losses and mean game points). This is an exploratory reanalysis of saved results. [Holm's multiple-comparison procedure](https://www.jstor.org/stable/4615733)

| Final-move VNM minus direct | Who finishes? | Measure | Mean change | Individual 95% interval | Raw p | Adjusted p |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| V | Source copies | Agreement | +7.91 | +5.67 to +10.14 | 0.00430 | 0.2492 |
| V | Source copies | Wins | +4.51 | +3.72 to +5.30 | 0.00166 | 0.0979 |
| Q | Source copies | Agreement | −6.11 | −10.14 to −2.08 | 0.02272 | 1.0000 |
| Q | Source copies | Wins | −0.77 | −4.96 to +3.43 | 0.51269 | 1.0000 |
| Q | Fitted copies | Agreement | −8.50 | −9.42 to −7.58 | 0.00063 | 0.0378 |

The last row has the smallest adjusted p-value across the 60 comparisons. An additional check asks only how often the differences point in the same direction. If either direction is equally likely on each independent fit, all three can agree in either direction with probability **2/8 = 0.25**. This is the smallest possible two-sided p-value from that direction-only check on three fits. The stronger numerical evidence for the mean in the last row depends on the similar sizes of the three decreases and on the distribution assumed for them. The archive has **one source, one board split and one opponent panel**. Variation across freshly trained sources, randomized holdouts and new opponent panels remains unmeasured. [All 60 differences, intervals, raw and adjusted p-values, and direction-only checks](experiments/checkpoint_d.json)

The review also summed fitted move frequencies across the game, positive for X's moves and negative for O's, and tried a version in which the candidate move has weight 1 and each following move receives 0.9 times the previous weight. With the latter utility and source copies, Q agreed on **84.96 ± 0.45%** of withheld boards and won **41.57 ± 3.25%** of games (mean ± sample standard deviation of three fits). Against Q direct, the paired change in agreement was **+1.24 points** (individual 95% interval −1.19 to +3.67) and in wins **−2.18 points** (−7.43 to +3.06). [Per-fit measurements and comparisons](experiments/checkpoint_d.json)

## What the review corrected

1. **How to count agreement for a randomized player.** One pilot player follows the win/block/center/corner priority rule with probability ¾ and otherwise draws uniformly from legal moves. On just 32 of the 780 withheld boards, only one move was possible. A VNM rule that maximizes expected final game points with this source playing both sides agreed on 32/32 forced moves. Phase III's original summary could display that 100% as agreement across all boards. It now gives the conditional agreement with its own denominator. Over all 780 boards, an average of **25.40%** of the move probability must be reassigned to reproduce the source. Zero would mean identical move probabilities at every board. The original pilot already reported the all-board probability distance correctly.
2. **How to show a network's raw count error.** Each legal move receives an independently fitted frequency between zero and one. Even on the same board those estimates can sum to something other than one. The board inspector formerly required a sum of one, hiding predictions from these fitted networks. It now shows each raw estimate multiplied by the actual number of visits. It separately reports the probabilities with which the deployed move-selection rule would play each move.

Here are two remaining disagreements from the first saved fit of each network. Both boards were withheld. Each string shows rows from top to bottom, a dot denotes an empty cell, and X moves next. The lists follow the legal cells in the indicated order, numbered 0–8 across the rows. Each predicted count is the network's frequency times the same 32 visits used for the observed counts. Counts are rounded to two decimals.

| Board | Legal cells in order | Observed source counts | V predicted counts | Q predicted counts |
| --- | --- | --- | --- | --- |
| `.......OX` | 0, 1, 2, 3, 4, 5, 6 | 0, 0, **32**, 0, 0, 0, 0 | 10.76, 0.29, **0.67**, 0.05, 1.88, 0.00, 0.07 | 18.52, 0.29, **1.70**, 0.00, 0.15, 0.00, 0.00 |
| `.....O..X` | 0, 1, 2, 3, 4, 6, 7 | 0, 0, 0, 0, **32**, 0, 0 | 7.82, 0.09, 0.02, 0.17, **1.97**, 0.46, 0.13 | 22.53, 0.64, 0.04, 0.18, **4.14**, 0.00, 0.00 |

Both networks selected cell 0 on both boards. The frozen source consistently selected cell 2 on the first and cell 4 on the second. Each disagreement is already visible in the fitted frequencies before any game is played forward. The examples measure prediction on unfamiliar boards. [Full-precision counts](experiments/checkpoint_d.json)

## How the players and their scores were fitted

The source takes the nine cells and the side to move as ten inputs, uses two intermediate layers of 128 units that retain positive inputs and replace negative inputs with zero, and outputs nine move scores. On all 4,520 reachable decision boards, it learned to minimize the negative natural logarithm of the probability assigned to the lowest-numbered best guaranteed move. Training used **Adam**, a method that adjusts updates using the sizes and directions of recent changes, with batches of 256 boards and an initial step size of 0.003. After 80 passes the network chose the teacher's move everywhere. Its weights were frozen before sampling. [Source architecture and training record](NEURAL_TICTACTOE_CONTROL.md)

Two copies of the frozen source played complete games. An initial collection used 256 empty-board starts. The collection used for fitting started 32 games at each of the 4,520 decision boards and retained visits whose parent board belonged to the same group as the start. This gave **243,488 observed moves**. The groups contained 2,980 boards for fitting, 760 for choosing training duration and 780 for measuring agreement. Rotations and reflections stay in the same group. A move's training label is the number of times the source took it divided by all retained visits to its parent board.

Both recovery networks used two layers of 128 units as above, then one output mapped to the interval from 0 to 1. V takes ten inputs from the resulting board. Q takes the current board and side plus nine indicators for the candidate cell, for 19 inputs. Each fit minimized the **mean squared difference** between the fitted frequency and the observed fraction, weighting each eligible training move equally. V excluded resulting-board inputs shared with the withheld groups. Both used Adam, batches of 256, an initial step of 0.003 reduced by a factor of 0.35 after pass 100, and up to 120 passes. The saved weights were those with the smallest error on the 760 boards reserved for choosing training duration. A **seed** is a number that fixes random choices such as starting weights and training order. Each cell below gives the recomputed training loss in squared move-frequency units and the saved pass number. [Visit collection and fit settings](NEURAL_TICTACTOE_CONTROL.md) · [Saved training curves](experiments/neural_tictactoe_control/training.json)

| Network | Equally weighted legal training moves | Seed 7201 | Seed 7202 | Seed 7203 | Loss, mean ± sample standard deviation |
| --- | ---: | ---: | ---: | ---: | ---: |
| V | 4,980 | 0.098445 at pass 120 | 0.101120 at pass 90 | 0.104791 at pass 60 | 0.101452 ± 0.003186 |
| Q | 10,864 | 0.046153 at pass 30 | 0.040348 at pass 30 | 0.035966 at pass 40 | 0.040822 ± 0.005110 |

## What each Phase III tool was checked against

The original Task 61 pilot had five sources. Three follow the best guaranteed outcome, win/block/center/corner priority, and randomized priority rules defined above. A fourth picks the lowest-numbered move that avoids winning immediately whenever both a win and such an alternative exist, otherwise following the priority rule. The fifth is a network trained to imitate the priority player on 528 boards and choosing its highest-scoring legal move. The original pilot fitted its V and Q networks to game points. The visit-count study above fitted V and Q to move frequencies. The review replayed both sets of saved results under their respective training targets. [Original pilot](TICTACTOE_PILOT.md) · [Visit-count study](NEURAL_TICTACTOE_CONTROL.md)

| Tasks | Check and observed result |
| --- | --- |
| 49–50 · Query a player and record how it is evaluated | Read the legal moves, source probabilities, split and saved decisions for 22,600 source–board cases. The records distinguish direct access to the source, a fitted network alone, and looking ahead using the source. The randomized source can now appear under its own original selection rule. |
| 51–52 · Encode or fit scores from choices | Converting the five source tables to scores and back recovered their move probabilities exactly. Reloading the pilot's separately fitted move imitators reproduced each legal probability within 0.000000000000001. A separate software check fitted two tiny networks to the randomized opening's probabilities and exercised the training interface. |
| 53–55 · Score complete games, ask about restricted moves and generate candidates | Recalculated all 80,835 candidate payoffs exactly from the possible terminal outcomes, using +1 for an X win, 0 for a draw and −1 for an O win. Every legal candidate was included. Tried three possible utilities for a draw on one archived opening choice. The original records contain choices on the full legal set only. Eight fresh queries to the explicit perfect-reply rule on successively smaller opening menus gave cells 0 through 7 in order. |
| 56–57 · Put scores in common units and compare methods | Subtracting the score at a common reference cell gave zero squared difference after a constant was added to all nine opening scores. Replayed the original 240 move, 900 game and 90 payoff rows. The latter compare predicted and exact game points on withheld boards, overall and grouped by prediction. For the visit-trained source, V and Q, replayed 172 move and 1,290 game rows. Every visit-study win, draw and loss rate equaled its archived value. |
| 58–59 · Repeat a fit and inspect a decision | Resuming the two opening fits reused both saved results and performed zero additional fits. Inspected two withheld disagreements for each network using the original counts, raw predictions and resulting choices at the same number of visits. |

**Scope of the finding.** A common table now shows where looking ahead improves imitation, where it moves away from the source, and what happens to wins and losses in either case. For V, the agreement gain fell from 7.91 to 0.81 percentage points when fitted copies took over from the original player to finish hypothetical games. Replaying the same weights, board split and opponents checks the measurement machinery. How much these differences recur on independently collected data, and what a learned utility predicts about new choices, remain open. Task 62, the separate step of fixing the design for independent confirmation, remains unstarted. [Research roadmap](TODO.md#resume-here)

## Reproduce the review

Install the development dependencies with `python -m pip install -e ".[dev]"`. The first command below reloads the archived weights and visits, checks the saved results, and writes a fresh report. Its two small opening fits are used solely to check fitting and resuming. The comparisons above use the archived networks.

```sh
python scripts/review_checkpoint_d.py --output checkpoint-d-replay.json
python -m pytest -q
python scripts/demo_tictactoe.py
python scripts/demo_value_atlas.py
```

The Windows and Ubuntu replays produced the same reported move agreement, probability distance, win, draw and loss rates on all 73 summary rows, and the same recomputed training losses on all six visit-trained fits. The repository suite ran 125 tests on Ubuntu. On Windows, 124 passed and the graphics-processor test was skipped because that installation of PyTorch supports only the central processor. Both demos ran on each platform.
