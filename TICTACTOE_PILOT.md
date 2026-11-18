# Tic-tac-toe move choices and game outcomes

November 17, 2026. A **source policy** is an original player's rule for choosing a legal move or assigning probabilities to legal moves. A **recovered player** uses scores constructed or learned from that source. A **payoff** awards the player one point for a win, zero for a draw, and minus one for a loss. We measured how closely each recovered player copied its source, how many payoff points its moves gave up, and how often it won a whole game.

## The five source policies

X moves first. Cells are numbered 0 through 8 from left to right across the rows. An **expected payoff**, also called a **return** here, is the average of the payoffs using the probabilities of the possible outcomes.

| Player used as the source | How it chooses a move |
| --- | --- |
| Priority-rule player (`heuristic`) | Take an immediate win, block an opponent's immediate win, take the center, take a corner, then take any available cell, in that order. Within the first applicable rule take the lowest-numbered legal cell. |
| Perfect-play player (`minimax`) | Consider every legal continuation assuming the opponent always makes its best reply. Choose the move with the highest guaranteed payoff. Take the lowest-numbered move in a tie. This rule is called **minimax**. |
| Missed-win player (`miss_win`) | When an immediate win and a move that does not win immediately are both available, choose the lowest-numbered such alternative. Otherwise follow the priority-rule player. |
| Trained neural player (`trained_net`) | A **neural network** calculates scores using adjustable numbers called **weights**. This one learned to copy the priority-rule player's moves on 528 distinct boards. Its inputs are the nine cells and whose turn it is. Its 16 intermediate units produce nine move scores. The weights are **frozen**, meaning they are held fixed during evaluation. It takes the highest-scoring legal cell and resolves ties by lowest cell number. |
| Randomized priority player (`stochastic_heuristic`) | On each turn independently use the priority rule with probability 3/4. With probability 1/4 choose a legal cell with equal probability. The random choice can also land on the priority move. |

The first four sources are **deterministic** (one fixed move per board). The fifth is **stochastic** (it draws a move from a probability distribution). The published tables identify each source by the name in parentheses. [Player settings and definitions](experiments/tictactoe_pilot/manifest.json)

## How the recovered players choose

- **Original** (`source`) plays the source rule itself. **Exact behavioral copy** (`behavior_exact`) uses the source's move probabilities directly. Expressing those probabilities as scores and decoding them back into probabilities supplies a reference with zero difference on every board.
- **Exact return** (`return_exact`) calculates the expected payoff of each move assuming the source plays both sides for the rest of the game. It chooses the best payoff for the player to move and selects the lowest-numbered cell among ties.
- **Direct imitation** (`policy_mlp`) is a new network trained to predict the source's moves. It takes the highest-scoring legal move for a deterministic source. For the randomized source it draws moves using **softmax**, which exponentiates each legal score and divides by the sum to obtain its probability.
- **Learned board value**, written **V** (`value_mlp`), is a network trained to predict X's final payoff from a board. It scores each move using the board the move would produce. **Learned action value**, written **Q** (`q_mlp`), is a network trained to predict that payoff from the current board and candidate move together. Both learn from **rollouts**, games played to completion using the source. X chooses the largest predicted X payoff and O the smallest.

In the project's **von Neumann–Morgenstern (VNM) expected-utility formulation**, each move induces a **lottery**, the possible outcomes and their probabilities. The move's score is the probability-weighted average of the numerical utilities assigned to those outcomes. Here the utility is the game's win/draw/loss payoff. Exact return calculates that average with the source continuing on both sides. Learned V and Q estimate it. Tasks 70 and 71 extend the comparison to sampling moves according to their scores and to utilities learned from choices. [Planned comparisons](TODO.md#formulations-to-compare-in-games)

## Move agreement and probability differences

**Move agreement** is the fraction of evaluated boards on which two players choose the same move. For a randomized source we use **total variation distance (TV)**. On each board, take half the sum of the absolute differences between the two players' legal-move probabilities. A distance of 0 means identical probabilities, and 1 means the players assign probability to disjoint sets of moves. We average this distance over boards. Smaller values mean closer imitation.

The full population comprises all 4,520 reachable boards with a move remaining. The 780 **test boards** were held out of training. **Uniform** weights give each board equal weight. **Visited** weights are proportional to the probability of reaching each board in **self-play**, where the source plays both sides from the empty board. We divide by the total weight in the population being evaluated.

| Frozen source | Exact return, uniform | Exact return, visited | Learned V, uniform test | Learned V, visited test | Test boards with positive visitation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Perfect play | 100.00% | 100.00% | 56.92% | 50.00% | 2 |
| Missed win | 37.70% | 33.33% | 33.85% | 66.67% | 3 |
| Priority rule | 75.93% | 77.78% | 62.69% | 66.67% | 3 |
| Trained neural player | 56.62% | 55.56% | 50.51% | 33.33% | 3 |
| Randomized priority (TV) | 0.2764 | 0.5381 | 0.4376 | 0.4872 | 780 |

Each deterministic source visited nine boards in its game from the empty board. Its visited-test percentages describe only two or three decisions. The randomized priority player reached all 4,520 decision boards with positive probability and averaged 7.613 decisions per game. Its test boards contributed 2.594 expected decisions per game. [Full measurements by player and board population](experiments/tictactoe_pilot/results/fidelity.csv)

## Why the moves differed

We calculated exact returns in two ways, once from X's payoff on the resulting boards and once from the acting player's payoff for each candidate move. Their difference was zero on all 16,167 legal moves for each source. The **mean payoff gap** measures the size of a deviation. Subtract the expected payoff of the chosen move from that of the best legal move for the acting player, continuing with the source after that one move. Average over move probabilities and boards. Zero points means the choices have equal return. Two points is the difference between a guaranteed loss and a guaranteed win.

| Deterministic source | Same move as exact return | Different equally good move | Strictly lower return | Mean payoff gap of source (points, all 4,520 boards equally weighted) |
| --- | ---: | ---: | ---: | ---: |
| Perfect play | 4,520 | 0 | 0 | 0.000 |
| Missed win | 1,704 | 1,016 | 1,800 | 0.558 |
| Priority rule | 3,432 | 853 | 235 | 0.054 |
| Trained neural player | 2,559 | 1,094 | 867 | 0.271 |

For the randomized priority player, those categories accounted for 72.36%, 5.63%, and 22.02% of move probability. Its mean payoff gap was 0.173 points. A player choosing exclusively among the best moves must reassign the 22.02% probability on lower-return moves. The smallest possible TV distance under that restriction is therefore 0.2202 when the player can distribute that probability among tied best moves. Always choosing the lowest-numbered best move gives an actual distance of 0.2764. [Per-board choices, values, and probabilities](experiments/tictactoe_pilot/results/boards.csv.gz)

Three held-out positions illustrate the attribution. Cells are numbered 0–8 in row-major order, and the three board rows are separated by slashes.

- **Tie.** On `.../.../.OX`, X’s priority rule takes the center (4). Moves 2, 4, and 5 all yield an X win when the priority rule continues. The exact return player picks 2 and learned V picks 4. Both give up zero points.
- **Missed win.** On `.../OXX/OXO`, X can take 0, 1, or 2. Their returns under the missed-win source are 0, +1, and 0. The source chooses 0 and draws, giving up one point. Exact return and both learned return models choose 1 and win immediately.
- **Prediction error.** On `..O/O.X/X.X`, O must play 7 to draw against perfect play. Each other legal move loses. Learned V predicts X payoffs of 0.954, 1.547, 1.058, and 1.517 for moves 0, 1, 4, and 7. O minimizes those predictions and chooses 0, giving up one point. Learned Q chooses 7. The networks' output layers allow predictions outside the game payoff range.

## Comparisons on the same 780 test boards

The first four measurement columns compare move choices on equally weighted test boards. **Mean squared error (MSE)** is the average squared difference between a predicted and exact X payoff, in squared payoff points. These two error columns cover all 2,747 legal moves on the test boards. The final column counts changes from the exact return player's move and distinguishes an equally good alternative from a lower-return move.

| Source | Exact return | Direct imitation | Learned V | Learned Q | V MSE | Q MSE | V changes (tied / lower return) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Perfect play | 100.00% | 76.54% | 56.92% | 49.36% | 0.583 | 0.520 | 203 / 133 |
| Missed win | 38.46% | 70.77% | 33.85% | 33.85% | 0.500 | 0.483 | 250 / 215 |
| Priority rule | 79.10% | 85.38% | 62.69% | 48.21% | 0.478 | 0.575 | 219 / 126 |
| Trained neural player | 56.92% | 90.90% | 50.51% | 53.08% | 0.625 | 0.624 | 201 / 153 |
| Randomized priority (TV) | 0.2540 | 0.1847 | 0.4376 | 0.6145 | 0.267 | 0.379 | 108 / 153 |

For the trained neural player, replacing exact returns with learned V restored its move on 95 boards and lost it on 145, a net change of −50/780, or −6.41 percentage points of agreement. Direct imitation had 40.38 percentage points more agreement than learned V. In the same comparison, direct imitation gave up 0.321 payoff points per decision and learned V gave up 0.277. Copying this source more closely sometimes copied its lower-return moves. For the randomized priority player, direct imitation's TV distance was 0.1847 on equally weighted test boards and 0.3014 when weighted by the source's visits. Each learned method was trained once per source. Task 67 will vary training settings and measure the resulting changes. [All move comparisons](experiments/tictactoe_pilot/results/fidelity.csv) · [Predicted and observed averages grouped by predicted score](experiments/tictactoe_pilot/results/calibration.csv)

## Playing strength

**Win, draw, and loss rates** are the probabilities of those outcomes when a player plays entire games from the empty board. A **seat** is playing as X or as O. The two seats have equal weight. The **opponent panel** is a weighted group of opponents. It gives one quarter each to (1) a player that chooses uniformly among legal moves each turn, (2) five untrained neural players, (3) the priority-rule player, and (4) the perfect-play player. Each untrained neural player has its own fixed random weights and always selects its highest-scoring legal move, taking the lowest-numbered cell in a tie. Each accounts for 5% of the panel. The **game score** is win probability minus loss probability, in points per game.

| Source | Original wins / draws / losses (%) | Original score | Exact return (win %, score) | Direct imitation (win %, score) | Learned V (win %, score) | Learned Q (win %, score) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Perfect play | 42.51 / 57.49 / 0.00 | +0.425 | 42.51%, +0.425 | 43.87%, +0.050 | 45.07%, +0.185 | 38.89%, −0.214 |
| Missed win | 18.68 / 68.26 / 13.07 | +0.056 | 32.38%, −0.305 | 14.79%, −0.213 | 29.74%, −0.370 | 18.31%, −0.516 |
| Priority rule | 45.21 / 54.64 / 0.16 | +0.450 | 51.99%, +0.377 | 46.88%, +0.465 | 42.74%, +0.135 | 33.68%, −0.287 |
| Trained neural player | 38.89 / 43.83 / 17.28 | +0.216 | 40.31%, −0.151 | 38.56%, +0.087 | 36.68%, −0.188 | 33.73%, −0.001 |
| Randomized priority | 41.67 / 37.39 / 20.94 | +0.207 | 56.05%, +0.433 | 40.16%, +0.120 | 39.24%, +0.081 | 28.14%, −0.170 |

Against the five untrained neural opponents alone, the original perfect-play, missed-win, priority-rule, trained neural, and randomized priority players won 80.00%, 40.00%, 90.00%, 80.00%, and 83.00%, respectively. Against the full panel, exact return raised the trained neural player's win probability from 38.89% to 40.31% while its loss probability rose from 17.28% to 55.45%. A move scored assuming the source will continue on both sides can have a different consequence when the recovered player takes every subsequent turn against another opponent.

The 900 matchup rows calculate outcome probabilities over all possible plays of these specified players. They include each seat, the equal-seat average, and differences from the original against the same opponent in the same seat. **Head-to-head** means the recovered player plays against its own source and is reported separately from the panel average. [Full matchup table](experiments/tictactoe_pilot/results/matchups.csv)

## Reproducing the measurements

The experiment settings and original player weights were fixed in code revision `eaf79b3` before measuring the results. The current reporting revision was made after the run. It gives numerical differences in choices and payoffs. A **checksum** is a number calculated from file contents for comparison between runs. A repeat run reproduced the saved predictions, fitted weights, and game outcomes.

Recovery used 2,980 training boards and reserved another 760 **validation boards**, positions set aside during training, as well as the 780 test boards. Rotations and reflections of the same board were assigned to the same group. Imitation used training decisions and Q used their 10,864 legal moves. V had 1,215 eligible training positions after also excluding every resulting board that a validation or test decision would ask it to score. V and Q consequently had different numbers of training examples. For the randomized priority source we used eight independent move draws per training decision and sixteen rollouts per estimated payoff. [Study settings](experiments/tictactoe_pilot/manifest.json) · [Training counts and file checksums](experiments/tictactoe_pilot/results/run.json)

Run these commands from the repository root.

```sh
python scripts/run_tictactoe_pilot.py --output-dir artifacts/tictactoe_pilot_replay
python -m pytest -q
python scripts/demo_tictactoe.py
```

The recorded run used Python 3.13.14 on Windows. The checksums in `run.json` use the SHA-256 algorithm. The compressed board table has 22,600 rows. Per-move lists follow the cell order in `legal_moves`. Exact probabilities, visit weights, and returns are stored as integers or fractions. The matchup table enumerates all outcomes and stores their probabilities to about 15 significant digits. The **seeds** (integers that reproduce random choices) listed for Task 62 are reserved for a separate experiment. [Per-board measurements](experiments/tictactoe_pilot/results/boards.csv.gz) · [Fitted network weights](experiments/tictactoe_pilot/results/recovery_weights.json)
