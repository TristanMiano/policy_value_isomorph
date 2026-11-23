# Three-game measurements

The **original** chooses the legal move with the highest output from the frozen player. **V** estimates the frequency of a move from its resulting board. **Q** also sees the preceding board and candidate move. **Direct** chooses the largest immediate fitted frequency. **Von Neumann–Morgenstern (VNM) selection** compares the expected numerical scores of complete games, viewed from the player choosing now. In the primary comparison each candidate is played out by two direct copies of the fitted network and only the final move’s fitted frequency counts, positive for the first player and negative for the second. The lowest move number breaks exact ties. No original is consulted during either primary selection.

Each value is a mean ± sample standard deviation across independently trained originals, after averaging both board partitions and both fresh fits in each partition. The three-game row gives each named game one third of the weight. Its spread describes the average of one independent original per game. Agreement is the fraction of withheld positions with the same chosen move. Wins, draws and losses are percentages of games against the specified opponents, averaged equally over both seats and the four opponent groups. Points are +1 for a win, 0 for a draw and −1 for a loss. Rate differences are in percentage points.

## Planned paired comparisons

Each difference subtracts direct selection on the same fit and evaluation cases. A 95% confidence interval describes uncertainty in the mean. The resampling interval repeats the calculation on 5,000 samples of entire originals, keeping the two Connect Four target groups in their original proportions. The raw p-value is the two-sided probability of a result at least this extreme under the model of zero mean difference. Holm adjustment accounts for examining 24 planned differences together. Intervals are individual. Within a game, the last column counts originals with a higher, equal or lower VNM result. For the three-game mean it shows the equally game-weighted percentages in those categories. Full precision and intervals on recurrence are in the saved results.

| Game | Originals | Network | Measure | Direct | VNM | VNM − direct | 95% interval | Resampling 95% | Raw p | Holm p | Higher / equal / lower |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tic-tac-toe | 32 | V | Same move (%) | 73.95 ± 2.56 | 68.56 ± 2.94 | -5.39 ± 2.54 | [-6.30, -4.47] | [-6.25, -4.54] | 3.36e-13 | 5.72e-12 | 0 / 0 / 32 |
| Tic-tac-toe | 32 | V | Wins (%) | 40.00 ± 3.20 | 44.19 ± 4.69 | 4.19 ± 5.14 | [2.33, 6.04] | [2.47, 5.94] | 6.58e-05 | 0.000461 | 26 / 0 / 6 |
| Tic-tac-toe | 32 | V | Points/game | 0.150 ± 0.095 | 0.183 ± 0.095 | 0.033 ± 0.086 | [0.002, 0.063] | [0.003, 0.062] | 0.0395 | 0.118 | 23 / 0 / 9 |
| Tic-tac-toe | 32 | Q | Same move (%) | 78.39 ± 2.14 | 67.10 ± 2.50 | -11.29 ± 2.72 | [-12.27, -10.31] | [-12.19, -10.37] | 2.78e-21 | 5.56e-20 | 0 / 0 / 32 |
| Tic-tac-toe | 32 | Q | Wins (%) | 44.04 ± 2.90 | 40.99 ± 4.11 | -3.04 ± 3.62 | [-4.35, -1.74] | [-4.30, -1.83] | 4.29e-05 | 0.000343 | 8 / 0 / 24 |
| Tic-tac-toe | 32 | Q | Points/game | 0.217 ± 0.090 | 0.121 ± 0.092 | -0.096 ± 0.105 | [-0.134, -0.058] | [-0.131, -0.060] | 1.42e-05 | 0.000142 | 5 / 0 / 27 |
| Connect Four | 24 | V | Same move (%) | 76.89 ± 4.43 | 23.81 ± 7.06 | -53.08 ± 5.11 | [-55.24, -50.92] | [-54.65, -51.41] | 3.89e-25 | 8.57e-24 | 0 / 0 / 24 |
| Connect Four | 24 | V | Wins (%) | 43.99 ± 20.40 | 53.73 ± 18.27 | 9.73 ± 8.63 | [6.09, 13.38] | [6.40, 13.10] | 1.28e-05 | 0.000141 | 21 / 0 / 3 |
| Connect Four | 24 | V | Points/game | -0.118 ± 0.411 | 0.077 ± 0.367 | 0.194 ± 0.174 | [0.121, 0.268] | [0.127, 0.263] | 1.48e-05 | 0.000142 | 21 / 0 / 3 |
| Connect Four | 24 | Q | Same move (%) | 92.98 ± 6.24 | 24.86 ± 4.38 | -68.12 ± 6.95 | [-71.06, -65.19] | [-69.73, -66.45] | 1.42e-24 | 2.98e-23 | 0 / 0 / 24 |
| Connect Four | 24 | Q | Wins (%) | 57.45 ± 11.30 | 49.37 ± 17.69 | -8.09 ± 9.55 | [-12.12, -4.05] | [-10.90, -5.47] | 0.00039 | 0.00195 | 5 / 1 / 18 |
| Connect Four | 24 | Q | Points/game | 0.160 ± 0.238 | -0.009 ± 0.357 | -0.168 ± 0.187 | [-0.247, -0.089] | [-0.224, -0.114] | 0.000207 | 0.00124 | 4 / 1 / 19 |
| 4×4 Reversi | 24 | V | Same move (%) | 83.91 ± 1.49 | 82.70 ± 1.60 | -1.20 ± 2.14 | [-2.11, -0.30] | [-2.02, -0.33] | 0.0115 | 0.0461 | 5 / 0 / 19 |
| 4×4 Reversi | 24 | V | Wins (%) | 46.27 ± 5.90 | 34.78 ± 5.39 | -11.49 ± 6.88 | [-14.40, -8.59] | [-14.32, -8.85] | 2.89e-08 | 3.76e-07 | 0 / 1 / 23 |
| 4×4 Reversi | 24 | V | Points/game | -0.024 ± 0.112 | -0.204 ± 0.101 | -0.180 ± 0.129 | [-0.235, -0.125] | [-0.233, -0.130] | 5.89e-07 | 7.07e-06 | 2 / 0 / 22 |
| 4×4 Reversi | 24 | Q | Same move (%) | 85.86 ± 1.26 | 79.89 ± 1.67 | -5.97 ± 1.92 | [-6.78, -5.16] | [-6.71, -5.17] | 1.61e-13 | 2.89e-12 | 0 / 0 / 24 |
| 4×4 Reversi | 24 | Q | Wins (%) | 49.87 ± 4.78 | 36.87 ± 5.02 | -13.00 ± 6.43 | [-15.72, -10.29] | [-15.51, -10.51] | 9.16e-10 | 1.28e-08 | 0 / 0 / 24 |
| 4×4 Reversi | 24 | Q | Points/game | 0.052 ± 0.085 | -0.166 ± 0.090 | -0.218 ± 0.101 | [-0.261, -0.175] | [-0.257, -0.177] | 2.71e-10 | 4.07e-09 | 0 / 0 / 24 |
| Three games, equal weight | 32/24/24 | V | Same move (%) | 78.25 ± 1.78 | 58.36 ± 2.61 | -19.89 ± 2.03 | [-20.71, -19.07] | [-20.56, -19.20] | 1.55e-37 | 3.71e-36 | 6.9% / 0.0% / 93.1% |
| Three games, equal weight | 32/24/24 | V | Wins (%) | 43.42 ± 7.16 | 44.23 ± 6.54 | 0.81 ± 4.06 | [-0.81, 2.43] | [-0.74, 2.36] | 0.322 | 0.643 | 56.2% / 1.4% / 42.4% |
| Three games, equal weight | 32/24/24 | V | Points/game | 0.003 ± 0.145 | 0.018 ± 0.131 | 0.016 ± 0.078 | [-0.016, 0.047] | [-0.015, 0.045] | 0.322 | 0.643 | 55.9% / 0.0% / 44.1% |
| Three games, equal weight | 32/24/24 | Q | Same move (%) | 85.74 ± 2.24 | 57.28 ± 1.77 | -28.46 ± 2.57 | [-29.51, -27.41] | [-29.12, -27.79] | 2.04e-33 | 4.7e-32 | 0.0% / 0.0% / 100.0% |
| Three games, equal weight | 32/24/24 | Q | Wins (%) | 50.45 ± 4.20 | 42.41 ± 6.28 | -8.05 ± 4.02 | [-9.68, -6.41] | [-9.36, -6.75] | 5.29e-13 | 8.46e-12 | 15.3% / 1.4% / 83.3% |
| Three games, equal weight | 32/24/24 | Q | Points/game | 0.143 ± 0.089 | -0.018 ± 0.126 | -0.161 ± 0.079 | [-0.192, -0.129] | [-0.186, -0.135] | 1.4e-13 | 2.65e-12 | 10.8% / 1.4% / 87.8% |

## Absolute outcomes and alternative whole-game scores

Terminal scores only the final move. Sum counts every move’s signed frequency. Discounted sum also multiplies each successive move’s weight by 0.9. Both players use fitted direct copies throughout these rollouts. The original row is the frozen player itself. Each rule takes its best-scoring move. Game points and agreement have different units.

| Game | Originals | Rule | Agreement (%) | Wins (%) | Draws (%) | Losses (%) | Points/game |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tic-tac-toe | 32 | Original | 100.00 ± 0.00 | 45.55 ± 6.01 | 54.45 ± 6.01 | 0.00 ± 0.00 | 0.455 ± 0.060 |
| Tic-tac-toe | 32 | V direct | 73.95 ± 2.56 | 40.00 ± 3.20 | 35.00 ± 6.01 | 25.00 ± 7.25 | 0.150 ± 0.095 |
| Tic-tac-toe | 32 | V VNM terminal | 68.56 ± 2.94 | 44.19 ± 4.69 | 29.88 ± 6.96 | 25.93 ± 6.89 | 0.183 ± 0.095 |
| Tic-tac-toe | 32 | V VNM sum | 75.38 ± 2.06 | 45.93 ± 4.83 | 39.18 ± 4.76 | 14.89 ± 5.32 | 0.310 ± 0.090 |
| Tic-tac-toe | 32 | V VNM discounted sum | 76.48 ± 2.03 | 46.11 ± 4.28 | 39.96 ± 4.52 | 13.93 ± 5.40 | 0.322 ± 0.086 |
| Tic-tac-toe | 32 | Q direct | 78.39 ± 2.14 | 44.04 ± 2.90 | 33.66 ± 7.89 | 22.30 ± 7.96 | 0.217 ± 0.090 |
| Tic-tac-toe | 32 | Q VNM terminal | 67.10 ± 2.50 | 40.99 ± 4.11 | 30.17 ± 4.99 | 28.84 ± 6.18 | 0.121 ± 0.092 |
| Tic-tac-toe | 32 | Q VNM sum | 79.70 ± 1.92 | 44.09 ± 3.97 | 35.33 ± 7.71 | 20.58 ± 6.12 | 0.235 ± 0.069 |
| Tic-tac-toe | 32 | Q VNM discounted sum | 81.06 ± 1.90 | 44.74 ± 4.41 | 35.28 ± 7.42 | 19.98 ± 5.73 | 0.248 ± 0.070 |
| Connect Four | 24 | Original | 100.00 ± 0.00 | 58.59 ± 12.01 | 0.59 ± 1.80 | 40.82 ± 12.84 | 0.178 ± 0.248 |
| Connect Four | 24 | V direct | 76.89 ± 4.43 | 43.99 ± 20.40 | 0.26 ± 0.70 | 55.75 ± 20.68 | -0.118 ± 0.411 |
| Connect Four | 24 | V VNM terminal | 23.81 ± 7.06 | 53.73 ± 18.27 | 0.21 ± 0.33 | 46.06 ± 18.46 | 0.077 ± 0.367 |
| Connect Four | 24 | V VNM sum | 41.07 ± 5.98 | 50.42 ± 27.48 | 1.73 ± 1.92 | 47.85 ± 28.46 | 0.026 ± 0.559 |
| Connect Four | 24 | V VNM discounted sum | 53.95 ± 6.08 | 53.17 ± 31.83 | 1.29 ± 2.26 | 45.54 ± 32.95 | 0.076 ± 0.648 |
| Connect Four | 24 | Q direct | 92.98 ± 6.24 | 57.45 ± 11.30 | 1.04 ± 2.07 | 41.50 ± 12.55 | 0.160 ± 0.238 |
| Connect Four | 24 | Q VNM terminal | 24.86 ± 4.38 | 49.37 ± 17.69 | 0.41 ± 0.63 | 50.23 ± 17.98 | -0.009 ± 0.357 |
| Connect Four | 24 | Q VNM sum | 55.65 ± 8.83 | 50.99 ± 18.65 | 0.78 ± 1.33 | 48.23 ± 19.09 | 0.028 ± 0.377 |
| Connect Four | 24 | Q VNM discounted sum | 82.49 ± 9.41 | 63.46 ± 16.60 | 1.12 ± 2.32 | 35.42 ± 17.76 | 0.280 ± 0.343 |
| 4×4 Reversi | 24 | Original | 100.00 ± 0.00 | 67.35 ± 2.05 | 1.35 ± 0.77 | 31.30 ± 1.97 | 0.361 ± 0.040 |
| 4×4 Reversi | 24 | V direct | 83.91 ± 1.49 | 46.27 ± 5.90 | 5.03 ± 1.56 | 48.70 ± 5.37 | -0.024 ± 0.112 |
| 4×4 Reversi | 24 | V VNM terminal | 82.70 ± 1.60 | 34.78 ± 5.39 | 10.01 ± 2.75 | 55.21 ± 5.09 | -0.204 ± 0.101 |
| 4×4 Reversi | 24 | V VNM sum | 82.70 ± 1.40 | 33.71 ± 3.65 | 7.39 ± 2.79 | 58.90 ± 3.20 | -0.252 ± 0.063 |
| 4×4 Reversi | 24 | V VNM discounted sum | 82.95 ± 1.37 | 35.34 ± 3.87 | 7.24 ± 2.79 | 57.42 ± 3.98 | -0.221 ± 0.073 |
| 4×4 Reversi | 24 | Q direct | 85.86 ± 1.26 | 49.87 ± 4.78 | 5.47 ± 2.26 | 44.66 ± 4.00 | 0.052 ± 0.085 |
| 4×4 Reversi | 24 | Q VNM terminal | 79.89 ± 1.67 | 36.87 ± 5.02 | 9.68 ± 3.33 | 53.45 ± 4.58 | -0.166 ± 0.090 |
| 4×4 Reversi | 24 | Q VNM sum | 83.36 ± 1.44 | 37.81 ± 8.17 | 7.91 ± 3.00 | 54.28 ± 6.41 | -0.165 ± 0.144 |
| 4×4 Reversi | 24 | Q VNM discounted sum | 83.84 ± 1.33 | 42.97 ± 6.93 | 6.88 ± 2.17 | 50.15 ± 5.84 | -0.072 ± 0.126 |

## Direct opponents and changed menus

For head-to-head games the VNM player uses final-move utility and fitted direct continuations, while its actual opponent always plays direct on the same weights. Each seat has equal weight. In the second table a probability error is the fraction of move probability that must be reassigned to reproduce the original answer. Smaller values indicate closer predictions. These are positions offering at least two moves before the favorite was removed. The imitator maximizes the fitted probability of observed moves. Its predictions, both direct frequency rules, and both VNM rules use observed play for training.

| Game | Originals | VNM network | VNM wins (%) | Draws (%) | Direct wins (%) | VNM wins − direct wins, points [95%] | Raw / Holm p | Played VNM wins / draws / direct wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tic-tac-toe | 32 | V | 77.34 ± 16.93 | 17.19 ± 14.46 | 5.47 ± 9.49 | 71.88 ± 23.33 [63.46, 80.29] | 1.43e-17 / 5.74e-17 | Exact game probabilities |
| Tic-tac-toe | 32 | Q | 74.61 ± 21.41 | 18.75 ± 17.10 | 6.64 ± 11.44 | 67.97 ± 29.77 [57.23, 78.70] | 5.17e-14 / 1.55e-13 | Exact game probabilities |
| Connect Four | 24 | V | 97.87 ± 3.27 | 0.39 ± 0.59 | 1.74 ± 3.25 | 96.13 ± 6.49 [93.39, 98.87] | 1.14e-28 / 5.72e-28 | 6013 / 24 / 107 |
| Connect Four | 24 | Q | 97.88 ± 3.19 | 1.73 ± 3.26 | 0.39 ± 0.42 | 97.49 ± 3.18 [96.15, 98.84] | 6.51e-36 / 3.9e-35 | 6014 / 106 / 24 |
| 4×4 Reversi | 24 | V | 44.55 ± 12.90 | 7.32 ± 6.60 | 48.13 ± 11.36 | -3.58 ± 23.39 [-13.46, 6.30] | 0.461 / 0.461 | 2737 / 450 / 2957 |
| 4×4 Reversi | 24 | Q | 41.21 ± 13.50 | 7.55 ± 5.44 | 51.24 ± 12.93 | -10.03 ± 25.87 [-20.95, 0.90] | 0.0702 / 0.14 | 2532 / 464 / 3148 |

For the head-to-head difference the reference is equal win rates for both players. Holm adjustment covers the six head-to-head comparisons. The tic-tac-toe results enumerate possible games from the opening. Each larger-game row contains 6,144 actual games across 24 originals.


| Game | Originals | Rule | Favorite removed, error (%) | Another move removed, error (%) | Either removal, error (%) |
| --- | --- | --- | --- | --- | --- |
| Tic-tac-toe | 32 | imitator | 12.94 ± 1.31 | 12.49 ± 1.84 | 12.71 ± 1.38 |
| Tic-tac-toe | 32 | V direct | 37.45 ± 2.65 | 19.92 ± 2.05 | 28.68 ± 2.11 |
| Tic-tac-toe | 32 | Q direct | 26.07 ± 1.91 | 16.25 ± 1.90 | 21.16 ± 1.65 |
| Tic-tac-toe | 32 | V VNM terminal copies | 51.21 ± 3.28 | 25.21 ± 2.45 | 38.21 ± 2.36 |
| Tic-tac-toe | 32 | Q VNM terminal copies | 54.10 ± 2.88 | 26.31 ± 2.26 | 40.20 ± 2.16 |
| Connect Four | 24 | imitator | 27.46 ± 13.63 | 5.72 ± 6.24 | 16.59 ± 9.28 |
| Connect Four | 24 | V direct | 56.90 ± 9.03 | 21.50 ± 4.51 | 39.20 ± 4.14 |
| Connect Four | 24 | Q direct | 37.13 ± 17.87 | 5.87 ± 5.50 | 21.50 ± 9.31 |
| Connect Four | 24 | V VNM terminal copies | 75.78 ± 6.17 | 72.84 ± 7.25 | 74.31 ± 6.27 |
| Connect Four | 24 | Q VNM terminal copies | 77.49 ± 3.93 | 72.22 ± 4.89 | 74.86 ± 3.72 |
| 4×4 Reversi | 24 | imitator | 11.67 ± 1.67 | 6.25 ± 2.08 | 8.96 ± 1.53 |
| 4×4 Reversi | 24 | V direct | 15.18 ± 1.88 | 10.73 ± 2.09 | 12.95 ± 1.56 |
| 4×4 Reversi | 24 | Q direct | 12.53 ± 1.75 | 8.17 ± 2.18 | 10.35 ± 1.58 |
| 4×4 Reversi | 24 | V VNM terminal copies | 17.50 ± 2.82 | 16.31 ± 3.08 | 16.91 ± 2.58 |
| 4×4 Reversi | 24 | Q VNM terminal copies | 19.14 ± 2.51 | 18.40 ± 3.12 | 18.77 ± 2.55 |

## Where the choices change

For fixed originals and final-move VNM, the first table groups withheld boards by which rules reproduce the original choice. These are raw fractions in the sampled cases, counting each position once. Reversi's main estimate additionally restores the population weights. See the per-fit board records for exact counts and denominators. The second table gives the percentage of independent originals with a higher, equal or lower VNM result and individual 95% binomial intervals on those shares.

| Game | Originals | Network | Both match (%) | Only direct (%) | Only VNM (%) | Neither (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Tic-tac-toe | 32 | V | 57.41 ± 3.26 | 16.54 ± 2.02 | 11.16 ± 1.32 | 14.89 ± 1.99 |
| Tic-tac-toe | 32 | Q | 58.43 ± 2.51 | 19.95 ± 2.11 | 8.66 ± 1.06 | 12.95 ± 1.63 |
| Connect Four | 24 | V | 19.36 ± 7.24 | 57.53 ± 4.82 | 4.45 ± 1.38 | 18.66 ± 4.01 |
| Connect Four | 24 | Q | 23.40 ± 4.69 | 69.58 ± 6.05 | 1.45 ± 1.37 | 5.57 ± 5.13 |
| 4×4 Reversi | 24 | V | 53.88 ± 2.39 | 17.93 ± 2.15 | 15.83 ± 2.09 | 12.36 ± 1.81 |
| 4×4 Reversi | 24 | Q | 51.64 ± 2.20 | 23.60 ± 1.98 | 13.15 ± 1.75 | 11.61 ± 1.84 |

| Game | Originals | Network | Measure | Higher (%) [95%] | Equal (%) [95%] | Lower (%) [95%] |
| --- | --- | --- | --- | --- | --- | --- |
| Tic-tac-toe | 32 | V | Same move (%) | 0.0 [0.00, 10.89] | 0.0 [0.00, 10.89] | 100.0 [89.11, 100.00] |
| Tic-tac-toe | 32 | V | Wins (%) | 81.2 [63.56, 92.79] | 0.0 [0.00, 10.89] | 18.8 [7.21, 36.44] |
| Tic-tac-toe | 32 | V | Points/game | 71.9 [53.25, 86.25] | 0.0 [0.00, 10.89] | 28.1 [13.75, 46.75] |
| Tic-tac-toe | 32 | Q | Same move (%) | 0.0 [0.00, 10.89] | 0.0 [0.00, 10.89] | 100.0 [89.11, 100.00] |
| Tic-tac-toe | 32 | Q | Wins (%) | 25.0 [11.46, 43.40] | 0.0 [0.00, 10.89] | 75.0 [56.60, 88.54] |
| Tic-tac-toe | 32 | Q | Points/game | 15.6 [5.28, 32.79] | 0.0 [0.00, 10.89] | 84.4 [67.21, 94.72] |
| Connect Four | 24 | V | Same move (%) | 0.0 [0.00, 14.25] | 0.0 [0.00, 14.25] | 100.0 [85.75, 100.00] |
| Connect Four | 24 | V | Wins (%) | 87.5 [67.64, 97.34] | 0.0 [0.00, 14.25] | 12.5 [2.66, 32.36] |
| Connect Four | 24 | V | Points/game | 87.5 [67.64, 97.34] | 0.0 [0.00, 14.25] | 12.5 [2.66, 32.36] |
| Connect Four | 24 | Q | Same move (%) | 0.0 [0.00, 14.25] | 0.0 [0.00, 14.25] | 100.0 [85.75, 100.00] |
| Connect Four | 24 | Q | Wins (%) | 20.8 [7.13, 42.15] | 4.2 [0.11, 21.12] | 75.0 [53.29, 90.23] |
| Connect Four | 24 | Q | Points/game | 16.7 [4.74, 37.38] | 4.2 [0.11, 21.12] | 79.2 [57.85, 92.87] |
| 4×4 Reversi | 24 | V | Same move (%) | 20.8 [7.13, 42.15] | 0.0 [0.00, 14.25] | 79.2 [57.85, 92.87] |
| 4×4 Reversi | 24 | V | Wins (%) | 0.0 [0.00, 14.25] | 4.2 [0.11, 21.12] | 95.8 [78.88, 99.89] |
| 4×4 Reversi | 24 | V | Points/game | 8.3 [1.03, 27.00] | 0.0 [0.00, 14.25] | 91.7 [73.00, 98.97] |
| 4×4 Reversi | 24 | Q | Same move (%) | 0.0 [0.00, 14.25] | 0.0 [0.00, 14.25] | 100.0 [85.75, 100.00] |
| 4×4 Reversi | 24 | Q | Wins (%) | 0.0 [0.00, 14.25] | 0.0 [0.00, 14.25] | 100.0 [85.75, 100.00] |
| 4×4 Reversi | 24 | Q | Points/game | 0.0 [0.00, 14.25] | 0.0 [0.00, 14.25] | 100.0 [85.75, 100.00] |

## Outcomes by opponent and seat

Each cell lists direct, final-move VNM, and their paired change in that order. Each value gives the mean ± spread across originals. New-game rates use eight openings per fit, opponent group and seat. The group of untrained networks averages two opponents in the new games and five in tic-tac-toe. The third and fourth opponent groups use the rules in the study report. All adjusted p-values and paired intervals for these exploratory differences are available in the source-level results.

| Game | Originals | Network | Seat | Opponent | Wins (%), direct / VNM / change | Draws (%), direct / VNM / change | Losses (%), direct / VNM / change | Points/game, direct / VNM / change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tic-tac-toe | 32 | V | First | Uniform | 93.23 ± 3.52 / 91.31 ± 3.39 / -1.92 ± 3.89 | 5.88 ± 3.16 / 7.67 ± 2.98 / 1.78 ± 3.74 | 0.88 ± 0.96 / 1.02 ± 0.80 / 0.14 ± 0.78 | 0.924 ± 0.041 / 0.903 ± 0.039 / -0.021 ± 0.042 |
| Tic-tac-toe | 32 | V | First | Untrained networks | 93.91 ± 7.15 / 89.38 ± 7.91 / -4.53 ± 10.50 | 4.53 ± 5.44 / 9.53 ± 7.66 / 5.00 ± 8.98 | 1.56 ± 4.83 / 1.09 ± 2.45 / -0.47 ± 4.98 | 0.923 ± 0.109 / 0.883 ± 0.089 / -0.041 ± 0.138 |
| Tic-tac-toe | 32 | V | First | Win/block/center/corner | 5.47 ± 12.27 / 34.38 ± 31.59 / 28.91 ± 33.06 | 85.94 ± 21.00 / 64.06 ± 30.41 / -21.88 ± 34.05 | 8.59 ± 17.52 / 1.56 ± 6.15 / -7.03 ± 17.08 | -0.031 ± 0.218 / 0.328 ± 0.339 / 0.359 ± 0.401 |
| Tic-tac-toe | 32 | V | First | Optimal | 0.00 ± 0.00 / 0.00 ± 0.00 / 0.00 ± 0.00 | 85.94 ± 20.02 / 85.94 ± 21.94 / 0.00 ± 29.10 | 14.06 ± 20.02 / 14.06 ± 21.94 / 0.00 ± 29.10 | -0.141 ± 0.200 / -0.141 ± 0.219 / 0.000 ± 0.291 |
| Tic-tac-toe | 32 | V | Second | Uniform | 61.77 ± 9.68 / 68.91 ± 3.53 / 7.13 ± 8.85 | 26.77 ± 6.70 / 11.70 ± 2.43 / -15.07 ± 7.66 | 11.46 ± 4.58 / 19.39 ± 4.07 / 7.93 ± 4.73 | 0.503 ± 0.136 / 0.495 ± 0.072 / -0.008 ± 0.119 |
| Tic-tac-toe | 32 | V | Second | Untrained networks | 65.62 ± 17.81 / 69.53 ± 13.87 / 3.91 ± 21.20 | 24.06 ± 13.65 / 10.16 ± 9.29 / -13.91 ± 15.38 | 10.31 ± 11.21 / 20.31 ± 13.07 / 10.00 ± 16.01 | 0.553 ± 0.265 / 0.492 ± 0.253 / -0.061 ± 0.343 |
| Tic-tac-toe | 32 | V | Second | Win/block/center/corner | 0.00 ± 0.00 / 0.00 ± 0.00 / 0.00 ± 0.00 | 37.50 ± 32.38 / 40.62 ± 34.05 / 3.12 ± 32.22 | 62.50 ± 32.38 / 59.38 ± 34.05 / -3.12 ± 32.22 | -0.625 ± 0.324 / -0.594 ± 0.340 / 0.031 ± 0.322 |
| Tic-tac-toe | 32 | V | Second | Optimal | 0.00 ± 0.00 / 0.00 ± 0.00 / 0.00 ± 0.00 | 9.38 ± 18.78 / 9.38 ± 18.78 / 0.00 ± 24.59 | 90.62 ± 18.78 / 90.62 ± 18.78 / 0.00 ± 24.59 | -0.906 ± 0.188 / -0.906 ± 0.188 / 0.000 ± 0.246 |
| Tic-tac-toe | 32 | Q | First | Uniform | 95.32 ± 1.79 / 87.81 ± 2.28 / -7.51 ± 2.50 | 3.93 ± 1.63 / 10.51 ± 1.70 / 6.59 ± 2.28 | 0.75 ± 0.65 / 1.68 ± 0.95 / 0.93 ± 1.11 | 0.946 ± 0.021 / 0.861 ± 0.031 / -0.084 ± 0.031 |
| Tic-tac-toe | 32 | Q | First | Untrained networks | 95.47 ± 5.87 / 87.97 ± 8.41 / -7.50 ± 10.85 | 4.06 ± 5.60 / 11.41 ± 8.54 / 7.34 ± 11.07 | 0.47 ± 2.65 / 0.62 ± 1.68 / 0.16 ± 3.23 | 0.950 ± 0.072 / 0.873 ± 0.086 / -0.077 ± 0.116 |
| Tic-tac-toe | 32 | Q | First | Win/block/center/corner | 8.59 ± 21.64 / 28.91 ± 22.10 / 20.31 ± 28.71 | 85.16 ± 25.29 / 68.75 ± 22.00 / -16.41 ± 31.50 | 6.25 ± 11.00 / 2.34 ± 7.40 / -3.91 ± 12.87 | 0.023 ± 0.232 / 0.266 ± 0.245 / 0.242 ± 0.314 |
| Tic-tac-toe | 32 | Q | First | Optimal | 0.00 ± 0.00 / 0.00 ± 0.00 / 0.00 ± 0.00 | 78.12 ± 29.61 / 81.25 ± 16.80 / 3.12 ± 36.89 | 21.88 ± 29.61 / 18.75 ± 16.80 / -3.12 ± 36.89 | -0.219 ± 0.296 / -0.188 ± 0.168 / 0.031 ± 0.369 |
| Tic-tac-toe | 32 | Q | Second | Uniform | 73.37 ± 6.32 / 61.68 ± 2.82 / -11.69 ± 5.62 | 20.06 ± 5.15 / 12.87 ± 1.90 / -7.19 ± 5.77 | 6.57 ± 2.66 / 25.45 ± 2.88 / 18.88 ± 2.94 | 0.668 ± 0.082 / 0.362 ± 0.054 / -0.306 ± 0.069 |
| Tic-tac-toe | 32 | Q | Second | Untrained networks | 79.53 ± 16.48 / 61.56 ± 15.05 / -17.97 ± 18.27 | 14.69 ± 13.44 / 14.38 ± 11.69 / -0.31 ± 17.13 | 5.78 ± 7.53 / 24.06 ± 13.47 / 18.28 ± 12.42 | 0.738 ± 0.218 / 0.375 ± 0.261 / -0.363 ± 0.261 |
| Tic-tac-toe | 32 | Q | Second | Win/block/center/corner | 0.00 ± 0.00 / 0.00 ± 0.00 / 0.00 ± 0.00 | 56.25 ± 44.90 / 35.94 ± 32.96 / -20.31 ± 55.88 | 43.75 ± 44.90 / 64.06 ± 32.96 / 20.31 ± 55.88 | -0.438 ± 0.449 / -0.641 ± 0.330 / -0.203 ± 0.559 |
| Tic-tac-toe | 32 | Q | Second | Optimal | 0.00 ± 0.00 / 0.00 ± 0.00 / 0.00 ± 0.00 | 7.03 ± 20.31 / 6.25 ± 14.20 / -0.78 ± 25.78 | 92.97 ± 20.31 / 93.75 ± 14.20 / 0.78 ± 25.78 | -0.930 ± 0.203 / -0.938 ± 0.142 / -0.008 ± 0.258 |
| Connect Four | 24 | V | First | Uniform | 66.93 ± 20.75 / 79.82 ± 19.72 / 12.89 ± 17.61 | 0.00 ± 0.00 / 0.13 ± 0.64 / 0.13 ± 0.64 | 33.07 ± 20.75 / 20.05 ± 19.81 / -13.02 ± 17.52 | 0.339 ± 0.415 / 0.598 ± 0.395 / 0.259 ± 0.351 |
| Connect Four | 24 | V | First | Untrained networks | 61.07 ± 28.37 / 75.13 ± 21.18 / 14.06 ± 25.65 | 0.26 ± 1.28 / 0.26 ± 1.28 / 0.00 ± 1.84 | 38.67 ± 28.53 / 24.61 ± 21.22 / -14.06 ± 26.05 | 0.224 ± 0.569 / 0.505 ± 0.424 / 0.281 ± 0.517 |
| Connect Four | 24 | V | First | Positional | 56.64 ± 33.19 / 57.55 ± 29.46 / 0.91 ± 18.69 | 0.00 ± 0.00 / 0.00 ± 0.00 / 0.00 ± 0.00 | 43.36 ± 33.19 / 42.45 ± 29.46 / -0.91 ± 18.69 | 0.133 ± 0.664 / 0.151 ± 0.589 / 0.018 ± 0.374 |
| Connect Four | 24 | V | First | Tactical or optimal | 14.19 ± 19.04 / 15.10 ± 13.78 / 0.91 ± 15.01 | 0.13 ± 0.64 / 0.00 ± 0.00 / -0.13 ± 0.64 | 85.68 ± 19.35 / 84.90 ± 13.78 / -0.78 ± 15.43 | -0.715 ± 0.384 / -0.698 ± 0.276 / 0.017 ± 0.304 |
| Connect Four | 24 | V | Second | Uniform | 58.07 ± 22.72 / 73.57 ± 20.52 / 15.49 ± 14.17 | 0.00 ± 0.00 / 0.52 ± 1.50 / 0.52 ± 1.50 | 41.93 ± 22.72 / 25.91 ± 20.70 / -16.02 ± 14.08 | 0.161 ± 0.454 / 0.477 ± 0.412 / 0.315 ± 0.282 |
| Connect Four | 24 | V | Second | Untrained networks | 50.13 ± 27.03 / 68.10 ± 27.51 / 17.97 ± 20.30 | 0.78 ± 3.83 / 0.26 ± 0.88 / -0.52 ± 3.98 | 49.09 ± 28.03 / 31.64 ± 27.68 / -17.45 ± 21.19 | 0.010 ± 0.549 / 0.365 ± 0.552 / 0.354 ± 0.413 |
| Connect Four | 24 | V | Second | Positional | 37.50 ± 30.16 / 49.48 ± 29.76 / 11.98 ± 23.60 | 0.00 ± 0.00 / 0.00 ± 0.00 / 0.00 ± 0.00 | 62.50 ± 30.16 / 50.52 ± 29.76 / -11.98 ± 23.60 | -0.250 ± 0.603 / -0.010 ± 0.595 / 0.240 ± 0.472 |
| Connect Four | 24 | V | Second | Tactical or optimal | 7.42 ± 13.44 / 11.07 ± 10.94 / 3.65 ± 13.63 | 0.91 ± 3.26 / 0.52 ± 1.76 / -0.39 ± 2.49 | 91.67 ± 14.36 / 88.41 ± 11.60 / -3.26 ± 14.87 | -0.842 ± 0.276 / -0.773 ± 0.225 / 0.069 ± 0.284 |
| Connect Four | 24 | Q | First | Uniform | 87.11 ± 9.09 / 72.53 ± 16.35 / -14.58 ± 12.21 | 0.00 ± 0.00 / 0.13 ± 0.64 / 0.13 ± 0.64 | 12.89 ± 9.09 / 27.34 ± 16.13 / 14.45 ± 12.01 | 0.742 ± 0.182 / 0.452 ± 0.325 / -0.290 ± 0.242 |
| Connect Four | 24 | Q | First | Untrained networks | 77.47 ± 22.70 / 69.53 ± 21.06 / -7.94 ± 22.83 | 0.13 ± 0.64 / 0.00 ± 0.00 / -0.13 ± 0.64 | 22.40 ± 22.81 / 30.47 ± 21.06 / 8.07 ± 22.83 | 0.551 ± 0.455 / 0.391 ± 0.421 / -0.160 ± 0.457 |
| Connect Four | 24 | Q | First | Positional | 79.30 ± 10.50 / 49.87 ± 31.76 / -29.43 ± 27.26 | 0.00 ± 0.00 / 0.00 ± 0.00 / 0.00 ± 0.00 | 20.70 ± 10.50 / 50.13 ± 31.76 / 29.43 ± 27.26 | 0.586 ± 0.210 / -0.003 ± 0.635 / -0.589 ± 0.545 |
| Connect Four | 24 | Q | First | Tactical or optimal | 8.33 ± 16.93 / 18.10 ± 19.63 / 9.77 ± 16.47 | 4.43 ± 12.05 / 0.78 ± 2.65 / -3.65 ± 12.32 | 87.24 ± 22.70 / 81.12 ± 19.75 / -6.12 ± 18.87 | -0.789 ± 0.382 / -0.630 ± 0.393 / 0.159 ± 0.332 |
| Connect Four | 24 | Q | Second | Uniform | 84.51 ± 9.19 / 66.80 ± 18.36 / -17.71 ± 13.25 | 0.00 ± 0.00 / 0.91 ± 1.72 / 0.91 ± 1.72 | 15.49 ± 9.19 / 32.29 ± 18.83 / 16.80 ± 13.41 | 0.690 ± 0.184 / 0.345 ± 0.372 / -0.345 ± 0.266 |
| Connect Four | 24 | Q | Second | Untrained networks | 71.22 ± 23.26 / 59.24 ± 27.34 / -11.98 ± 21.15 | 0.13 ± 0.64 / 0.00 ± 0.00 / -0.13 ± 0.64 | 28.65 ± 23.07 / 40.76 ± 27.34 / 12.11 ± 21.24 | 0.426 ± 0.463 / 0.185 ± 0.547 / -0.241 ± 0.424 |
| Connect Four | 24 | Q | Second | Positional | 47.66 ± 28.67 / 44.79 ± 26.69 / -2.86 ± 25.20 | 0.13 ± 0.64 / 0.13 ± 0.64 / 0.00 ± 0.92 | 52.21 ± 28.72 / 55.08 ± 26.70 / 2.86 ± 25.15 | -0.046 ± 0.574 / -0.103 ± 0.534 / -0.057 ± 0.503 |
| Connect Four | 24 | Q | Second | Tactical or optimal | 4.04 ± 8.62 / 14.06 ± 15.72 / 10.03 ± 15.20 | 3.52 ± 10.68 / 1.30 ± 3.31 / -2.21 ± 10.53 | 92.45 ± 17.94 / 84.64 ± 16.46 / -7.81 ± 18.18 | -0.884 ± 0.260 / -0.706 ± 0.320 / 0.178 ± 0.318 |
| 4×4 Reversi | 24 | V | First | Uniform | 46.74 ± 9.14 / 48.05 ± 11.05 / 1.30 ± 11.83 | 6.38 ± 3.74 / 8.59 ± 5.85 / 2.21 ± 7.46 | 46.88 ± 9.22 / 43.36 ± 9.32 / -3.52 ± 9.50 | -0.001 ± 0.180 / 0.047 ± 0.196 / 0.048 ± 0.201 |
| 4×4 Reversi | 24 | V | First | Untrained networks | 50.91 ± 16.68 / 47.66 ± 13.36 / -3.26 ± 16.45 | 5.99 ± 8.13 / 5.86 ± 6.34 / -0.13 ± 7.57 | 43.10 ± 18.31 / 46.48 ± 14.29 / 3.39 ± 17.34 | 0.078 ± 0.341 / 0.012 ± 0.269 / -0.066 ± 0.329 |
| 4×4 Reversi | 24 | V | First | Positional | 12.24 ± 4.60 / 19.01 ± 12.19 / 6.77 ± 12.14 | 2.73 ± 2.32 / 2.86 ± 2.42 / 0.13 ± 2.98 | 85.03 ± 4.70 / 78.12 ± 12.74 / -6.90 ± 12.01 | -0.728 ± 0.090 / -0.591 ± 0.248 / 0.137 ± 0.240 |
| 4×4 Reversi | 24 | V | First | Tactical or optimal | 12.24 ± 6.51 / 8.33 ± 4.09 / -3.91 ± 5.39 | 1.04 ± 1.76 / 0.91 ± 1.45 / -0.13 ± 1.95 | 86.72 ± 6.27 / 90.76 ± 4.65 / 4.04 ± 5.25 | -0.745 ± 0.127 / -0.824 ± 0.086 / -0.079 ± 0.105 |
| 4×4 Reversi | 24 | V | Second | Uniform | 67.71 ± 8.43 / 56.90 ± 10.05 / -10.81 ± 13.10 | 6.38 ± 3.74 / 13.15 ± 6.89 / 6.77 ± 7.52 | 25.91 ± 8.47 / 29.95 ± 9.44 / 4.04 ± 11.85 | 0.418 ± 0.165 / 0.270 ± 0.182 / -0.148 ± 0.238 |
| 4×4 Reversi | 24 | V | Second | Untrained networks | 62.63 ± 14.23 / 55.73 ± 17.93 / -6.90 ± 20.02 | 9.24 ± 8.16 / 13.93 ± 10.14 / 4.69 ± 8.03 | 28.12 ± 13.16 / 30.34 ± 15.52 / 2.21 ± 18.37 | 0.345 ± 0.262 / 0.254 ± 0.320 / -0.091 ± 0.376 |
| 4×4 Reversi | 24 | V | Second | Positional | 68.36 ± 13.14 / 25.65 ± 15.80 / -42.71 ± 18.38 | 7.16 ± 8.22 / 33.20 ± 16.25 / 26.04 ± 15.30 | 24.48 ± 11.90 / 41.15 ± 18.14 / 16.67 ± 18.14 | 0.439 ± 0.237 / -0.155 ± 0.299 / -0.594 ± 0.332 |
| 4×4 Reversi | 24 | V | Second | Tactical or optimal | 49.35 ± 21.96 / 16.93 ± 11.09 / -32.42 ± 21.09 | 1.30 ± 1.82 / 1.56 ± 2.44 / 0.26 ± 1.82 | 49.35 ± 22.40 / 81.51 ± 11.62 / 32.16 ± 21.16 | 0.000 ± 0.443 / -0.646 ± 0.226 / -0.646 ± 0.422 |
| 4×4 Reversi | 24 | Q | First | Uniform | 50.13 ± 11.85 / 47.40 ± 8.68 / -2.73 ± 10.27 | 7.94 ± 4.12 / 6.77 ± 5.43 / -1.17 ± 5.96 | 41.93 ± 10.66 / 45.83 ± 8.63 / 3.91 ± 10.23 | 0.082 ± 0.222 / 0.016 ± 0.164 / -0.066 ± 0.196 |
| 4×4 Reversi | 24 | Q | First | Untrained networks | 53.65 ± 21.19 / 50.13 ± 15.07 / -3.52 ± 16.36 | 7.68 ± 9.79 / 5.60 ± 6.58 / -2.08 ± 11.90 | 38.67 ± 21.01 / 44.27 ± 16.24 / 5.60 ± 18.10 | 0.150 ± 0.410 / 0.059 ± 0.306 / -0.091 ± 0.324 |
| 4×4 Reversi | 24 | Q | First | Positional | 12.11 ± 5.16 / 19.27 ± 13.84 / 7.16 ± 14.05 | 2.34 ± 2.30 / 2.73 ± 2.49 / 0.39 ± 3.61 | 85.55 ± 4.20 / 77.99 ± 14.76 / -7.55 ± 14.60 | -0.734 ± 0.091 / -0.587 ± 0.285 / 0.147 ± 0.284 |
| 4×4 Reversi | 24 | Q | First | Tactical or optimal | 13.02 ± 6.23 / 8.98 ± 6.67 / -4.04 ± 6.28 | 1.30 ± 2.04 / 0.91 ± 1.72 / -0.39 ± 2.66 | 85.68 ± 5.67 / 90.10 ± 6.09 / 4.43 ± 5.82 | -0.727 ± 0.117 / -0.811 ± 0.127 / -0.085 ± 0.118 |
| 4×4 Reversi | 24 | Q | Second | Uniform | 68.88 ± 10.45 / 58.85 ± 9.20 / -10.03 ± 15.39 | 7.94 ± 5.45 / 15.10 ± 5.95 / 7.16 ± 7.17 | 23.18 ± 9.92 / 26.04 ± 7.64 / 2.86 ± 13.70 | 0.457 ± 0.196 / 0.328 ± 0.158 / -0.129 ± 0.282 |
| 4×4 Reversi | 24 | Q | Second | Untrained networks | 61.46 ± 13.87 / 62.63 ± 14.47 / 1.17 ± 19.80 | 9.77 ± 10.76 / 10.94 ± 7.87 / 1.17 ± 12.36 | 28.78 ± 12.36 / 26.43 ± 10.59 / -2.34 ± 15.29 | 0.327 ± 0.240 / 0.362 ± 0.241 / 0.035 ± 0.332 |
| 4×4 Reversi | 24 | Q | Second | Positional | 74.87 ± 9.37 / 28.78 ± 14.45 / -46.09 ± 15.37 | 5.47 ± 7.67 / 33.33 ± 16.55 / 27.86 ± 17.51 | 19.66 ± 6.55 / 37.89 ± 16.36 / 18.23 ± 15.02 | 0.552 ± 0.142 / -0.091 ± 0.261 / -0.643 ± 0.248 |
| 4×4 Reversi | 24 | Q | Second | Tactical or optimal | 64.84 ± 14.78 / 18.88 ± 11.82 / -45.96 ± 15.57 | 1.30 ± 2.24 / 2.08 ± 2.87 / 0.78 ± 3.60 | 33.85 ± 14.79 / 79.04 ± 12.31 / 45.18 ± 14.86 | 0.310 ± 0.295 / -0.602 ± 0.240 / -0.911 ± 0.302 |

## Fixed-player controls against direct Q

Each entry gives the absolute mean ± spread. The paired differences below subtract Q direct on the identical cases. A move imitator learns the probability of observed actions. A cell frequency pools choices by action, a revealed ranking uses extra restricted-menu queries, and distillation fits the original numerical scores. Game-points and outcome-frequency networks learn from observed endings. Supplied-payoff VNM gives complete games their actual points, and fitted-payoff VNM learns the value of a draw. Both use two fitted direct copies wherever labeled. The source-continuation rows consult the original at future turns. A dash denotes an unavailable measurement. All intervals and p-values for these fixed-control contrasts are in controls.jsonl.gz; Holm adjustment covers this table's paired control contrasts.

| Game | Originals | Method and training/deployment access | Agreement (%) | Wins (%) | Draws (%) | Losses (%) | Points/game | Error after removing favorite (%) | Agreement change [95%] | Wins change [95%] | Removal error change [95%] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tic-tac-toe | 32 | Q direct, visits | 78.39 ± 2.14 | 44.04 ± 2.90 | 33.66 ± 7.89 | 22.30 ± 7.96 | 0.217 ± 0.090 | 26.07 ± 1.91 | — | — | — |
| Tic-tac-toe | 32 | Move imitator, visits | 82.59 ± 2.14 | 42.59 ± 2.91 | 35.71 ± 8.58 | 21.71 ± 9.05 | 0.209 ± 0.103 | 12.94 ± 1.31 | 4.21 ± 1.80 [3.56, 4.86] | -1.45 ± 2.55 [-2.37, -0.53] | -13.13 ± 1.23 [-13.57, -12.69] |
| Tic-tac-toe | 32 | Pooled cell frequency, visits | 52.98 ± 4.78 | 27.92 ± 5.82 | 13.77 ± 13.59 | 58.31 ± 12.61 | -0.304 ± 0.142 | 33.81 ± 4.91 | -25.41 ± 3.63 [-26.72, -24.10] | -16.11 ± 5.47 [-18.09, -14.14] | 7.75 ± 4.10 [6.27, 9.22] |
| Tic-tac-toe | 32 | Game-points network, observed endings | 55.66 ± 2.98 | 38.80 ± 2.47 | 21.34 ± 7.50 | 39.87 ± 7.17 | -0.011 ± 0.077 | 47.06 ± 2.62 | -22.72 ± 1.76 [-23.36, -22.09] | -5.24 ± 3.40 [-6.47, -4.01] | 21.00 ± 2.13 [20.23, 21.77] |
| Tic-tac-toe | 32 | Outcome-frequency network, observed endings | 57.38 ± 2.85 | 38.75 ± 2.51 | 25.54 ± 7.51 | 35.72 ± 7.23 | 0.030 ± 0.078 | 47.05 ± 2.64 | -21.01 ± 1.60 [-21.59, -20.43] | -5.29 ± 3.23 [-6.46, -4.12] | 20.98 ± 2.15 [20.21, 21.75] |
| Tic-tac-toe | 32 | Supplied-payoff VNM, fitted Q copies | 68.39 ± 7.01 | 44.57 ± 5.02 | 37.41 ± 6.31 | 18.03 ± 5.43 | 0.265 ± 0.083 | 47.94 ± 6.01 | -9.99 ± 7.41 [-12.67, -7.32] | 0.53 ± 4.95 [-1.26, 2.32] | 21.88 ± 6.17 [19.65, 24.10] |
| Tic-tac-toe | 32 | Fitted-payoff VNM, fitted Q copies | 68.39 ± 7.01 | 44.57 ± 5.02 | 37.41 ± 6.31 | 18.03 ± 5.43 | 0.265 ± 0.083 | 47.94 ± 6.01 | -9.99 ± 7.41 [-12.67, -7.32] | 0.53 ± 4.95 [-1.26, 2.32] | 21.88 ± 6.17 [19.65, 24.10] |
| Tic-tac-toe | 32 | Fitted-payoff VNM, original continuations | 71.09 ± 7.63 | 44.94 ± 2.15 | 55.06 ± 2.15 | 0.00 ± 0.00 | 0.449 ± 0.022 | 47.55 ± 6.24 | -7.29 ± 8.08 [-10.21, -4.38] | 0.90 ± 3.26 [-0.28, 2.08] | 21.49 ± 6.35 [19.20, 23.78] |
| Connect Four | 24 | Q direct, visits | 92.98 ± 6.24 | 57.45 ± 11.30 | 1.04 ± 2.07 | 41.50 ± 12.55 | 0.160 ± 0.238 | 37.13 ± 17.87 | — | — | — |
| Connect Four | 24 | Move imitator, visits | 92.76 ± 6.98 | 55.11 ± 10.41 | 0.91 ± 2.11 | 43.98 ± 12.01 | 0.111 ± 0.224 | 27.46 ± 13.63 | -0.22 ± 1.98 [-1.06, 0.61] | -2.34 ± 7.11 [-5.35, 0.66] | -9.67 ± 17.94 [-17.25, -2.09] |
| Connect Four | 24 | Pooled cell frequency, visits | 56.21 ± 29.88 | 49.50 ± 4.13 | 0.02 ± 0.08 | 50.49 ± 4.14 | -0.010 ± 0.083 | 38.44 ± 24.25 | -36.77 ± 31.21 [-49.95, -23.59] | -7.96 ± 10.93 [-12.57, -3.35] | 1.31 ± 26.18 [-9.75, 12.36] |
| Connect Four | 24 | Revealed order, extra menu queries | 78.35 ± 5.63 | 46.63 ± 13.65 | 0.37 ± 0.80 | 52.99 ± 14.08 | -0.064 ± 0.277 | 31.66 ± 12.94 | -14.64 ± 4.49 [-16.53, -12.74] | -10.82 ± 6.85 [-13.71, -7.93] | -5.47 ± 17.40 [-12.81, 1.88] |
| Connect Four | 24 | Distillation, original scores during fitting | 92.31 ± 6.32 | 59.15 ± 12.92 | 0.57 ± 1.08 | 40.28 ± 13.56 | 0.189 ± 0.265 | 53.69 ± 18.08 | -0.67 ± 1.68 [-1.38, 0.04] | 1.69 ± 6.74 [-1.16, 4.54] | 16.56 ± 17.64 [9.12, 24.01] |
| Connect Four | 24 | Game-points network, observed endings | 16.62 ± 5.05 | 50.90 ± 6.68 | 0.02 ± 0.08 | 49.09 ± 6.71 | 0.018 ± 0.134 | 84.29 ± 3.78 | -76.36 ± 10.66 [-80.86, -71.86] | -6.56 ± 15.58 [-13.14, 0.02] | 47.16 ± 18.42 [39.38, 54.94] |
| Connect Four | 24 | Outcome-frequency network, observed endings | 15.23 ± 5.31 | 47.74 ± 6.52 | 0.00 ± 0.00 | 52.26 ± 6.52 | -0.045 ± 0.130 | 84.54 ± 3.86 | -77.75 ± 10.97 [-82.39, -73.12] | -9.72 ± 15.96 [-16.46, -2.98] | 47.41 ± 18.95 [39.41, 55.41] |
| Connect Four | 24 | Supplied-payoff VNM, fitted Q copies | 19.02 ± 12.92 | 58.25 ± 11.79 | 0.16 ± 0.50 | 41.59 ± 11.89 | 0.167 ± 0.237 | 81.10 ± 13.42 | -73.96 ± 15.73 [-80.61, -67.32] | 0.80 ± 9.15 [-3.07, 4.66] | 43.97 ± 22.61 [34.42, 53.51] |
| Connect Four | 24 | Fitted-payoff VNM, fitted Q copies | 19.04 ± 13.17 | 58.24 ± 11.65 | 0.24 ± 0.57 | 41.52 ± 11.87 | 0.167 ± 0.235 | 81.17 ± 13.46 | -73.94 ± 15.93 [-80.66, -67.21] | 0.78 ± 9.22 [-3.11, 4.68] | 44.04 ± 22.58 [34.50, 53.57] |
| Connect Four | 24 | Fitted-payoff VNM, original continuations | 19.47 ± 13.53 | 61.10 ± 13.31 | 0.20 ± 0.72 | 38.70 ± 13.52 | 0.224 ± 0.268 | 81.40 ± 13.53 | -73.52 ± 16.31 [-80.40, -66.63] | 3.65 ± 9.63 [-0.42, 7.71] | 44.27 ± 22.07 [34.95, 53.59] |
| 4×4 Reversi | 24 | Q direct, visits | 85.86 ± 1.26 | 49.87 ± 4.78 | 5.47 ± 2.26 | 44.66 ± 4.00 | 0.052 ± 0.085 | 12.53 ± 1.75 | — | — | — |
| 4×4 Reversi | 24 | Move imitator, visits | 87.85 ± 1.17 | 51.51 ± 4.31 | 4.25 ± 1.40 | 44.24 ± 4.24 | 0.073 ± 0.084 | 11.67 ± 1.67 | 1.99 ± 0.87 [1.62, 2.35] | 1.64 ± 3.49 [0.17, 3.12] | -0.86 ± 1.48 [-1.49, -0.24] |
| 4×4 Reversi | 24 | Pooled cell frequency, visits | 78.21 ± 2.36 | 37.79 ± 7.66 | 5.84 ± 2.28 | 56.36 ± 7.61 | -0.186 ± 0.151 | 18.98 ± 3.11 | -7.65 ± 2.55 [-8.73, -6.57] | -12.08 ± 8.79 [-15.79, -8.37] | 6.45 ± 2.35 [5.45, 7.44] |
| 4×4 Reversi | 24 | Revealed order, extra menu queries | 81.74 ± 1.33 | 41.57 ± 7.50 | 6.54 ± 2.15 | 51.89 ± 6.97 | -0.103 ± 0.143 | 15.12 ± 2.30 | -4.13 ± 1.44 [-4.73, -3.52] | -8.30 ± 7.21 [-11.34, -5.26] | 2.59 ± 1.90 [1.79, 3.39] |
| 4×4 Reversi | 24 | Distillation, original scores during fitting | 86.83 ± 1.30 | 50.33 ± 4.48 | 5.44 ± 2.14 | 44.24 ± 4.53 | 0.061 ± 0.088 | 11.21 ± 1.57 | 0.97 ± 1.13 [0.49, 1.44] | 0.46 ± 5.01 [-1.66, 2.57] | -1.33 ± 1.51 [-1.96, -0.69] |
| 4×4 Reversi | 24 | Game-points network, observed endings | 81.56 ± 1.40 | 55.94 ± 6.92 | 3.48 ± 1.51 | 40.58 ± 6.75 | 0.154 ± 0.136 | 18.33 ± 3.26 | -4.31 ± 1.85 [-5.09, -3.52] | 6.07 ± 8.55 [2.46, 9.68] | 5.79 ± 2.53 [4.72, 6.86] |
| 4×4 Reversi | 24 | Outcome-frequency network, observed endings | 81.80 ± 1.41 | 54.61 ± 6.37 | 3.50 ± 1.59 | 41.89 ± 5.79 | 0.127 ± 0.121 | 18.38 ± 2.85 | -4.06 ± 1.95 [-4.89, -3.24] | 4.74 ± 7.36 [1.63, 7.84] | 5.85 ± 2.29 [4.88, 6.82] |
| 4×4 Reversi | 24 | Supplied-payoff VNM, fitted Q copies | 96.76 ± 0.59 | 59.70 ± 3.69 | 2.70 ± 1.48 | 37.60 ± 3.80 | 0.221 ± 0.073 | 12.79 ± 1.87 | 10.90 ± 1.25 [10.37, 11.43] | 9.83 ± 5.97 [7.31, 12.35] | 0.26 ± 1.66 [-0.44, 0.96] |
| 4×4 Reversi | 24 | Fitted-payoff VNM, fitted Q copies | 96.22 ± 0.55 | 57.65 ± 3.38 | 4.28 ± 1.83 | 38.07 ± 3.61 | 0.196 ± 0.067 | 12.27 ± 1.87 | 10.35 ± 1.19 [9.85, 10.86] | 7.78 ± 5.94 [5.27, 10.29] | -0.26 ± 1.78 [-1.01, 0.49] |
| 4×4 Reversi | 24 | Fitted-payoff VNM, original continuations | 100.00 ± 0.00 | 67.35 ± 2.05 | 1.35 ± 0.77 | 31.30 ± 1.97 | 0.361 ± 0.040 | 12.04 ± 1.99 | 14.14 ± 1.26 [13.61, 14.67] | 17.48 ± 4.62 [15.53, 19.43] | -0.49 ± 1.82 [-1.26, 0.28] |

## All common deterministic choices and randomized originals

A randomized original takes its favorite with probability ¾ and otherwise samples uniformly. Independent-draw agreement is the probability that one draw from each player gives the same move. For a positive temperature T, each legal move gets sampling weight exp(score/T). At T = 0 exact best moves share probability equally. Here each method’s temperature for randomized originals is selected by its own validation boards. The direct comparator uses its own validation setting. Copies finish with fitted direct play. Original continuations consult the frozen player at subsequent turns. Each cell is VNM minus direct, mean ± spread with individual 95% interval, raw p and adjusted p. Adjustments use the separate exploratory family except where the fixed-choice comparison was designated primary. Differences for every individual temperature, game average, position type, seat and opponent are machine-readable.

### Fixed-choice originals


| Game | Originals | Network | VNM whole-game score | Both seats continue with | Same move (%) | Wins (%) | Points/game |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tic-tac-toe | 32 | V | Final move | Fitted copies | -5.39 ± 2.54 [-6.30, -4.47], 3.36e-13 / 5.72e-12 | 4.19 ± 5.14 [2.33, 6.04], 6.58e-05 / 0.000461 | 0.033 ± 0.086 [0.002, 0.063], 0.0395 / 0.118 |
| Tic-tac-toe | 32 | V | Final move | Original | -3.77 ± 2.73 [-4.76, -2.79], 8.11e-09 / 0.000148 | 1.28 ± 4.56 [-0.36, 2.92], 0.122 / 1 | 0.047 ± 0.100 [0.011, 0.083], 0.0121 / 1 |
| Tic-tac-toe | 32 | V | Sum | Fitted copies | 1.43 ± 2.36 [0.58, 2.28], 0.00172 / 1 | 5.93 ± 5.09 [4.09, 7.76], 2.29e-07 / 0.00394 | 0.160 ± 0.090 [0.128, 0.193], 2.48e-11 / 5.02e-07 |
| Tic-tac-toe | 32 | V | Sum | Original | 2.67 ± 2.42 [1.80, 3.55], 6.23e-07 / 0.0105 | 4.09 ± 3.89 [2.69, 5.50], 1.38e-06 / 0.0228 | 0.247 ± 0.089 [0.215, 0.279], 2.34e-16 / 5.3e-12 |
| Tic-tac-toe | 32 | V | Discounted sum | Fitted copies | 2.52 ± 2.21 [1.73, 3.32], 3.33e-07 / 0.00567 | 6.11 ± 4.69 [4.42, 7.80], 2.73e-08 / 0.000488 | 0.172 ± 0.086 [0.141, 0.203], 1.57e-12 / 3.28e-08 |
| Tic-tac-toe | 32 | V | Discounted sum | Original | 3.67 ± 2.30 [2.84, 4.49], 3.47e-10 / 6.68e-06 | 4.12 ± 3.29 [2.93, 5.31], 5.98e-08 / 0.00105 | 0.255 ± 0.085 [0.224, 0.285], 3.12e-17 / 7.2e-13 |
| Tic-tac-toe | 32 | Q | Final move | Fitted copies | -11.29 ± 2.72 [-12.27, -10.31], 2.78e-21 / 5.56e-20 | -3.04 ± 3.62 [-4.35, -1.74], 4.29e-05 / 0.000343 | -0.096 ± 0.105 [-0.134, -0.058], 1.42e-05 / 0.000142 |
| Tic-tac-toe | 32 | Q | Final move | Original | -9.62 ± 2.68 [-10.59, -8.66], 1.76e-19 / 4.2e-15 | -3.43 ± 3.45 [-4.68, -2.19], 3.6e-06 / 0.0581 | -0.054 ± 0.094 [-0.087, -0.020], 0.00293 / 1 |
| Tic-tac-toe | 32 | Q | Sum | Fitted copies | 1.31 ± 1.49 [0.77, 1.85], 2.38e-05 / 0.367 | 0.05 ± 3.50 [-1.21, 1.32], 0.931 / 1 | 0.018 ± 0.095 [-0.017, 0.052], 0.3 / 1 |
| Tic-tac-toe | 32 | Q | Sum | Original | 2.16 ± 1.47 [1.63, 2.69], 2.13e-09 / 3.98e-05 | 0.51 ± 3.88 [-0.89, 1.90], 0.465 / 1 | 0.079 ± 0.081 [0.050, 0.108], 5.18e-06 / 0.0829 |
| Tic-tac-toe | 32 | Q | Discounted sum | Fitted copies | 2.67 ± 1.22 [2.23, 3.11], 1.57e-13 / 3.35e-09 | 0.70 ± 3.84 [-0.68, 2.09], 0.308 / 1 | 0.030 ± 0.069 [0.005, 0.055], 0.0185 / 1 |
| Tic-tac-toe | 32 | Q | Discounted sum | Original | 3.34 ± 1.21 [2.90, 3.77], 3.53e-16 / 7.98e-12 | 0.18 ± 3.69 [-1.15, 1.51], 0.787 / 1 | 0.067 ± 0.066 [0.044, 0.091], 2.14e-06 / 0.035 |
| Connect Four | 24 | V | Final move | Fitted copies | -53.08 ± 5.11 [-55.24, -50.92], 3.89e-25 / 8.57e-24 | 9.73 ± 8.63 [6.09, 13.38], 1.28e-05 / 0.000141 | 0.194 ± 0.174 [0.121, 0.268], 1.48e-05 / 0.000142 |
| Connect Four | 24 | V | Final move | Original | -51.00 ± 5.06 [-53.14, -48.87], 7.57e-25 / 1.92e-20 | 9.80 ± 6.12 [7.21, 12.38], 6e-08 / 0.00106 | 0.198 ± 0.121 [0.148, 0.249], 3.73e-08 / 0.000662 |
| Connect Four | 24 | V | Sum | Fitted copies | -35.82 ± 5.45 [-38.12, -33.52], 1.25e-20 / 3.04e-16 | 6.43 ± 10.78 [1.88, 10.98], 0.00766 / 1 | 0.143 ± 0.220 [0.050, 0.236], 0.00412 / 1 |
| Connect Four | 24 | V | Sum | Original | -39.24 ± 5.68 [-41.63, -36.84], 3.93e-21 / 9.65e-17 | 2.07 ± 13.57 [-3.66, 7.80], 0.463 / 1 | 0.056 ± 0.282 [-0.063, 0.176], 0.339 / 1 |
| Connect Four | 24 | V | Discounted sum | Fitted copies | -22.94 ± 6.43 [-25.66, -20.23], 8.82e-15 / 1.94e-10 | 9.18 ± 14.29 [3.15, 15.21], 0.00452 / 1 | 0.194 ± 0.293 [0.070, 0.318], 0.00363 / 1 |
| Connect Four | 24 | V | Discounted sum | Original | -28.59 ± 5.55 [-30.93, -26.24], 2.87e-18 / 6.71e-14 | 8.85 ± 16.12 [2.05, 15.66], 0.013 / 1 | 0.185 ± 0.329 [0.046, 0.323], 0.0114 / 1 |
| Connect Four | 24 | Q | Final move | Fitted copies | -68.12 ± 6.95 [-71.06, -65.19], 1.42e-24 / 2.98e-23 | -8.09 ± 9.55 [-12.12, -4.05], 0.00039 / 0.00195 | -0.168 ± 0.187 [-0.247, -0.089], 0.000207 / 0.00124 |
| Connect Four | 24 | Q | Final move | Original | -67.00 ± 7.80 [-70.29, -63.71], 2.85e-23 / 7.16e-19 | -4.80 ± 12.39 [-10.03, 0.43], 0.0702 / 1 | -0.103 ± 0.241 [-0.204, -0.001], 0.0477 / 1 |
| Connect Four | 24 | Q | Sum | Fitted copies | -37.33 ± 7.11 [-40.33, -34.33], 1.88e-18 / 4.43e-14 | -6.46 ± 10.13 [-10.74, -2.19], 0.00474 / 1 | -0.132 ± 0.200 [-0.216, -0.047], 0.00378 / 1 |
| Connect Four | 24 | Q | Sum | Original | -38.01 ± 6.67 [-40.83, -35.19], 3.08e-19 / 7.32e-15 | -6.62 ± 11.78 [-11.60, -1.65], 0.0113 / 1 | -0.130 ± 0.235 [-0.229, -0.031], 0.0126 / 1 |
| Connect Four | 24 | Q | Discounted sum | Fitted copies | -10.49 ± 3.86 [-12.12, -8.86], 2.66e-12 / 5.53e-08 | 6.01 ± 7.57 [2.81, 9.20], 0.000745 / 1 | 0.121 ± 0.152 [0.057, 0.185], 0.000742 / 1 |
| Connect Four | 24 | Q | Discounted sum | Original | -11.71 ± 4.71 [-13.70, -9.72], 1.64e-11 / 3.34e-07 | 5.18 ± 7.45 [2.03, 8.32], 0.00243 / 1 | 0.109 ± 0.155 [0.043, 0.174], 0.00232 / 1 |
| 4×4 Reversi | 24 | V | Final move | Fitted copies | -1.20 ± 2.14 [-2.11, -0.30], 0.0115 / 0.0461 | -11.49 ± 6.88 [-14.40, -8.59], 2.89e-08 / 3.76e-07 | -0.180 ± 0.129 [-0.235, -0.125], 5.89e-07 / 7.07e-06 |
| 4×4 Reversi | 24 | V | Final move | Original | -0.46 ± 1.86 [-1.24, 0.33], 0.241 / 1 | -11.52 ± 6.68 [-14.34, -8.70], 1.66e-08 / 0.000298 | -0.155 ± 0.120 [-0.205, -0.104], 1.88e-06 / 0.0309 |
| 4×4 Reversi | 24 | V | Sum | Fitted copies | -1.20 ± 1.50 [-1.84, -0.57], 0.000658 / 1 | -12.57 ± 5.41 [-14.85, -10.28], 6.31e-11 / 1.26e-06 | -0.228 ± 0.109 [-0.274, -0.182], 4.75e-10 / 9.09e-06 |
| 4×4 Reversi | 24 | V | Sum | Original | -0.89 ± 1.36 [-1.47, -0.32], 0.00391 / 1 | -13.69 ± 7.02 [-16.65, -10.72], 1.79e-09 / 3.35e-05 | -0.242 ± 0.134 [-0.298, -0.185], 8.03e-09 / 0.000146 |
| 4×4 Reversi | 24 | V | Discounted sum | Fitted copies | -0.96 ± 1.49 [-1.58, -0.33], 0.00452 / 1 | -10.94 ± 5.57 [-13.29, -8.59], 1.57e-09 / 2.94e-05 | -0.197 ± 0.112 [-0.244, -0.150], 1.13e-08 / 0.000205 |
| 4×4 Reversi | 24 | V | Discounted sum | Original | -0.78 ± 1.33 [-1.34, -0.22], 0.00883 / 1 | -10.99 ± 6.57 [-13.76, -8.21], 2.86e-08 / 0.00051 | -0.190 ± 0.122 [-0.242, -0.138], 1.01e-07 / 0.00176 |
| 4×4 Reversi | 24 | Q | Final move | Fitted copies | -5.97 ± 1.92 [-6.78, -5.16], 1.61e-13 / 2.89e-12 | -13.00 ± 6.43 [-15.72, -10.29], 9.16e-10 / 1.28e-08 | -0.218 ± 0.101 [-0.261, -0.175], 2.71e-10 / 4.07e-09 |
| 4×4 Reversi | 24 | Q | Final move | Original | -5.08 ± 1.86 [-5.87, -4.30], 2.44e-12 / 5.07e-08 | -11.70 ± 5.90 [-14.19, -9.21], 1.32e-09 / 2.49e-05 | -0.163 ± 0.096 [-0.204, -0.122], 2.32e-08 / 0.000415 |
| 4×4 Reversi | 24 | Q | Sum | Fitted copies | -2.50 ± 1.52 [-3.15, -1.86], 3.76e-08 / 0.000668 | -12.06 ± 6.81 [-14.94, -9.18], 1.05e-08 / 0.00019 | -0.217 ± 0.116 [-0.266, -0.168], 3.7e-09 / 6.84e-05 |
| 4×4 Reversi | 24 | Q | Sum | Original | -2.19 ± 1.48 [-2.81, -1.56], 2.2e-07 / 0.00378 | -11.67 ± 7.34 [-14.77, -8.57], 6.71e-08 / 0.00118 | -0.195 ± 0.129 [-0.249, -0.141], 1.49e-07 / 0.00258 |
| 4×4 Reversi | 24 | Q | Discounted sum | Fitted copies | -2.02 ± 1.35 [-2.59, -1.45], 1.76e-07 / 0.00304 | -6.90 ± 6.32 [-9.57, -4.23], 1.97e-05 / 0.305 | -0.124 ± 0.110 [-0.170, -0.078], 1.24e-05 / 0.195 |
| 4×4 Reversi | 24 | Q | Discounted sum | Original | -1.72 ± 1.34 [-2.29, -1.15], 2.11e-06 / 0.0345 | -6.79 ± 6.34 [-9.47, -4.11], 2.57e-05 / 0.397 | -0.104 ± 0.110 [-0.150, -0.057], 0.000123 / 1 |
| Three games, equal weight | 32/24/24 | V | Final move | Fitted copies | -19.89 ± 2.03 [-20.71, -19.07], 1.55e-37 / 3.71e-36 | 0.81 ± 4.06 [-0.81, 2.43], 0.322 / 0.643 | 0.016 ± 0.078 [-0.016, 0.047], 0.322 / 0.643 |
| Three games, equal weight | 32/24/24 | V | Final move | Original | -18.41 ± 2.01 [-19.22, -17.60], 3.11e-36 / 8.52e-32 | -0.15 ± 3.38 [-1.49, 1.20], 0.826 / 1 | 0.030 ± 0.066 [0.004, 0.056], 0.0226 / 1 |
| Three games, equal weight | 32/24/24 | V | Sum | Fitted copies | -11.86 ± 2.04 [-12.70, -11.03], 3.18e-25 / 8.11e-21 | -0.07 ± 4.36 [-1.83, 1.69], 0.937 / 1 | 0.025 ± 0.087 [-0.010, 0.061], 0.157 / 1 |
| Three games, equal weight | 32/24/24 | V | Sum | Original | -12.49 ± 2.11 [-13.35, -11.63], 5.95e-25 / 1.52e-20 | -2.51 ± 5.25 [-4.66, -0.35], 0.0237 / 1 | 0.021 ± 0.108 [-0.024, 0.065], 0.353 / 1 |
| Three games, equal weight | 32/24/24 | V | Discounted sum | Fitted copies | -7.13 ± 2.32 [-8.08, -6.17], 1.34e-15 / 3e-11 | 1.45 ± 5.35 [-0.74, 3.64], 0.188 / 1 | 0.056 ± 0.108 [0.012, 0.101], 0.015 / 1 |
| Three games, equal weight | 32/24/24 | V | Discounted sum | Original | -8.57 ± 2.05 [-9.40, -7.73], 3.94e-20 / 9.53e-16 | 0.66 ± 5.90 [-1.78, 3.11], 0.585 / 1 | 0.083 ± 0.120 [0.033, 0.133], 0.00181 / 1 |
| Three games, equal weight | 32/24/24 | Q | Final move | Fitted copies | -28.46 ± 2.57 [-29.51, -27.41], 2.04e-33 / 4.7e-32 | -8.05 ± 4.02 [-9.68, -6.41], 5.29e-13 / 8.46e-12 | -0.161 ± 0.079 [-0.192, -0.129], 1.4e-13 / 2.65e-12 |
| Three games, equal weight | 32/24/24 | Q | Final move | Original | -27.23 ± 2.82 [-28.39, -26.07], 7.5e-30 / 1.97e-25 | -6.65 ± 4.72 [-8.58, -4.71], 3.74e-08 / 0.000664 | -0.106 ± 0.092 [-0.144, -0.069], 1.45e-06 / 0.0239 |
| Three games, equal weight | 32/24/24 | Q | Sum | Fitted copies | -12.84 ± 2.47 [-13.87, -11.81], 2.67e-20 / 6.47e-16 | -6.16 ± 4.23 [-7.88, -4.43], 5.13e-09 / 9.42e-05 | -0.110 ± 0.083 [-0.144, -0.077], 3.73e-08 / 0.000663 |
| Three games, equal weight | 32/24/24 | Q | Sum | Original | -12.68 ± 2.33 [-13.65, -11.71], 5.44e-21 / 1.33e-16 | -5.93 ± 4.80 [-7.89, -3.97], 2.6e-07 / 0.00445 | -0.082 ± 0.093 [-0.120, -0.044], 8.92e-05 / 1 |
| Three games, equal weight | 32/24/24 | Q | Discounted sum | Fitted copies | -3.28 ± 1.42 [-3.86, -2.69], 6.87e-13 / 1.45e-08 | -0.06 ± 3.53 [-1.48, 1.36], 0.929 / 1 | 0.009 ± 0.067 [-0.018, 0.036], 0.499 / 1 |
| Three games, equal weight | 32/24/24 | Q | Discounted sum | Original | -3.37 ± 1.68 [-4.06, -2.67], 8.31e-11 / 1.65e-06 | -0.48 ± 3.49 [-1.88, 0.93], 0.498 / 1 | 0.024 ± 0.067 [-0.003, 0.051], 0.0803 / 1 |

### Randomized originals


| Game | Originals | Network | VNM whole-game score | Both seats continue with | Probability error (%) | Same move (%) | Wins (%) | Points/game |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tic-tac-toe | 32 | V | Final move | Fitted copies | 3.97 ± 1.44 [3.45, 4.49], 3.44e-16 / 7.78e-12 | -5.20 ± 2.22 [-6.00, -4.40], 2.5e-14 / 5.43e-10 | 2.00 ± 2.64 [1.05, 2.96], 0.000164 / 1 | 0.042 ± 0.057 [0.021, 0.062], 0.000247 / 1 |
| Tic-tac-toe | 32 | V | Final move | Original | 3.02 ± 1.43 [2.51, 3.54], 3.84e-13 / 8.14e-09 | -6.49 ± 1.64 [-7.09, -5.90], 1.08e-20 / 2.64e-16 | 1.81 ± 2.24 [1.00, 2.62], 7.41e-05 / 1 | 0.037 ± 0.060 [0.015, 0.059], 0.00154 / 1 |
| Tic-tac-toe | 32 | V | Sum | Fitted copies | -1.06 ± 1.44 [-1.58, -0.54], 0.000247 / 1 | -1.17 ± 2.04 [-1.91, -0.44], 0.00279 / 1 | 4.53 ± 2.39 [3.66, 5.39], 5.92e-12 / 1.22e-07 | 0.133 ± 0.045 [0.117, 0.149], 3.44e-17 / 7.93e-13 |
| Tic-tac-toe | 32 | V | Sum | Original | -1.60 ± 1.52 [-2.15, -1.05], 1.43e-06 / 0.0236 | -2.19 ± 2.00 [-2.91, -1.47], 6.99e-07 / 0.0117 | 3.91 ± 2.34 [3.07, 4.76], 1.18e-10 / 2.32e-06 | 0.125 ± 0.067 [0.100, 0.149], 9.94e-12 / 2.03e-07 |
| Tic-tac-toe | 32 | V | Discounted sum | Fitted copies | -1.56 ± 1.36 [-2.05, -1.07], 3.18e-07 / 0.00542 | 0.32 ± 2.55 [-0.60, 1.24], 0.484 / 1 | 4.92 ± 2.94 [3.86, 5.98], 1.11e-10 / 2.19e-06 | 0.151 ± 0.067 [0.127, 0.175], 7.83e-14 / 1.69e-09 |
| Tic-tac-toe | 32 | V | Discounted sum | Original | -1.76 ± 1.38 [-2.25, -1.26], 4.39e-08 / 0.000776 | -0.08 ± 3.08 [-1.19, 1.03], 0.883 / 1 | 4.40 ± 2.48 [3.51, 5.29], 2.81e-11 / 5.67e-07 | 0.163 ± 0.090 [0.130, 0.196], 1.97e-11 / 3.98e-07 |
| Tic-tac-toe | 32 | Q | Final move | Fitted copies | 5.82 ± 1.48 [5.29, 6.36], 1.16e-20 / 2.82e-16 | -11.56 ± 2.21 [-12.35, -10.76], 2.9e-24 / 7.34e-20 | -4.23 ± 2.53 [-5.15, -3.32], 1.15e-10 / 2.26e-06 | -0.126 ± 0.086 [-0.157, -0.095], 2.38e-09 / 4.44e-05 |
| Tic-tac-toe | 32 | Q | Final move | Original | 4.61 ± 1.31 [4.14, 5.08], 3.34e-19 / 7.93e-15 | -11.88 ± 2.03 [-12.62, -11.15], 9.84e-26 / 2.52e-21 | -3.46 ± 2.42 [-4.33, -2.58], 4.03e-09 / 7.45e-05 | -0.114 ± 0.077 [-0.142, -0.086], 1.62e-09 / 3.04e-05 |
| Tic-tac-toe | 32 | Q | Sum | Fitted copies | -1.28 ± 1.07 [-1.66, -0.89], 1.48e-07 / 0.00257 | -3.75 ± 2.63 [-4.70, -2.80], 4.23e-09 / 7.8e-05 | -1.28 ± 2.14 [-2.05, -0.51], 0.00195 / 1 | -0.035 ± 0.085 [-0.066, -0.004], 0.0271 / 1 |
| Tic-tac-toe | 32 | Q | Sum | Original | -1.84 ± 1.15 [-2.26, -1.42], 3.51e-10 / 6.77e-06 | -4.41 ± 2.59 [-5.34, -3.47], 7.88e-11 / 1.56e-06 | -0.98 ± 2.02 [-1.70, -0.25], 0.0104 / 1 | -0.039 ± 0.078 [-0.067, -0.011], 0.00737 / 1 |
| Tic-tac-toe | 32 | Q | Discounted sum | Fitted copies | -1.74 ± 1.23 [-2.18, -1.30], 4.66e-09 / 8.57e-05 | -1.41 ± 3.17 [-2.55, -0.27], 0.0173 / 1 | -0.17 ± 2.76 [-1.17, 0.82], 0.725 / 1 | -0.001 ± 0.083 [-0.031, 0.029], 0.948 / 1 |
| Tic-tac-toe | 32 | Q | Discounted sum | Original | -1.84 ± 1.12 [-2.24, -1.43], 1.86e-10 / 3.62e-06 | -1.32 ± 3.09 [-2.43, -0.20], 0.022 / 1 | 0.38 ± 2.11 [-0.38, 1.14], 0.317 / 1 | 0.008 ± 0.074 [-0.019, 0.035], 0.541 / 1 |
| Connect Four | 24 | V | Final move | Fitted copies | 34.83 ± 5.30 [32.59, 37.07], 1.24e-20 / 3.03e-16 | -42.67 ± 2.26 [-43.63, -41.71], 4.62e-31 / 1.22e-26 | 0.76 ± 9.85 [-3.39, 4.92], 0.707 / 1 | 0.018 ± 0.196 [-0.065, 0.101], 0.654 / 1 |
| Connect Four | 24 | V | Final move | Original | 35.21 ± 5.10 [33.06, 37.37], 4.1e-21 / 1.01e-16 | -42.21 ± 2.82 [-43.41, -41.02], 9.09e-29 / 2.37e-24 | 4.79 ± 8.90 [1.03, 8.54], 0.0148 / 1 | 0.097 ± 0.178 [0.022, 0.172], 0.0136 / 1 |
| Connect Four | 24 | V | Sum | Fitted copies | 30.80 ± 3.04 [29.52, 32.08], 6.8e-25 / 1.73e-20 | -33.02 ± 3.02 [-34.29, -31.74], 1.21e-25 / 3.1e-21 | -0.88 ± 16.60 [-7.89, 6.13], 0.798 / 1 | -0.011 ± 0.336 [-0.153, 0.130], 0.869 / 1 |
| Connect Four | 24 | V | Sum | Original | 34.43 ± 3.42 [32.99, 35.87], 7.63e-25 / 1.94e-20 | -37.75 ± 3.45 [-39.21, -36.29], 1.19e-25 / 3.06e-21 | -3.34 ± 17.36 [-10.67, 4.00], 0.356 / 1 | -0.063 ± 0.348 [-0.210, 0.084], 0.387 / 1 |
| Connect Four | 24 | V | Discounted sum | Fitted copies | 22.76 ± 3.58 [21.25, 24.27], 2.58e-20 / 6.24e-16 | -24.55 ± 3.90 [-26.19, -22.90], 3.2e-20 / 7.74e-16 | 4.07 ± 18.35 [-3.68, 11.82], 0.289 / 1 | 0.090 ± 0.374 [-0.067, 0.248], 0.248 / 1 |
| Connect Four | 24 | V | Discounted sum | Original | 27.02 ± 3.70 [25.46, 28.58], 1.14e-21 / 2.82e-17 | -30.17 ± 2.96 [-31.42, -28.92], 5.95e-25 / 1.52e-20 | 3.81 ± 17.88 [-3.74, 11.36], 0.308 / 1 | 0.079 ± 0.359 [-0.073, 0.231], 0.291 / 1 |
| Connect Four | 24 | Q | Final move | Fitted copies | 45.68 ± 5.47 [43.37, 47.99], 5.4e-23 / 1.35e-18 | -52.24 ± 5.64 [-54.62, -49.86], 5.21e-24 / 1.31e-19 | -8.84 ± 8.55 [-12.45, -5.23], 3.96e-05 / 0.602 | -0.181 ± 0.170 [-0.253, -0.110], 2.72e-05 / 0.419 |
| Connect Four | 24 | Q | Final move | Original | 44.60 ± 5.63 [42.22, 46.97], 1.81e-22 / 4.51e-18 | -52.08 ± 7.01 [-55.04, -49.12], 7.63e-22 / 1.89e-17 | 0.37 ± 8.54 [-3.23, 3.98], 0.832 / 1 | -0.002 ± 0.172 [-0.074, 0.071], 0.96 / 1 |
| Connect Four | 24 | Q | Sum | Fitted copies | 28.42 ± 4.40 [26.56, 30.28], 1.86e-20 / 4.51e-16 | -29.73 ± 4.48 [-31.62, -27.84], 1e-20 / 2.44e-16 | -4.46 ± 10.14 [-8.74, -0.18], 0.0419 / 1 | -0.087 ± 0.204 [-0.173, -0.001], 0.047 / 1 |
| Connect Four | 24 | Q | Sum | Original | 27.09 ± 4.68 [25.12, 29.07], 2.13e-19 / 5.07e-15 | -28.40 ± 4.50 [-30.30, -26.50], 3.02e-20 / 7.31e-16 | -4.90 ± 8.17 [-8.35, -1.45], 0.00739 / 1 | -0.102 ± 0.165 [-0.172, -0.032], 0.00595 / 1 |
| Connect Four | 24 | Q | Discounted sum | Fitted copies | 9.08 ± 3.12 [7.76, 10.40], 6.77e-13 / 1.43e-08 | -9.56 ± 3.16 [-10.90, -8.23], 2.85e-13 / 6.06e-09 | 7.10 ± 7.97 [3.73, 10.46], 0.000228 / 1 | 0.143 ± 0.160 [0.075, 0.210], 0.000234 / 1 |
| Connect Four | 24 | Q | Discounted sum | Original | 12.20 ± 3.27 [10.82, 13.59], 3.46e-15 / 7.68e-11 | -13.08 ± 2.74 [-14.24, -11.93], 1.54e-17 / 3.57e-13 | 2.64 ± 6.55 [-0.13, 5.40], 0.0607 / 1 | 0.055 ± 0.143 [-0.005, 0.115], 0.0718 / 1 |
| 4×4 Reversi | 24 | V | Final move | Fitted copies | -0.38 ± 1.43 [-0.98, 0.23], 0.212 / 1 | -2.78 ± 0.80 [-3.12, -2.44], 1.75e-14 / 3.81e-10 | -4.67 ± 5.58 [-7.03, -2.32], 0.000436 / 1 | -0.081 ± 0.103 [-0.125, -0.038], 0.000769 / 1 |
| 4×4 Reversi | 24 | V | Final move | Original | -0.44 ± 1.36 [-1.01, 0.14], 0.129 / 1 | -2.35 ± 0.72 [-2.65, -2.05], 5.32e-14 / 1.15e-09 | -0.88 ± 5.79 [-3.33, 1.57], 0.465 / 1 | -0.000 ± 0.107 [-0.045, 0.045], 0.982 / 1 |
| 4×4 Reversi | 24 | V | Sum | Fitted copies | -0.06 ± 1.30 [-0.61, 0.49], 0.826 / 1 | -2.11 ± 0.97 [-2.52, -1.70], 2.29e-10 / 4.44e-06 | -12.73 ± 4.42 [-14.60, -10.86], 8.36e-13 / 1.76e-08 | -0.227 ± 0.086 [-0.264, -0.191], 5.3e-12 / 1.09e-07 |
| 4×4 Reversi | 24 | V | Sum | Original | -0.06 ± 1.16 [-0.55, 0.43], 0.792 / 1 | -2.06 ± 0.97 [-2.47, -1.65], 3.29e-10 / 6.35e-06 | -11.65 ± 5.14 [-13.82, -9.48], 1.01e-10 / 1.98e-06 | -0.209 ± 0.099 [-0.251, -0.167], 3.71e-10 / 7.15e-06 |
| 4×4 Reversi | 24 | V | Discounted sum | Fitted copies | -0.14 ± 1.30 [-0.69, 0.41], 0.592 / 1 | -1.85 ± 1.01 [-2.28, -1.42], 5.46e-09 / 0.0001 | -11.49 ± 4.36 [-13.33, -9.65], 4.96e-12 / 1.02e-07 | -0.201 ± 0.088 [-0.238, -0.164], 8.12e-11 / 1.61e-06 |
| 4×4 Reversi | 24 | V | Discounted sum | Original | -0.12 ± 1.25 [-0.65, 0.41], 0.647 / 1 | -1.69 ± 0.99 [-2.11, -1.27], 1.98e-08 / 0.000356 | -11.25 ± 5.00 [-13.36, -9.14], 1.19e-10 / 2.33e-06 | -0.200 ± 0.099 [-0.242, -0.158], 9.72e-10 / 1.84e-05 |
| 4×4 Reversi | 24 | Q | Final move | Fitted copies | 1.90 ± 1.59 [1.23, 2.57], 5.74e-06 / 0.0916 | -4.53 ± 0.66 [-4.81, -4.25], 4.4e-21 / 1.08e-16 | -9.33 ± 4.77 [-11.34, -7.31], 1.74e-09 / 3.25e-05 | -0.178 ± 0.087 [-0.214, -0.141], 6.82e-10 / 1.3e-05 |
| 4×4 Reversi | 24 | Q | Final move | Original | 1.79 ± 1.44 [1.18, 2.40], 3.29e-06 / 0.0533 | -4.02 ± 0.64 [-4.30, -3.75], 3.63e-20 / 8.79e-16 | -5.24 ± 5.65 [-7.63, -2.86], 0.000145 / 1 | -0.096 ± 0.103 [-0.139, -0.053], 0.00013 / 1 |
| 4×4 Reversi | 24 | Q | Sum | Fitted copies | 1.33 ± 1.21 [0.82, 1.84], 1.7e-05 / 0.265 | -3.59 ± 1.13 [-4.07, -3.11], 1.04e-13 / 2.24e-09 | -15.51 ± 4.24 [-17.30, -13.72], 5.19e-15 / 1.15e-10 | -0.293 ± 0.080 [-0.327, -0.259], 5e-15 / 1.11e-10 |
| 4×4 Reversi | 24 | Q | Sum | Original | 1.24 ± 1.26 [0.71, 1.77], 6.89e-05 / 1 | -3.43 ± 0.86 [-3.79, -3.07], 7.87e-16 / 1.77e-11 | -15.89 ± 5.09 [-18.03, -13.74], 1.52e-13 / 3.25e-09 | -0.294 ± 0.094 [-0.334, -0.254], 1.59e-13 / 3.39e-09 |
| 4×4 Reversi | 24 | Q | Discounted sum | Fitted copies | 1.28 ± 1.14 [0.80, 1.76], 1.24e-05 / 0.195 | -3.05 ± 1.17 [-3.55, -2.56], 5.74e-12 / 1.18e-07 | -13.48 ± 4.50 [-15.38, -11.58], 3.63e-13 / 7.7e-09 | -0.251 ± 0.087 [-0.287, -0.214], 7.65e-13 / 1.61e-08 |
| 4×4 Reversi | 24 | Q | Discounted sum | Original | 1.18 ± 1.24 [0.66, 1.71], 0.000109 / 1 | -2.99 ± 1.03 [-3.43, -2.55], 7.35e-13 / 1.55e-08 | -14.62 ± 5.20 [-16.81, -12.42], 1.36e-12 / 2.85e-08 | -0.267 ± 0.098 [-0.308, -0.226], 2.35e-12 / 4.89e-08 |
| Three games, equal weight | 32/24/24 | V | Final move | Fitted copies | 12.81 ± 1.89 [12.02, 13.59], 9.41e-25 / 2.39e-20 | -16.88 ± 1.09 [-17.30, -16.46], 1.44e-59 / 4.08e-55 | -0.63 ± 3.87 [-2.22, 0.95], 0.424 / 1 | -0.007 ± 0.076 [-0.038, 0.024], 0.645 / 1 |
| Three games, equal weight | 32/24/24 | V | Final move | Original | 12.60 ± 1.82 [11.84, 13.35], 4.71e-25 / 1.2e-20 | -17.02 ± 1.11 [-17.47, -16.57], 2.2e-43 / 6.17e-39 | 1.91 ± 3.62 [0.42, 3.39], 0.013 / 1 | 0.045 ± 0.072 [0.015, 0.074], 0.00383 / 1 |
| Three games, equal weight | 32/24/24 | V | Sum | Fitted copies | 9.90 ± 1.20 [9.41, 10.38], 2.97e-34 / 8.04e-30 | -12.10 ± 1.26 [-12.60, -11.60], 6.95e-40 / 1.93e-35 | -3.03 ± 5.78 [-5.44, -0.61], 0.016 / 1 | -0.035 ± 0.117 [-0.084, 0.014], 0.151 / 1 |
| Three games, equal weight | 32/24/24 | V | Sum | Original | 10.92 ± 1.30 [10.39, 11.45], 6.9e-32 / 1.84e-27 | -14.00 ± 1.37 [-14.55, -13.45], 3.05e-37 / 8.38e-33 | -3.69 ± 6.09 [-6.23, -1.15], 0.00598 / 1 | -0.049 ± 0.123 [-0.100, 0.002], 0.059 / 1 |
| Three games, equal weight | 32/24/24 | V | Discounted sum | Fitted copies | 7.02 ± 1.35 [6.47, 7.57], 3.48e-24 / 8.8e-20 | -8.69 ± 1.59 [-9.32, -8.06], 3.12e-28 / 8.13e-24 | -0.83 ± 6.36 [-3.49, 1.83], 0.525 / 1 | 0.013 ± 0.130 [-0.041, 0.068], 0.615 / 1 |
| Three games, equal weight | 32/24/24 | V | Discounted sum | Original | 8.38 ± 1.38 [7.82, 8.95], 7.52e-26 / 1.93e-21 | -10.65 ± 1.46 [-11.21, -10.09], 4.42e-42 / 1.24e-37 | -1.01 ± 6.24 [-3.62, 1.60], 0.433 / 1 | 0.014 ± 0.128 [-0.039, 0.067], 0.591 / 1 |
| Three games, equal weight | 32/24/24 | Q | Final move | Fitted copies | 17.80 ± 1.96 [16.99, 18.61], 9.52e-29 / 2.49e-24 | -22.77 ± 2.03 [-23.61, -21.94], 4.23e-31 / 1.12e-26 | -7.47 ± 3.37 [-8.85, -6.09], 1.52e-13 / 3.25e-09 | -0.162 ± 0.070 [-0.190, -0.134], 5.43e-15 / 1.2e-10 |
| Three games, equal weight | 32/24/24 | Q | Final move | Original | 17.00 ± 1.99 [16.17, 17.82], 6.99e-27 / 1.81e-22 | -22.66 ± 2.44 [-23.68, -21.65], 1.13e-26 / 2.92e-22 | -2.77 ± 3.51 [-4.21, -1.34], 0.000331 / 1 | -0.071 ± 0.071 [-0.099, -0.042], 1.16e-05 / 0.182 |
| Three games, equal weight | 32/24/24 | Q | Sum | Fitted copies | 9.49 ± 1.56 [8.84, 10.14], 3.82e-23 / 9.58e-19 | -12.36 ± 1.77 [-13.07, -11.65], 1e-30 / 2.65e-26 | -7.08 ± 3.73 [-8.63, -5.54], 9.93e-11 / 1.96e-06 | -0.138 ± 0.078 [-0.170, -0.107], 1.41e-10 / 2.76e-06 |
| Three games, equal weight | 32/24/24 | Q | Sum | Original | 8.83 ± 1.66 [8.14, 9.52], 1.58e-21 / 3.89e-17 | -12.08 ± 1.75 [-12.78, -11.37], 1.77e-29 / 4.63e-25 | -7.25 ± 3.28 [-8.60, -5.91], 1.09e-13 / 2.34e-09 | -0.145 ± 0.068 [-0.173, -0.118], 7.5e-14 / 1.62e-09 |
| Three games, equal weight | 32/24/24 | Q | Discounted sum | Fitted copies | 2.87 ± 1.18 [2.39, 3.36], 4.84e-14 / 1.05e-09 | -4.68 ± 1.54 [-5.27, -4.08], 1.55e-22 / 3.87e-18 | -2.18 ± 3.19 [-3.48, -0.89], 0.00153 / 1 | -0.036 ± 0.067 [-0.063, -0.010], 0.009 / 1 |
| Three games, equal weight | 32/24/24 | Q | Discounted sum | Original | 3.85 ± 1.23 [3.35, 4.35], 4.81e-17 / 1.11e-12 | -5.80 ± 1.42 [-6.34, -5.26], 9.61e-30 / 2.53e-25 | -3.87 ± 2.88 [-5.04, -2.70], 2.67e-08 / 0.000477 | -0.068 ± 0.063 [-0.093, -0.043], 1.66e-06 / 0.0273 |

[Report and interpretation](../../PHASE_V.md) · [Design](../../PHASE_V_PROTOCOL.md) · [File guide and denominators](README.md) · [All source-level results](summary.jsonl.gz) · [Equally weighted three-game comparisons](three_games.jsonl.gz) · [Method rankings and paired differences](rankings.json)
