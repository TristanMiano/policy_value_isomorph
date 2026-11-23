# Confirmation results for visit-trained game scores

The frozen original takes its highest-scoring legal move. A network scoring the resulting board (V) or the board and move (Q) estimates the fraction of observed visits taking a move. Direct selection takes the largest immediate fit. Von Neumann–Morgenstern (VNM) selection maximizes expected complete-game score, replanning on each actual turn. Ties prefer the lowest move index. 'copies' projects highest fitted choices for both seats. The rule 'uniform_reply' projects greedy own moves against uniform legal replies. The rule 'mixed_reply' mixes those two opponents equally once per path. The rule 'uniform_both' projects uniform legal moves for both seats. Each random component gets the listed number of full paths, with exact expectations in tic-tac-toe. The assisted comparison includes an opponent considering tactics and one using positional priorities. The supplied-outcome reference adds actual game results and projects guarded own play against four opponent types.

The utility name specifies the score of a completed game from the current choosing seat. 'signed' counts own moves positively and opposing moves negatively, 'unsigned' counts both positively, and 'own' counts only own moves. 'last' counts the final fitted frequency. The rule 'sum' adds the whole path. The suffix 50, 75, 90 or 97 gives the discount per turn in percent, starting at weight one. 'sqrt_last' and 'square_last' apply the square root or square to the final frequency, signed by mover. 'last_plus_sum' adds the signed last frequency and 0.1 times the signed 0.9-discounted sum. A constant score uses 0.5 in the identical formula. Constant continuations also break all greedy choices by move index. Both constants induce the identical player when all continuing moves are uniformly random.

Win, draw and loss percentages are seat-balanced against four equally weighted external opponent groups. Agreement is the weighted percentage of withheld boards on which the reconstruction chose the original's favorite. Exact-copy self-play uses two identical deployed players and counts both seats. Values give means ± sample standard deviations across originals after averaging their four fits. The equal-game spread describes the average of one independent original per named game. The selected rule is VNM Q, signed_last_90, uniform_both, 8 paths.

Independent original counts for tic-tac-toe, Connect Four and 4×4 Reversi are 24, 16, 16.
For paired changes, brackets give individual 95% confidence intervals, followed by the two-sided p-value for a zero mean difference, Holm-adjusted across the table's win and agreement comparisons in all four populations. A dash means the across-original variance is exactly zero. Discovery contrasts are exploratory because the leader was chosen on those data. The compressed estimates and contrasts alongside this table contain all unrounded measurements, raw p-values, original-level differences, and intervals resampling the independent originals.

## External win rate (%)

| Rule | Tic-tac-toe | Connect Four | 4×4 Reversi | Equal games |
|---|---|---|---|---|
| VNM V, supplied outcomes and frequencies, guarded mixture | 50.56 ± 4.65 | 90.62 ± 3.20 | 76.88 ± 2.25 | 72.69 ± 2.03 |
| VNM Q, signed_last_90, uniform_both, 8 paths, constant scores and continuations | 46.88 ± 2.02 | 78.22 ± 4.82 | 51.20 ± 5.13 | 58.76 ± 2.44 |
| VNM Q, signed_last_90, uniform_both, 8 paths, constant scores | 46.88 ± 2.02 | 78.22 ± 4.82 | 51.20 ± 5.13 | 58.76 ± 2.44 |
| Frozen original | 45.71 ± 4.97 | 58.45 ± 12.27 | 69.24 ± 2.04 | 57.80 ± 4.47 |
| VNM Q, signed_last_90, uniform_both, 8 paths | 45.74 ± 1.89 | 67.75 ± 11.40 | 45.75 ± 4.52 | 53.08 ± 4.14 |
| VNM V, signed_last_90, uniform_both, 8 paths | 45.49 ± 1.95 | 67.82 ± 8.83 | 44.09 ± 5.21 | 52.47 ± 3.48 |
| VNM V, signed_last_90, mixed_reply, 8 paths | 43.79 ± 2.09 | 69.41 ± 11.85 | 42.02 ± 5.77 | 51.74 ± 4.45 |
| VNM V, signed final frequency, assisted opponent mixture | 39.98 ± 2.10 | 70.51 ± 6.40 | 43.02 ± 5.60 | 51.17 ± 2.92 |
| VNM Q, signed_last, uniform_both, 8 paths | 43.83 ± 2.19 | 59.84 ± 7.44 | 45.51 ± 4.17 | 49.72 ± 2.93 |
| VNM Q, signed_90, copies, 8 paths | 37.96 ± 3.60 | 54.66 ± 7.16 | 48.83 ± 5.84 | 47.15 ± 3.30 |
| VNM Q, signed_90, uniform_both, 8 paths | 37.74 ± 4.72 | 51.78 ± 4.96 | 43.31 ± 4.49 | 44.28 ± 2.73 |
| Direct Q | 29.25 ± 6.47 | 49.76 ± 5.67 | 52.88 ± 2.73 | 43.96 ± 3.01 |
| VNM Q, own_90, uniform_both, 8 paths | 24.85 ± 4.28 | 45.73 ± 7.66 | 51.29 ± 3.10 | 40.62 ± 3.10 |
| Direct V | 28.44 ± 3.92 | 37.67 ± 10.90 | 45.87 ± 4.63 | 37.33 ± 4.16 |
| VNM Q, unsigned_90, uniform_both, 8 paths | 16.00 ± 2.17 | 21.07 ± 7.63 | 50.98 ± 3.33 | 29.35 ± 2.87 |
| VNM Q, unsigned_90, uniform_reply, 8 paths | 16.24 ± 2.36 | 14.38 ± 4.52 | 54.08 ± 3.57 | 28.23 ± 2.07 |

## Tic-tac-toe

| Rule | External wins / draws / losses (%) | Move agreement (%) | Exact-copy wins / draws (%) | Selected minus row in wins, points [95%], p | Selected minus row in agreement, points [95%], p |
|---|---|---|---|---|---|
| VNM V, supplied outcomes and frequencies, guarded mixture | 50.56 ± 4.65 / 28.67 ± 9.75 / 20.76 ± 8.81 | 76.77 ± 2.65 | 35.94 ± 12.93 / 28.12 ± 25.87 | -4.82 ± 5.04 [-6.95, -2.70], 0.00412 | -12.67 ± 2.79 [-13.85, -11.49], 4.99e-15 |
| VNM Q, signed_last_90, uniform_both, 8 paths, constant scores and continuations | 46.88 ± 2.02 / 25.40 ± 1.02 / 27.73 ± 1.92 | 62.57 ± 3.08 | 50.00 ± 0.00 / 0.00 ± 0.00 | -1.13 ± 1.72 [-1.86, -0.41], 0.124 | +1.53 ± 2.86 [0.32, 2.74], 0.396 |
| VNM Q, signed_last_90, uniform_both, 8 paths, constant scores | 46.88 ± 2.02 / 25.40 ± 1.02 / 27.73 ± 1.92 | 62.57 ± 3.08 | 50.00 ± 0.00 / 0.00 ± 0.00 | -1.13 ± 1.72 [-1.86, -0.41], 0.124 | +1.53 ± 2.86 [0.32, 2.74], 0.396 |
| Frozen original | 45.71 ± 4.97 / 54.29 ± 4.97 / 0.00 ± 0.00 | 100.00 ± 0.00 | 0.00 ± 0.00 / 100.00 ± 0.00 | +0.03 ± 5.16 [-2.14, 2.21], 1 | -35.90 ± 2.51 [-36.96, -34.84], 2.97e-26 |
| VNM Q, signed_last_90, uniform_both, 8 paths | 45.74 ± 1.89 / 27.53 ± 2.61 / 26.73 ± 2.15 | 64.10 ± 2.51 | 43.23 ± 9.01 / 13.54 ± 18.03 | reference | reference |
| VNM V, signed_last_90, uniform_both, 8 paths | 45.49 ± 1.95 / 27.27 ± 1.40 / 27.24 ± 1.48 | 64.23 ± 3.24 | 40.62 ± 11.80 / 18.75 ± 23.60 | +0.25 ± 1.64 [-0.44, 0.95], 1 | -0.13 ± 1.73 [-0.87, 0.60], 1 |
| VNM V, signed_last_90, mixed_reply, 8 paths | 43.79 ± 2.09 / 22.19 ± 4.90 / 34.03 ± 4.74 | 61.55 ± 3.05 | 42.71 ± 8.97 / 14.58 ± 17.93 | +1.95 ± 2.94 [0.71, 3.20], 0.12 | +2.55 ± 2.05 [1.68, 3.41], 0.000189 |
| VNM V, signed final frequency, assisted opponent mixture | 39.98 ± 2.10 / 28.70 ± 3.56 / 31.32 ± 3.25 | 58.46 ± 3.13 | 43.75 ± 7.37 / 12.50 ± 14.74 | +5.76 ± 3.09 [4.45, 7.07], 3.17e-07 | +5.64 ± 1.91 [4.84, 6.45], 4.74e-11 |
| VNM Q, signed_last, uniform_both, 8 paths | 43.83 ± 2.19 / 28.94 ± 3.05 / 27.23 ± 1.91 | 61.32 ± 2.60 | 45.31 ± 8.09 / 9.38 ± 16.17 | +1.91 ± 1.58 [1.25, 2.58], 0.000261 | +2.78 ± 0.58 [2.53, 3.02], 1.74e-15 |
| VNM Q, signed_90, copies, 8 paths | 37.96 ± 3.60 / 13.90 ± 8.38 / 48.14 ± 8.70 | 65.20 ± 2.37 | 41.67 ± 10.85 / 16.67 ± 21.70 | +7.78 ± 3.66 [6.23, 9.32], 2.81e-08 | -1.10 ± 2.94 [-2.35, 0.14], 1 |
| VNM Q, signed_90, uniform_both, 8 paths | 37.74 ± 4.72 / 16.18 ± 10.35 / 46.08 ± 8.08 | 65.56 ± 2.23 | 45.83 ± 7.96 / 8.33 ± 15.93 | +8.00 ± 4.89 [5.93, 10.06], 2.92e-06 | -1.46 ± 3.05 [-2.75, -0.18], 0.665 |
| Direct Q | 29.25 ± 6.47 / 12.61 ± 9.46 / 58.14 ± 8.41 | 56.96 ± 3.43 | 44.79 ± 7.29 / 10.42 ± 14.59 | +16.49 ± 6.60 [13.70, 19.28], 1.31e-09 | +7.14 ± 4.07 [5.42, 8.85], 8.76e-07 |
| VNM Q, own_90, uniform_both, 8 paths | 24.85 ± 4.28 / 29.87 ± 11.72 / 45.28 ± 10.51 | 50.61 ± 3.09 | 25.52 ± 12.49 / 48.96 ± 24.98 | +20.89 ± 4.63 [18.94, 22.85], 5.76e-15 | +13.49 ± 3.87 [11.86, 15.12], 1.47e-12 |
| Direct V | 28.44 ± 3.92 / 14.71 ± 8.74 / 56.85 ± 7.79 | 53.45 ± 3.52 | 44.79 ± 7.29 / 10.42 ± 14.59 | +17.30 ± 4.48 [15.41, 19.19], 1.66e-13 | +10.65 ± 3.99 [8.97, 12.33], 3.47e-10 |
| VNM Q, unsigned_90, uniform_both, 8 paths | 16.00 ± 2.17 / 43.73 ± 10.29 / 40.28 ± 10.00 | 37.62 ± 2.11 | 24.48 ± 11.93 / 51.04 ± 23.86 | +29.74 ± 2.76 [28.58, 30.91], 1.92e-23 | +26.48 ± 3.70 [24.92, 28.04], 2.05e-19 |
| VNM Q, unsigned_90, uniform_reply, 8 paths | 16.24 ± 2.36 / 43.47 ± 9.15 / 40.29 ± 9.18 | 36.27 ± 2.03 | 24.48 ± 13.02 / 51.04 ± 26.04 | +29.50 ± 3.11 [28.19, 30.81], 3.5e-22 | +27.83 ± 3.53 [26.34, 29.32], 2.26e-20 |

## Connect Four

| Rule | External wins / draws / losses (%) | Move agreement (%) | Exact-copy wins / draws (%) | Selected minus row in wins, points [95%], p | Selected minus row in agreement, points [95%], p |
|---|---|---|---|---|---|
| VNM V, supplied outcomes and frequencies, guarded mixture | 90.62 ± 3.20 / 0.66 ± 0.71 / 8.72 ± 3.11 | 30.97 ± 6.96 | 49.34 ± 1.73 / 1.32 ± 3.46 | -22.88 ± 11.42 [-28.96, -16.79], 5.42e-05 | +0.50 ± 5.58 [-2.48, 3.47], 1 |
| VNM Q, signed_last_90, uniform_both, 8 paths, constant scores and continuations | 78.22 ± 4.82 / 0.27 ± 0.51 / 21.51 ± 5.01 | 26.78 ± 10.82 | 49.41 ± 1.65 / 1.17 ± 3.30 | -10.47 ± 14.08 [-17.98, -2.97], 0.274 | +4.69 ± 8.93 [-0.07, 9.45], 1 |
| VNM Q, signed_last_90, uniform_both, 8 paths, constant scores | 78.22 ± 4.82 / 0.27 ± 0.51 / 21.51 ± 5.01 | 26.78 ± 10.82 | 49.41 ± 1.65 / 1.17 ± 3.30 | -10.47 ± 14.08 [-17.98, -2.97], 0.274 | +4.69 ± 8.93 [-0.07, 9.45], 1 |
| Frozen original | 58.45 ± 12.27 / 0.68 ± 1.94 / 40.87 ± 13.28 | 100.00 ± 0.00 | 49.49 ± 0.86 / 1.03 ± 1.73 | +9.30 ± 9.37 [4.31, 14.29], 0.0455 | -68.54 ± 2.58 [-69.91, -67.16], 6.13e-21 |
| VNM Q, signed_last_90, uniform_both, 8 paths | 67.75 ± 11.40 / 0.00 ± 0.00 / 32.25 ± 11.40 | 31.46 ± 2.58 | 49.88 ± 0.24 / 0.24 ± 0.47 | reference | reference |
| VNM V, signed_last_90, uniform_both, 8 paths | 67.82 ± 8.83 / 0.07 ± 0.21 / 32.10 ± 8.80 | 32.70 ± 3.48 | 49.95 ± 0.13 / 0.10 ± 0.27 | -0.07 ± 6.96 [-3.78, 3.64], 1 | -1.23 ± 2.26 [-2.43, -0.03], 1 |
| VNM V, signed_last_90, mixed_reply, 8 paths | 69.41 ± 11.85 / 0.20 ± 0.43 / 30.40 ± 11.98 | 27.78 ± 4.53 | 49.56 ± 0.45 / 0.88 ± 0.90 | -1.66 ± 5.75 [-4.73, 1.41], 1 | +3.68 ± 3.85 [1.63, 5.73], 0.0599 |
| VNM V, signed final frequency, assisted opponent mixture | 70.51 ± 6.40 / 0.54 ± 0.75 / 28.96 ± 6.97 | 25.68 ± 4.34 | 48.90 ± 2.03 / 2.20 ± 4.06 | -2.76 ± 7.15 [-6.57, 1.05], 1 | +5.78 ± 3.47 [3.93, 7.63], 0.000382 |
| VNM Q, signed_last, uniform_both, 8 paths | 59.84 ± 7.44 / 0.17 ± 0.49 / 39.99 ± 7.55 | 28.91 ± 2.66 | 49.78 ± 0.32 / 0.44 ± 0.64 | +7.91 ± 5.37 [5.05, 10.77], 0.0013 | +2.56 ± 1.50 [1.76, 3.36], 0.000307 |
| VNM Q, signed_90, copies, 8 paths | 54.66 ± 7.16 / 0.27 ± 0.63 / 45.07 ± 7.55 | 82.42 ± 10.53 | 49.71 ± 0.30 / 0.59 ± 0.61 | +13.09 ± 6.20 [9.78, 16.39], 2.9e-05 | -50.96 ± 11.89 [-57.29, -44.62], 2.47e-09 |
| VNM Q, signed_90, uniform_both, 8 paths | 51.78 ± 4.96 / 0.27 ± 0.68 / 47.95 ± 5.20 | 85.59 ± 10.70 | 49.83 ± 0.40 / 0.34 ± 0.81 | +15.97 ± 8.33 [11.53, 20.41], 8.85e-05 | -54.12 ± 12.04 [-60.54, -47.71], 1.29e-09 |
| Direct Q | 49.76 ± 5.67 / 0.02 ± 0.10 / 50.22 ± 5.63 | 87.97 ± 10.79 | 49.58 ± 0.56 / 0.83 ± 1.12 | +17.99 ± 11.06 [12.10, 23.88], 0.000483 | -56.50 ± 12.32 [-63.07, -49.94], 9.81e-10 |
| VNM Q, own_90, uniform_both, 8 paths | 45.73 ± 7.66 / 2.03 ± 2.61 / 52.25 ± 8.91 | 84.86 ± 12.05 | 36.87 ± 8.59 / 26.27 ± 17.17 | +22.02 ± 7.89 [17.82, 26.23], 8.29e-07 | -53.40 ± 13.46 [-60.57, -46.23], 7.1e-09 |
| Direct V | 37.67 ± 10.90 / 0.46 ± 0.73 / 61.87 ± 11.22 | 71.97 ± 4.49 | 49.49 ± 0.58 / 1.03 ± 1.17 | +30.08 ± 7.51 [26.08, 34.08], 6.21e-09 | -40.51 ± 5.71 [-43.55, -37.47], 1.89e-12 |
| VNM Q, unsigned_90, uniform_both, 8 paths | 21.07 ± 7.63 / 4.15 ± 4.39 / 74.78 ± 11.14 | 76.63 ± 11.58 | 31.91 ± 8.85 / 36.18 ± 17.69 | +46.68 ± 7.11 [42.89, 50.47], 5.87e-12 | -45.17 ± 12.86 [-52.02, -38.32], 3.74e-08 |
| VNM Q, unsigned_90, uniform_reply, 8 paths | 14.38 ± 4.52 / 4.88 ± 3.70 / 80.74 ± 7.34 | 49.35 ± 6.57 | 33.91 ± 10.97 / 32.18 ± 21.94 | +53.37 ± 8.08 [49.07, 57.67], 5.38e-12 | -17.88 ± 7.77 [-22.02, -13.74], 9.85e-06 |

## 4×4 Reversi

| Rule | External wins / draws / losses (%) | Move agreement (%) | Exact-copy wins / draws (%) | Selected minus row in wins, points [95%], p | Selected minus row in agreement, points [95%], p |
|---|---|---|---|---|---|
| VNM V, supplied outcomes and frequencies, guarded mixture | 76.88 ± 2.25 / 1.78 ± 0.87 / 21.34 ± 2.20 | 86.12 ± 0.99 | 49.73 ± 0.31 / 0.54 ± 0.62 | -31.13 ± 4.96 [-33.77, -28.49], 1.12e-11 | -5.71 ± 1.55 [-6.53, -4.88], 1.95e-08 |
| VNM Q, signed_last_90, uniform_both, 8 paths, constant scores and continuations | 51.20 ± 5.13 / 5.76 ± 1.64 / 43.04 ± 4.92 | 86.66 ± 1.16 | 46.39 ± 3.26 / 7.23 ± 6.52 | -5.44 ± 3.24 [-7.17, -3.72], 0.000362 | -6.24 ± 1.23 [-6.89, -5.59], 2.3e-10 |
| VNM Q, signed_last_90, uniform_both, 8 paths, constant scores | 51.20 ± 5.13 / 5.76 ± 1.64 / 43.04 ± 4.92 | 86.66 ± 1.16 | 46.39 ± 3.26 / 7.23 ± 6.52 | -5.44 ± 3.24 [-7.17, -3.72], 0.000362 | -6.24 ± 1.23 [-6.89, -5.59], 2.3e-10 |
| Frozen original | 69.24 ± 2.04 / 1.42 ± 0.62 / 29.35 ± 1.78 | 100.00 ± 0.00 | 49.78 ± 0.32 / 0.44 ± 0.64 | -23.49 ± 4.91 [-26.10, -20.87], 5.42e-10 | -19.58 ± 1.50 [-20.38, -18.78], 2.33e-16 |
| VNM Q, signed_last_90, uniform_both, 8 paths | 45.75 ± 4.52 / 8.33 ± 2.24 / 45.92 ± 3.98 | 80.42 ± 1.50 | 46.66 ± 2.57 / 6.69 ± 5.13 | reference | reference |
| VNM V, signed_last_90, uniform_both, 8 paths | 44.09 ± 5.21 / 8.42 ± 2.11 / 47.49 ± 4.79 | 82.34 ± 1.40 | 46.56 ± 2.99 / 6.88 ± 5.97 | +1.66 ± 2.26 [0.45, 2.87], 0.276 | -1.92 ± 0.94 [-2.42, -1.42], 4.28e-05 |
| VNM V, signed_last_90, mixed_reply, 8 paths | 42.02 ± 5.77 / 8.54 ± 2.42 / 49.44 ± 5.24 | 82.27 ± 1.17 | 47.09 ± 2.75 / 5.81 ± 5.50 | +3.74 ± 7.51 [-0.26, 7.73], 1 | -1.86 ± 1.19 [-2.49, -1.22], 0.00072 |
| VNM V, signed final frequency, assisted opponent mixture | 43.02 ± 5.60 / 8.67 ± 2.40 / 48.32 ± 5.21 | 82.73 ± 0.98 | 46.41 ± 3.03 / 7.18 ± 6.06 | +2.73 ± 6.62 [-0.79, 6.26], 1 | -2.32 ± 1.25 [-2.98, -1.65], 0.000124 |
| VNM Q, signed_last, uniform_both, 8 paths | 45.51 ± 4.17 / 8.54 ± 1.96 / 45.95 ± 3.78 | 80.42 ± 1.37 | 46.75 ± 2.76 / 6.49 ± 5.51 | +0.24 ± 2.11 [-0.88, 1.37], 1 | -0.00 ± 0.33 [-0.18, 0.17], 1 |
| VNM Q, signed_90, copies, 8 paths | 48.83 ± 5.84 / 5.54 ± 1.83 / 45.63 ± 5.32 | 83.82 ± 1.35 | 46.46 ± 3.86 / 7.08 ± 7.72 | -3.08 ± 7.73 [-7.19, 1.04], 1 | -3.40 ± 1.93 [-4.43, -2.37], 0.000223 |
| VNM Q, signed_90, uniform_both, 8 paths | 43.31 ± 4.49 / 7.23 ± 1.97 / 49.46 ± 4.19 | 83.85 ± 1.12 | 46.29 ± 3.08 / 7.42 ± 6.16 | +2.44 ± 6.14 [-0.83, 5.71], 1 | -3.44 ± 1.82 [-4.41, -2.46], 0.000107 |
| Direct Q | 52.88 ± 2.73 / 5.25 ± 1.61 / 41.87 ± 3.19 | 86.51 ± 1.29 | 48.49 ± 0.78 / 3.03 ± 1.56 | -7.13 ± 4.47 [-9.51, -4.75], 0.000592 | -6.09 ± 2.06 [-7.19, -4.99], 3.96e-07 |
| VNM Q, own_90, uniform_both, 8 paths | 51.29 ± 3.10 / 4.74 ± 1.23 / 43.97 ± 3.31 | 85.54 ± 1.55 | 48.41 ± 0.97 / 3.17 ± 1.94 | -5.54 ± 5.96 [-8.72, -2.37], 0.0717 | -5.12 ± 1.94 [-6.16, -4.09], 1.71e-06 |
| Direct V | 45.87 ± 4.63 / 4.81 ± 1.73 / 49.32 ± 4.70 | 84.28 ± 1.42 | 47.90 ± 2.02 / 4.20 ± 4.03 | -0.12 ± 5.09 [-2.84, 2.59], 1 | -3.87 ± 2.26 [-5.07, -2.66], 0.000307 |
| VNM Q, unsigned_90, uniform_both, 8 paths | 50.98 ± 3.33 / 4.83 ± 1.29 / 44.19 ± 3.15 | 83.45 ± 1.50 | 48.66 ± 1.58 / 2.69 ± 3.15 | -5.22 ± 6.32 [-8.59, -1.86], 0.144 | -3.03 ± 1.96 [-4.08, -1.99], 0.000798 |
| VNM Q, unsigned_90, uniform_reply, 8 paths | 54.08 ± 3.57 / 5.03 ± 1.56 / 40.89 ± 3.01 | 83.47 ± 1.50 | 48.58 ± 1.81 / 2.83 ± 3.62 | -8.33 ± 4.24 [-10.58, -6.07], 6.61e-05 | -3.05 ± 2.09 [-4.16, -1.94], 0.00141 |

## Equal games

| Rule | External wins / draws / losses (%) | Move agreement (%) | Exact-copy wins / draws (%) | Selected minus row in wins, points [95%], p | Selected minus row in agreement, points [95%], p |
|---|---|---|---|---|---|
| VNM V, supplied outcomes and frequencies, guarded mixture | 72.69 ± 2.03 / 10.37 ± 3.27 / 16.94 ± 3.20 | 64.62 ± 2.51 | 45.00 ± 4.35 / 9.99 ± 8.70 | -19.61 ± 4.48 [-21.86, -17.36], 9.72e-14 | -5.96 ± 2.14 [-7.03, -4.89], 5.29e-09 |
| VNM Q, signed_last_90, uniform_both, 8 paths, constant scores and continuations | 58.76 ± 2.44 / 10.48 ± 0.67 / 30.76 ± 2.43 | 58.67 ± 3.77 | 48.60 ± 1.22 / 2.80 ± 2.44 | -5.68 ± 4.85 [-8.24, -3.13], 0.0084 | -0.01 ± 3.15 [-1.64, 1.63], 1 |
| VNM Q, signed_last_90, uniform_both, 8 paths, constant scores | 58.76 ± 2.44 / 10.48 ± 0.67 / 30.76 ± 2.43 | 58.67 ± 3.77 | 48.60 ± 1.22 / 2.80 ± 2.44 | -5.68 ± 4.85 [-8.24, -3.13], 0.0084 | -0.01 ± 3.15 [-1.64, 1.63], 1 |
| Frozen original | 57.80 ± 4.47 / 18.80 ± 1.79 / 23.40 ± 4.47 | 100.00 ± 0.00 | 33.09 ± 0.31 / 33.82 ± 0.61 | -4.72 ± 3.92 [-6.66, -2.78], 0.00116 | -41.34 ± 1.30 [-41.95, -40.73], 1.44e-55 |
| VNM Q, signed_last_90, uniform_both, 8 paths | 53.08 ± 4.14 / 11.95 ± 1.15 / 34.97 ± 4.09 | 58.66 ± 1.30 | 46.59 ± 3.12 / 6.83 ± 6.25 | reference | reference |
| VNM V, signed_last_90, uniform_both, 8 paths | 52.47 ± 3.48 / 11.92 ± 0.85 / 35.61 ± 3.37 | 59.76 ± 1.65 | 45.71 ± 4.06 / 8.58 ± 8.12 | +0.61 ± 2.50 [-0.68, 1.91], 1 | -1.10 ± 1.00 [-1.57, -0.62], 0.0022 |
| VNM V, signed_last_90, mixed_reply, 8 paths | 51.74 ± 4.45 / 10.31 ± 1.83 / 37.95 ± 4.64 | 57.20 ± 1.86 | 46.45 ± 3.13 / 7.09 ± 6.26 | +1.34 ± 3.30 [-0.31, 3.00], 1 | +1.46 ± 1.51 [0.71, 2.21], 0.0198 |
| VNM V, signed final frequency, assisted opponent mixture | 51.17 ± 2.92 / 12.63 ± 1.45 / 36.20 ± 3.10 | 55.62 ± 1.81 | 46.35 ± 2.74 / 7.29 ± 5.48 | +1.91 ± 3.41 [0.21, 3.62], 0.671 | +3.04 ± 1.39 [2.35, 3.72], 1.35e-07 |
| VNM Q, signed_last, uniform_both, 8 paths | 49.72 ± 2.93 / 12.55 ± 1.22 / 37.72 ± 2.88 | 56.88 ± 1.32 | 47.28 ± 2.85 / 5.44 ± 5.70 | +3.36 ± 1.99 [2.33, 4.38], 5.45e-05 | +1.78 ± 0.55 [1.50, 2.06], 2.47e-09 |
| VNM Q, signed_90, copies, 8 paths | 47.15 ± 3.30 / 6.57 ± 2.87 / 46.28 ± 4.23 | 77.15 ± 3.62 | 45.94 ± 3.84 / 8.11 ± 7.68 | +5.93 ± 3.52 [4.18, 7.68], 4.54e-06 | -18.49 ± 4.13 [-20.65, -16.33], 1.4e-10 |
| VNM Q, signed_90, uniform_both, 8 paths | 44.28 ± 2.73 / 7.89 ± 3.52 / 47.83 ± 3.49 | 78.33 ± 3.66 | 47.32 ± 2.85 / 5.37 ± 5.70 | +8.80 ± 3.82 [6.93, 10.68], 2.26e-09 | -19.67 ± 4.18 [-21.86, -17.49], 6.6e-11 |
| Direct Q | 43.96 ± 3.01 / 5.96 ± 3.20 / 50.08 ± 3.54 | 77.15 ± 3.80 | 47.62 ± 2.45 / 4.76 ± 4.91 | +9.12 ± 4.54 [6.88, 11.35], 3.45e-07 | -18.49 ± 4.38 [-20.75, -16.22], 1.15e-10 |
| VNM Q, own_90, uniform_both, 8 paths | 40.62 ± 3.10 / 12.21 ± 4.02 / 47.17 ± 4.72 | 73.67 ± 4.18 | 36.93 ± 5.06 / 26.13 ± 10.12 | +12.46 ± 3.64 [10.67, 14.25], 3.58e-14 | -15.01 ± 4.71 [-17.46, -12.56], 2.05e-08 |
| Direct V | 37.33 ± 4.16 / 6.66 ± 2.98 / 56.01 ± 4.81 | 69.90 ± 1.96 | 47.39 ± 2.53 / 5.21 ± 5.06 | +15.75 ± 3.37 [14.10, 17.41], 4.39e-18 | -11.24 ± 2.44 [-12.42, -10.06], 1.5e-16 |
| VNM Q, unsigned_90, uniform_both, 8 paths | 29.35 ± 2.87 / 17.57 ± 3.75 / 53.08 ± 5.10 | 65.90 ± 3.95 | 35.02 ± 4.98 / 29.97 ± 9.96 | +23.73 ± 3.30 [22.08, 25.39], 5.54e-23 | -7.24 ± 4.51 [-9.58, -4.90], 0.000263 |
| VNM Q, unsigned_90, uniform_reply, 8 paths | 28.23 ± 2.07 / 17.79 ± 3.33 / 53.97 ± 4.04 | 56.36 ± 2.35 | 35.66 ± 5.71 / 28.68 ± 11.41 | +24.85 ± 3.21 [23.23, 26.47], 2.79e-20 | +2.30 ± 2.93 [0.82, 3.78], 0.124 |

## Scores on actual external games

Win minus loss subtracts each fitted player's mean utility on its lost games from the mean on its won games. Per turn is the within-fit least-squares change in utility per additional observed turn. 'unsigned_last' is the raw fitted frequency of the last played move. 'chooser_moved_last' is 1 if the evaluated seat made that move, otherwise 0. 'turns' counts the remaining turns including mandatory passes. Slopes and differences are averaged per original first. All diagnostic contrasts share a separate Holm adjustment. Missing outcomes are omitted from the corresponding fit-specific mean. Counts of observed wins, draws and losses and the fit-level values are in the path diagnostics file.

| Game (originals) | Measure | Outcome or association | Mean ± spread [95%], adjusted p for differences and slopes | Contributing fits |
|---|---|---|---|---|
| tictactoe (24) | chooser_moved_last | draw | +0.9673 ± 0.0624 [0.9409, 0.9936], — | 96 |
| tictactoe (24) | chooser_moved_last | loss | +0.0000 ± 0.0000 [0.0000, 0.0000], — | 96 |
| tictactoe (24) | chooser_moved_last | per_turn | +0.0191 ± 0.0252 [0.0085, 0.0298], 0.0136 | 96 |
| tictactoe (24) | chooser_moved_last | win | +1.0000 ± 0.0000 [1.0000, 1.0000], — | 96 |
| tictactoe (24) | chooser_moved_last | win_minus_loss | +1.0000 ± 0.0000 [1.0000, 1.0000], — | 96 |
| tictactoe (24) | signed_90 | draw | -0.0868 ± 0.1924 [-0.1680, -0.0055], — | 96 |
| tictactoe (24) | signed_90 | loss | -0.0539 ± 0.1217 [-0.1053, -0.0026], — | 96 |
| tictactoe (24) | signed_90 | per_turn | -0.0930 ± 0.0524 [-0.1151, -0.0708], 2.3e-07 | 96 |
| tictactoe (24) | signed_90 | win | +0.2552 ± 0.1120 [0.2079, 0.3025], — | 96 |
| tictactoe (24) | signed_90 | win_minus_loss | +0.3092 ± 0.1633 [0.2402, 0.3782], 7.47e-08 | 96 |
| tictactoe (24) | signed_last | draw | +0.5212 ± 0.1659 [0.4511, 0.5912], — | 96 |
| tictactoe (24) | signed_last | loss | -0.2776 ± 0.0685 [-0.3066, -0.2487], — | 96 |
| tictactoe (24) | signed_last | per_turn | +0.0183 ± 0.0351 [0.0035, 0.0332], 0.141 | 96 |
| tictactoe (24) | signed_last | win | +0.4920 ± 0.0244 [0.4816, 0.5023], — | 96 |
| tictactoe (24) | signed_last | win_minus_loss | +0.7696 ± 0.0765 [0.7373, 0.8019], 2.89e-23 | 96 |
| tictactoe (24) | signed_last_90 | draw | +0.2243 ± 0.0714 [0.1942, 0.2545], — | 96 |
| tictactoe (24) | signed_last_90 | loss | -0.1495 ± 0.0367 [-0.1650, -0.1340], — | 96 |
| tictactoe (24) | signed_last_90 | per_turn | -0.0167 ± 0.0162 [-0.0236, -0.0099], 0.000656 | 96 |
| tictactoe (24) | signed_last_90 | win | +0.2962 ± 0.0180 [0.2887, 0.3038], — | 96 |
| tictactoe (24) | signed_last_90 | win_minus_loss | +0.4458 ± 0.0427 [0.4277, 0.4638], 1.24e-23 | 96 |
| tictactoe (24) | turns | draw | +9.0000 ± 0.0000 [9.0000, 9.0000], — | 96 |
| tictactoe (24) | turns | loss | +6.7087 ± 0.3249 [6.5715, 6.8459], — | 96 |
| tictactoe (24) | turns | win | +5.9261 ± 0.1559 [5.8603, 5.9920], — | 96 |
| tictactoe (24) | turns | win_minus_loss | -0.7826 ± 0.3738 [-0.9405, -0.6248], 1.23e-08 | 96 |
| tictactoe (24) | unsigned_90 | draw | +1.3450 ± 0.1395 [1.2861, 1.4039], — | 96 |
| tictactoe (24) | unsigned_90 | loss | +0.8327 ± 0.1790 [0.7572, 0.9083], — | 96 |
| tictactoe (24) | unsigned_90 | per_turn | +0.1885 ± 0.0258 [0.1776, 0.1994], 4.04e-20 | 96 |
| tictactoe (24) | unsigned_90 | win | +0.7664 ± 0.1186 [0.7163, 0.8165], — | 96 |
| tictactoe (24) | unsigned_90 | win_minus_loss | -0.0663 ± 0.1492 [-0.1293, -0.0033], 0.239 | 96 |
| tictactoe (24) | unsigned_last | draw | +0.5374 ± 0.1498 [0.4742, 0.6007], — | 96 |
| tictactoe (24) | unsigned_last | loss | +0.2776 ± 0.0685 [0.2487, 0.3066], — | 96 |
| tictactoe (24) | unsigned_last | per_turn | +0.0157 ± 0.0321 [0.0021, 0.0292], 0.176 | 96 |
| tictactoe (24) | unsigned_last | win | +0.4920 ± 0.0244 [0.4816, 0.5023], — | 96 |
| tictactoe (24) | unsigned_last | win_minus_loss | +0.2143 ± 0.0688 [0.1853, 0.2434], 4.72e-12 | 96 |
| connect4 (16) | chooser_moved_last | loss | +0.0000 ± 0.0000 [0.0000, 0.0000], — | 64 |
| connect4 (16) | chooser_moved_last | per_turn | -0.0046 ± 0.0193 [-0.0149, 0.0057], 1 | 64 |
| connect4 (16) | chooser_moved_last | win | +1.0000 ± 0.0000 [1.0000, 1.0000], — | 64 |
| connect4 (16) | chooser_moved_last | win_minus_loss | +1.0000 ± 0.0000 [1.0000, 1.0000], — | 64 |
| connect4 (16) | signed_90 | loss | -0.2330 ± 0.9668 [-0.7482, 0.2822], — | 64 |
| connect4 (16) | signed_90 | per_turn | -0.0525 ± 0.0328 [-0.0700, -0.0350], 0.000217 | 64 |
| connect4 (16) | signed_90 | win | +0.1705 ± 0.5027 [-0.0974, 0.4384], — | 64 |
| connect4 (16) | signed_90 | win_minus_loss | +0.4036 ± 0.5307 [0.1208, 0.6864], 0.0906 | 64 |
| connect4 (16) | signed_last | loss | -0.1487 ± 0.1115 [-0.2081, -0.0893], — | 64 |
| connect4 (16) | signed_last | per_turn | -0.0143 ± 0.0135 [-0.0215, -0.0072], 0.009 | 64 |
| connect4 (16) | signed_last | win | +0.6228 ± 0.0426 [0.6001, 0.6455], — | 64 |
| connect4 (16) | signed_last | win_minus_loss | +0.7715 ± 0.1095 [0.7131, 0.8299], 6.92e-13 | 64 |
| connect4 (16) | signed_last_90 | loss | -0.0480 ± 0.0344 [-0.0664, -0.0297], — | 64 |
| connect4 (16) | signed_last_90 | per_turn | -0.0145 ± 0.0046 [-0.0170, -0.0121], 5.7e-08 | 64 |
| connect4 (16) | signed_last_90 | win | +0.2405 ± 0.0214 [0.2291, 0.2519], — | 64 |
| connect4 (16) | signed_last_90 | win_minus_loss | +0.2886 ± 0.0350 [0.2700, 0.3072], 6.89e-14 | 64 |
| connect4 (16) | turns | loss | +13.5685 ± 3.1735 [11.8775, 15.2595], — | 64 |
| connect4 (16) | turns | win | +12.3053 ± 0.8331 [11.8614, 12.7492], — | 64 |
| connect4 (16) | turns | win_minus_loss | -1.2632 ± 3.3627 [-3.0550, 0.5287], 0.769 | 64 |
| connect4 (16) | unsigned_90 | loss | +1.9898 ± 1.1146 [1.3959, 2.5837], — | 64 |
| connect4 (16) | unsigned_90 | per_turn | +0.0814 ± 0.0413 [0.0594, 0.1034], 2.27e-05 | 64 |
| connect4 (16) | unsigned_90 | win | +1.8403 ± 0.5106 [1.5682, 2.1123], — | 64 |
| connect4 (16) | unsigned_90 | win_minus_loss | -0.1496 ± 0.6898 [-0.5171, 0.2180], 1 | 64 |
| connect4 (16) | unsigned_last | loss | +0.1487 ± 0.1115 [0.0893, 0.2081], — | 64 |
| connect4 (16) | unsigned_last | per_turn | -0.0122 ± 0.0112 [-0.0182, -0.0063], 0.00778 | 64 |
| connect4 (16) | unsigned_last | win | +0.6228 ± 0.0426 [0.6001, 0.6455], — | 64 |
| connect4 (16) | unsigned_last | win_minus_loss | +0.4741 ± 0.1285 [0.4056, 0.5426], 6.87e-09 | 64 |
| reversi4 (16) | chooser_moved_last | draw | +0.9184 ± 0.0567 [0.8882, 0.9486], — | 63 |
| reversi4 (16) | chooser_moved_last | loss | +0.4910 ± 0.0667 [0.4554, 0.5265], — | 64 |
| reversi4 (16) | chooser_moved_last | per_turn | +0.0037 ± 0.0169 [-0.0053, 0.0127], 1 | 64 |
| reversi4 (16) | chooser_moved_last | win | +0.9858 ± 0.0092 [0.9810, 0.9907], — | 64 |
| reversi4 (16) | chooser_moved_last | win_minus_loss | +0.4949 ± 0.0713 [0.4569, 0.5329], 8.29e-13 | 64 |
| reversi4 (16) | signed_90 | draw | +0.0145 ± 0.2653 [-0.1269, 0.1559], — | 63 |
| reversi4 (16) | signed_90 | loss | -0.1650 ± 0.1053 [-0.2210, -0.1089], — | 64 |
| reversi4 (16) | signed_90 | per_turn | -0.0353 ± 0.0194 [-0.0456, -0.0249], 5.62e-05 | 64 |
| reversi4 (16) | signed_90 | win | -0.0416 ± 0.0934 [-0.0913, 0.0082], — | 64 |
| reversi4 (16) | signed_90 | win_minus_loss | +0.1234 ± 0.1676 [0.0341, 0.2127], 0.1 | 64 |
| reversi4 (16) | signed_last | draw | +0.7013 ± 0.0884 [0.6542, 0.7484], — | 63 |
| reversi4 (16) | signed_last | loss | +0.0312 ± 0.1058 [-0.0252, 0.0876], — | 64 |
| reversi4 (16) | signed_last | per_turn | +0.0076 ± 0.0212 [-0.0037, 0.0189], 0.769 | 64 |
| reversi4 (16) | signed_last | win | +0.7570 ± 0.0394 [0.7360, 0.7779], — | 64 |
| reversi4 (16) | signed_last | win_minus_loss | +0.7257 ± 0.1270 [0.6580, 0.7934], 1.31e-11 | 64 |
| reversi4 (16) | signed_last_90 | draw | +0.2592 ± 0.0411 [0.2373, 0.2811], — | 63 |
| reversi4 (16) | signed_last_90 | loss | +0.0086 ± 0.0364 [-0.0108, 0.0280], — | 64 |
| reversi4 (16) | signed_last_90 | per_turn | -0.0129 ± 0.0084 [-0.0174, -0.0084], 0.000307 | 64 |
| reversi4 (16) | signed_last_90 | win | +0.2899 ± 0.0187 [0.2800, 0.2999], — | 64 |
| reversi4 (16) | signed_last_90 | win_minus_loss | +0.2813 ± 0.0442 [0.2577, 0.3049], 2.94e-12 | 64 |
| reversi4 (16) | turns | draw | +10.3376 ± 0.5203 [10.0604, 10.6149], — | 63 |
| reversi4 (16) | turns | loss | +10.6421 ± 0.3108 [10.4765, 10.8077], — | 64 |
| reversi4 (16) | turns | win | +10.2728 ± 0.3319 [10.0959, 10.4497], — | 64 |
| reversi4 (16) | turns | win_minus_loss | -0.3693 ± 0.5383 [-0.6561, -0.0825], 0.135 | 64 |
| reversi4 (16) | unsigned_90 | draw | +3.1416 ± 0.2111 [3.0291, 3.2541], — | 63 |
| reversi4 (16) | unsigned_90 | loss | +3.5352 ± 0.1713 [3.4439, 3.6264], — | 64 |
| reversi4 (16) | unsigned_90 | per_turn | +0.1528 ± 0.0431 [0.1298, 0.1757], 1.18e-08 | 64 |
| reversi4 (16) | unsigned_90 | win | +3.0909 ± 0.2090 [2.9795, 3.2022], — | 64 |
| reversi4 (16) | unsigned_90 | win_minus_loss | -0.4443 ± 0.2640 [-0.5849, -0.3036], 0.000128 | 64 |
| reversi4 (16) | unsigned_last | draw | +0.8322 ± 0.0476 [0.8069, 0.8576], — | 63 |
| reversi4 (16) | unsigned_last | loss | +0.7496 ± 0.0303 [0.7334, 0.7658], — | 64 |
| reversi4 (16) | unsigned_last | per_turn | +0.0083 ± 0.0048 [0.0057, 0.0108], 0.000111 | 64 |
| reversi4 (16) | unsigned_last | win | +0.7776 ± 0.0353 [0.7588, 0.7964], — | 64 |
| reversi4 (16) | unsigned_last | win_minus_loss | +0.0280 ± 0.0243 [0.0150, 0.0410], 0.00515 | 64 |

## Network fitting

| Game (originals) | Network | Training mean squared frequency error | Fitting boards | Fitting legal candidates | Distinct observed decision boards |
|---|---|---|---|---|---|
| tictactoe (24) | V | 0.1676 ± 0.0048 | 2287.6 ± 113.9 | 4681.6 ± 318.5 | 4520.0 ± 0.0 |
| tictactoe (24) | Q | 0.1607 ± 0.0055 | 2942.7 ± 62.1 | 10543.4 ± 217.6 | 4520.0 ± 0.0 |
| connect4 (16) | V | 0.1155 ± 0.0051 | 2720.0 ± 523.6 | 14919.2 ± 2649.9 | 4187.5 ± 797.7 |
| connect4 (16) | Q | 0.0569 ± 0.0133 | 2720.0 ± 523.6 | 15263.3 ± 2640.3 | 4187.5 ± 797.7 |
| reversi4 (16) | V | 0.1279 ± 0.0090 | 727.7 ± 31.6 | 1370.0 ± 73.4 | 1188.4 ± 24.5 |
| reversi4 (16) | Q | 0.0605 ± 0.0137 | 776.0 ± 31.3 | 1473.8 ± 71.1 | 1188.4 ± 24.5 |
