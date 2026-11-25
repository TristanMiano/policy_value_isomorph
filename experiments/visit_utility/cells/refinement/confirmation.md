# Confirmation results for visit-trained game scores

The frozen original takes its highest-scoring legal move. A network scoring the resulting board (V) or the board and move (Q) estimates the fraction of observed visits taking a move. Direct selection takes the largest immediate fit. Von Neumann–Morgenstern (VNM) selection maximizes expected complete-game score, replanning on each actual turn. Ties prefer the lowest move index. 'copies' projects highest fitted choices for both seats. The rule 'uniform_reply' projects greedy own moves against uniform legal replies. The rule 'mixed_reply' mixes those two opponents equally once per path. The rule 'uniform_both' projects uniform legal moves for both seats. Each random component gets the listed number of full paths, with exact expectations in tic-tac-toe. The assisted comparison includes an opponent considering tactics and one using positional priorities. The supplied-outcome reference adds actual game results and projects guarded own play against four opponent types.

The utility name specifies the score of a completed game from the current choosing seat. 'signed' counts own moves positively and opposing moves negatively, 'unsigned' counts both positively, and 'own' counts only own moves. 'last' counts the final fitted frequency. The rule 'sum' adds the whole path. The suffix 50, 75, 90 or 97 gives the discount per turn in percent, starting at weight one. 'sqrt_last' and 'square_last' apply the square root or square to the final frequency, signed by mover. 'last_plus_sum' adds the signed last frequency and 0.1 times the signed 0.9-discounted sum. A constant score uses 0.5 in the identical formula. Constant continuations also break all greedy choices by move index. Both constants induce the identical player when all continuing moves are uniformly random.

Win, draw and loss percentages are seat-balanced against four equally weighted external opponent groups. Agreement is the weighted percentage of withheld boards on which the reconstruction chose the original's favorite. Exact-copy self-play uses two identical deployed players and counts both seats. Values give means ± sample standard deviations across originals after averaging their four fits. The equal-game spread describes the average of one independent original per named game. The selected rule is VNM Q, sqrt_last_90, uniform_both, 64 paths.

Independent original counts for tic-tac-toe, Connect Four and 4×4 Reversi are 24, 16, 16.
For paired changes, brackets give individual 95% confidence intervals, followed by the two-sided p-value for a zero mean difference, Holm-adjusted across the table's win and agreement comparisons in all four populations. A dash means the across-original variance is exactly zero. Discovery contrasts are exploratory because the leader was chosen on those data. The compressed estimates and contrasts alongside this table contain all unrounded measurements, raw p-values, original-level differences, and intervals resampling the independent originals.

## External win rate (%)

| Rule | Tic-tac-toe | Connect Four | 4×4 Reversi | Equal games |
|---|---|---|---|---|
| VNM V, supplied outcomes and frequencies, guarded mixture | 50.38 ± 4.34 | 91.36 ± 3.03 | 77.34 ± 2.08 | 73.03 ± 1.89 |
| VNM Q, sqrt_last_90, uniform_both, 64 paths, constant scores | 46.67 ± 1.65 | 94.36 ± 1.99 | 55.64 ± 3.02 | 65.56 ± 1.32 |
| VNM Q, sqrt_last_90, uniform_both, 64 paths, constant scores and continuations | 46.67 ± 1.65 | 94.36 ± 1.99 | 55.64 ± 3.02 | 65.56 ± 1.32 |
| VNM V, sqrt_last_90, uniform_both, 64 paths | 46.75 ± 1.47 | 84.38 ± 8.85 | 51.10 ± 4.49 | 60.74 ± 3.34 |
| VNM Q, sqrt_last_90, uniform_both, 64 paths | 46.58 ± 1.41 | 82.62 ± 9.09 | 52.64 ± 4.79 | 60.61 ± 3.46 |
| VNM Q, signed_last_90, uniform_both, 64 paths | 45.69 ± 1.55 | 78.59 ± 10.88 | 50.05 ± 4.96 | 58.11 ± 4.02 |
| VNM Q, signed_last, uniform_both, 64 paths | 43.91 ± 1.60 | 76.95 ± 10.82 | 51.20 ± 4.51 | 57.35 ± 3.94 |
| Frozen original | 44.67 ± 5.80 | 59.30 ± 11.40 | 67.80 ± 2.75 | 57.26 ± 4.36 |
| VNM Q, sqrt_last_90, uniform_both, 8 paths | 46.58 ± 1.41 | 71.41 ± 9.27 | 45.68 ± 7.47 | 54.56 ± 4.00 |
| VNM Q, square_last_90, uniform_both, 64 paths | 44.57 ± 1.60 | 73.49 ± 9.94 | 44.95 ± 4.49 | 54.33 ± 3.68 |
| VNM Q, signed_last_90, uniform_both, 8 paths | 45.69 ± 1.55 | 68.55 ± 10.08 | 44.70 ± 6.39 | 52.98 ± 4.01 |
| VNM V, signed final frequency, assisted opponent mixture | 40.32 ± 2.59 | 73.49 ± 6.03 | 40.77 ± 5.46 | 51.53 ± 2.85 |
| VNM Q, signed_90, uniform_both, 64 paths | 37.43 ± 4.04 | 53.78 ± 5.60 | 41.16 ± 4.91 | 44.13 ± 2.82 |
| Direct Q | 29.35 ± 5.29 | 51.42 ± 4.72 | 50.81 ± 3.58 | 43.86 ± 2.65 |
| VNM Q, own_90, uniform_both, 64 paths | 26.38 ± 2.96 | 47.34 ± 8.34 | 49.24 ± 4.40 | 40.99 ± 3.29 |
| Direct V | 28.94 ± 3.75 | 37.62 ± 12.19 | 44.48 ± 6.62 | 37.01 ± 4.79 |
| VNM Q, unsigned_90, uniform_both, 64 paths | 15.87 ± 2.15 | 22.92 ± 9.18 | 51.54 ± 3.19 | 30.11 ± 3.32 |

## Tic-tac-toe

| Rule | External wins / draws / losses (%) | Move agreement (%) | Exact-copy wins / draws (%) | Selected minus row in wins, points [95%], p | Selected minus row in agreement, points [95%], p |
|---|---|---|---|---|---|
| VNM V, supplied outcomes and frequencies, guarded mixture | 50.38 ± 4.34 / 28.62 ± 9.52 / 21.00 ± 8.84 | 76.77 ± 2.65 | 35.94 ± 12.93 / 28.12 ± 25.87 | -3.80 ± 4.31 [-5.62, -1.98], 0.0113 | -11.36 ± 2.53 [-12.43, -10.29], 6.97e-15 |
| VNM Q, sqrt_last_90, uniform_both, 64 paths, constant scores | 46.67 ± 1.65 / 25.50 ± 0.84 / 27.83 ± 1.75 | 62.57 ± 3.08 | 50.00 ± 0.00 / 0.00 ± 0.00 | -0.09 ± 1.72 [-0.81, 0.64], 1 | +2.84 ± 2.88 [1.63, 4.06], 0.00349 |
| VNM Q, sqrt_last_90, uniform_both, 64 paths, constant scores and continuations | 46.67 ± 1.65 / 25.50 ± 0.84 / 27.83 ± 1.75 | 62.57 ± 3.08 | 50.00 ± 0.00 / 0.00 ± 0.00 | -0.09 ± 1.72 [-0.81, 0.64], 1 | +2.84 ± 2.88 [1.63, 4.06], 0.00349 |
| VNM V, sqrt_last_90, uniform_both, 64 paths | 46.75 ± 1.47 / 25.94 ± 0.73 / 27.31 ± 1.47 | 65.50 ± 3.07 | 50.00 ± 0.00 / 0.00 ± 0.00 | -0.17 ± 1.14 [-0.65, 0.32], 1 | -0.09 ± 1.15 [-0.58, 0.39], 1 |
| VNM Q, sqrt_last_90, uniform_both, 64 paths | 46.58 ± 1.41 / 26.12 ± 1.27 / 27.30 ± 1.36 | 65.41 ± 2.57 | 47.92 ± 4.76 / 4.17 ± 9.52 | reference | reference |
| VNM Q, signed_last_90, uniform_both, 64 paths | 45.69 ± 1.55 / 27.64 ± 2.91 / 26.67 ± 1.99 | 64.10 ± 2.51 | 43.23 ± 9.01 / 13.54 ± 18.03 | +0.89 ± 1.05 [0.45, 1.33], 0.0156 | +1.31 ± 0.64 [1.04, 1.58], 5.71e-08 |
| VNM Q, signed_last, uniform_both, 64 paths | 43.91 ± 1.60 / 28.79 ± 2.81 / 27.31 ± 1.89 | 61.32 ± 2.60 | 45.31 ± 8.09 / 9.38 ± 16.17 | +2.68 ± 1.40 [2.08, 3.27], 2.1e-07 | +4.09 ± 1.00 [3.66, 4.51], 5.95e-14 |
| Frozen original | 44.67 ± 5.80 / 55.33 ± 5.80 / 0.00 ± 0.00 | 100.00 ± 0.00 | 0.00 ± 0.00 / 100.00 ± 0.00 | +1.92 ± 5.98 [-0.61, 4.44], 1 | -34.59 ± 2.57 [-35.67, -33.51], 1.26e-25 |
| VNM Q, sqrt_last_90, uniform_both, 8 paths | 46.58 ± 1.41 / 26.12 ± 1.27 / 27.30 ± 1.36 | 65.41 ± 2.57 | 47.92 ± 4.76 / 4.17 ± 9.52 | +0.00 ± 0.00 [0.00, 0.00], — | +0.00 ± 0.00 [0.00, 0.00], — |
| VNM Q, square_last_90, uniform_both, 64 paths | 44.57 ± 1.60 / 25.87 ± 5.58 / 29.56 ± 5.31 | 62.02 ± 2.51 | 43.75 ± 8.24 / 12.50 ± 16.48 | +2.01 ± 1.69 [1.30, 2.72], 0.000365 | +3.39 ± 0.97 [2.98, 3.80], 1.45e-12 |
| VNM Q, signed_last_90, uniform_both, 8 paths | 45.69 ± 1.55 / 27.64 ± 2.91 / 26.67 ± 1.99 | 64.10 ± 2.51 | 43.23 ± 9.01 / 13.54 ± 18.03 | +0.89 ± 1.05 [0.45, 1.33], 0.0156 | +1.31 ± 0.64 [1.04, 1.58], 5.71e-08 |
| VNM V, signed final frequency, assisted opponent mixture | 40.32 ± 2.59 / 28.54 ± 3.78 / 31.14 ± 2.59 | 58.46 ± 3.13 | 43.75 ± 7.37 / 12.50 ± 14.74 | +6.26 ± 3.31 [4.86, 7.66], 2.53e-07 | +6.96 ± 1.91 [6.15, 7.76], 6.2e-13 |
| VNM Q, signed_90, uniform_both, 64 paths | 37.43 ± 4.04 / 16.44 ± 10.69 / 46.13 ± 8.51 | 65.56 ± 2.23 | 45.83 ± 7.96 / 8.33 ± 15.93 | +9.15 ± 4.13 [7.41, 10.90], 1.46e-08 | -0.15 ± 3.03 [-1.43, 1.13], 1 |
| Direct Q | 29.35 ± 5.29 / 12.54 ± 9.61 / 58.11 ± 8.30 | 56.96 ± 3.43 | 44.79 ± 7.29 / 10.42 ± 14.59 | +17.23 ± 5.46 [14.93, 19.54], 1.28e-11 | +8.45 ± 3.98 [6.77, 10.13], 3.24e-08 |
| VNM Q, own_90, uniform_both, 64 paths | 26.38 ± 2.96 / 29.22 ± 10.92 / 44.40 ± 10.41 | 50.61 ± 3.09 | 25.52 ± 12.49 / 48.96 ± 24.98 | +20.20 ± 3.06 [18.91, 21.49], 1.38e-18 | +14.80 ± 3.82 [13.19, 16.41], 1.69e-13 |
| Direct V | 28.94 ± 3.75 / 14.60 ± 9.59 / 56.46 ± 8.91 | 53.45 ± 3.52 | 44.79 ± 7.29 / 10.42 ± 14.59 | +17.64 ± 4.10 [15.92, 19.37], 1.72e-14 | +11.96 ± 3.86 [10.33, 13.59], 1.89e-11 |
| VNM Q, unsigned_90, uniform_both, 64 paths | 15.87 ± 2.15 / 43.28 ± 8.90 / 40.85 ± 8.62 | 37.62 ± 2.11 | 24.48 ± 11.93 / 51.04 ± 23.86 | +30.72 ± 2.60 [29.62, 31.81], 2.42e-24 | +27.79 ± 3.68 [26.24, 29.34], 6.66e-20 |

## Connect Four

| Rule | External wins / draws / losses (%) | Move agreement (%) | Exact-copy wins / draws (%) | Selected minus row in wins, points [95%], p | Selected minus row in agreement, points [95%], p |
|---|---|---|---|---|---|
| VNM V, supplied outcomes and frequencies, guarded mixture | 91.36 ± 3.03 / 0.66 ± 0.74 / 7.98 ± 3.01 | 30.83 ± 6.73 | 49.71 ± 0.33 / 0.59 ± 0.67 | -8.74 ± 8.24 [-13.13, -4.35], 0.0285 | +5.27 ± 5.61 [2.28, 8.26], 0.0627 |
| VNM Q, sqrt_last_90, uniform_both, 64 paths, constant scores | 94.36 ± 1.99 / 0.66 ± 0.76 / 4.98 ± 1.73 | 28.76 ± 13.29 | 46.80 ± 3.04 / 6.40 ± 6.09 | -11.74 ± 9.39 [-16.75, -6.74], 0.00744 | +7.34 ± 10.43 [1.79, 12.90], 0.366 |
| VNM Q, sqrt_last_90, uniform_both, 64 paths, constant scores and continuations | 94.36 ± 1.99 / 0.66 ± 0.76 / 4.98 ± 1.73 | 28.76 ± 13.29 | 46.80 ± 3.04 / 6.40 ± 6.09 | -11.74 ± 9.39 [-16.75, -6.74], 0.00744 | +7.34 ± 10.43 [1.79, 12.90], 0.366 |
| VNM V, sqrt_last_90, uniform_both, 64 paths | 84.38 ± 8.85 / 0.34 ± 0.53 / 15.28 ± 9.11 | 37.65 ± 6.54 | 48.75 ± 1.92 / 2.49 ± 3.84 | -1.76 ± 5.32 [-4.59, 1.08], 1 | -1.54 ± 2.46 [-2.85, -0.23], 0.604 |
| VNM Q, sqrt_last_90, uniform_both, 64 paths | 82.62 ± 9.09 / 0.27 ± 0.51 / 17.11 ± 9.15 | 36.10 ± 5.16 | 48.73 ± 1.51 / 2.54 ± 3.01 | reference | reference |
| VNM Q, signed_last_90, uniform_both, 64 paths | 78.59 ± 10.88 / 0.24 ± 0.49 / 21.17 ± 10.91 | 39.20 ± 5.15 | 49.15 ± 1.68 / 1.71 ± 3.35 | +4.03 ± 3.91 [1.94, 6.11], 0.0356 | -3.10 ± 3.22 [-4.81, -1.38], 0.0573 |
| VNM Q, signed_last, uniform_both, 64 paths | 76.95 ± 10.82 / 0.07 ± 0.16 / 22.97 ± 10.89 | 37.02 ± 5.26 | 49.15 ± 1.78 / 1.71 ± 3.55 | +5.66 ± 5.22 [2.88, 8.44], 0.0237 | -0.92 ± 3.34 [-2.70, 0.86], 1 |
| Frozen original | 59.30 ± 11.40 / 0.76 ± 2.17 / 39.94 ± 12.73 | 100.00 ± 0.00 | 49.49 ± 0.75 / 1.03 ± 1.50 | +23.32 ± 9.22 [18.40, 28.23], 3.15e-06 | -63.90 ± 5.16 [-66.65, -61.15], 5.7e-16 |
| VNM Q, sqrt_last_90, uniform_both, 8 paths | 71.41 ± 9.27 / 0.02 ± 0.10 / 28.56 ± 9.30 | 30.42 ± 4.86 | 49.85 ± 0.24 / 0.29 ± 0.48 | +11.21 ± 4.98 [8.55, 13.86], 1.37e-05 | +5.68 ± 3.53 [3.80, 7.57], 0.000632 |
| VNM Q, square_last_90, uniform_both, 64 paths | 73.49 ± 9.94 / 0.12 ± 0.19 / 26.39 ± 9.92 | 42.35 ± 5.98 | 49.29 ± 1.56 / 1.42 ± 3.11 | +9.13 ± 3.43 [7.30, 10.96], 1.66e-06 | -6.25 ± 6.24 [-9.57, -2.93], 0.0433 |
| VNM Q, signed_last_90, uniform_both, 8 paths | 68.55 ± 10.08 / 0.05 ± 0.13 / 31.40 ± 10.05 | 31.04 ± 3.33 | 49.78 ± 0.28 / 0.44 ± 0.57 | +14.06 ± 4.91 [11.44, 16.68], 6.38e-07 | +5.07 ± 3.86 [3.01, 7.13], 0.0047 |
| VNM V, signed final frequency, assisted opponent mixture | 73.49 ± 6.03 / 0.66 ± 0.58 / 25.85 ± 6.22 | 25.06 ± 4.23 | 49.63 ± 0.30 / 0.73 ± 0.60 | +9.13 ± 6.65 [5.59, 12.67], 0.00314 | +11.04 ± 4.61 [8.59, 13.50], 6.36e-06 |
| VNM Q, signed_90, uniform_both, 64 paths | 53.78 ± 5.60 / 0.02 ± 0.10 / 46.19 ± 5.62 | 87.84 ± 10.17 | 48.88 ± 2.79 / 2.25 ± 5.57 | +28.83 ± 8.12 [24.51, 33.16], 3.68e-08 | -51.74 ± 11.29 [-57.75, -45.72], 1.06e-09 |
| Direct Q | 51.42 ± 4.72 / 0.00 ± 0.00 / 48.58 ± 4.72 | 87.97 ± 10.79 | 49.73 ± 0.34 / 0.54 ± 0.68 | +31.20 ± 8.89 [26.46, 35.94], 4.28e-08 | -51.87 ± 12.01 [-58.27, -45.46], 2.49e-09 |
| VNM Q, own_90, uniform_both, 64 paths | 47.34 ± 8.34 / 1.54 ± 2.74 / 51.12 ± 9.80 | 86.10 ± 12.04 | 37.77 ± 11.11 / 24.46 ± 22.22 | +35.28 ± 9.05 [30.46, 40.10], 1.03e-08 | -50.00 ± 13.21 [-57.04, -42.96], 1.53e-08 |
| Direct V | 37.62 ± 12.19 / 0.27 ± 0.62 / 62.11 ± 12.39 | 71.97 ± 4.49 | 49.22 ± 0.91 / 1.56 ± 1.83 | +45.00 ± 8.89 [40.26, 49.73], 2.62e-10 | -35.87 ± 6.37 [-39.26, -32.48], 5.72e-11 |
| VNM Q, unsigned_90, uniform_both, 64 paths | 22.92 ± 9.18 / 5.91 ± 8.32 / 71.17 ± 15.31 | 81.73 ± 12.14 | 32.23 ± 12.03 / 35.55 ± 24.05 | +59.69 ± 6.67 [56.14, 63.25], 6.93e-14 | -45.62 ± 13.28 [-52.70, -38.55], 5.71e-08 |

## 4×4 Reversi

| Rule | External wins / draws / losses (%) | Move agreement (%) | Exact-copy wins / draws (%) | Selected minus row in wins, points [95%], p | Selected minus row in agreement, points [95%], p |
|---|---|---|---|---|---|
| VNM V, supplied outcomes and frequencies, guarded mixture | 77.34 ± 2.08 / 1.78 ± 1.23 / 20.87 ± 2.35 | 86.16 ± 0.96 | 49.80 ± 0.29 / 0.39 ± 0.57 | -24.71 ± 4.88 [-27.31, -22.11], 2.62e-10 | -5.85 ± 1.74 [-6.78, -4.92], 7.39e-08 |
| VNM Q, sqrt_last_90, uniform_both, 64 paths, constant scores | 55.64 ± 3.02 / 4.66 ± 1.76 / 39.70 ± 2.33 | 86.57 ± 1.21 | 47.85 ± 2.06 / 4.30 ± 4.11 | -3.00 ± 3.12 [-4.67, -1.34], 0.0573 | -6.25 ± 1.22 [-6.90, -5.60], 2.22e-10 |
| VNM Q, sqrt_last_90, uniform_both, 64 paths, constant scores and continuations | 55.64 ± 3.02 / 4.66 ± 1.76 / 39.70 ± 2.33 | 86.57 ± 1.21 | 47.85 ± 2.06 / 4.30 ± 4.11 | -3.00 ± 3.12 [-4.67, -1.34], 0.0573 | -6.25 ± 1.22 [-6.90, -5.60], 2.22e-10 |
| VNM V, sqrt_last_90, uniform_both, 64 paths | 51.10 ± 4.49 / 7.50 ± 2.87 / 41.41 ± 2.97 | 82.39 ± 1.45 | 46.48 ± 3.26 / 7.03 ± 6.52 | +1.54 ± 2.21 [0.36, 2.72], 0.366 | -2.08 ± 1.23 [-2.74, -1.43], 0.000388 |
| VNM Q, sqrt_last_90, uniform_both, 64 paths | 52.64 ± 4.79 / 7.42 ± 2.97 / 39.94 ± 3.11 | 80.31 ± 1.66 | 47.27 ± 2.45 / 5.47 ± 4.91 | reference | reference |
| VNM Q, signed_last_90, uniform_both, 64 paths | 50.05 ± 4.96 / 8.54 ± 3.21 / 41.41 ± 3.28 | 80.39 ± 1.55 | 46.78 ± 2.72 / 6.45 ± 5.43 | +2.59 ± 3.66 [0.64, 4.54], 0.366 | -0.08 ± 0.34 [-0.26, 0.10], 1 |
| VNM Q, signed_last, uniform_both, 64 paths | 51.20 ± 4.51 / 8.33 ± 2.73 / 40.48 ± 3.20 | 80.37 ± 1.47 | 46.97 ± 2.79 / 6.05 ± 5.57 | +1.44 ± 2.95 [-0.13, 3.01], 1 | -0.06 ± 0.56 [-0.36, 0.24], 1 |
| Frozen original | 67.80 ± 2.75 / 1.51 ± 0.45 / 30.69 ± 2.72 | 100.00 ± 0.00 | 49.71 ± 0.30 / 0.59 ± 0.61 | -15.16 ± 5.22 [-17.94, -12.38], 5.33e-07 | -19.69 ± 1.66 [-20.57, -18.80], 1.06e-15 |
| VNM Q, sqrt_last_90, uniform_both, 8 paths | 45.68 ± 7.47 / 7.47 ± 2.62 / 46.85 ± 6.32 | 80.12 ± 1.57 | 45.26 ± 3.07 / 9.47 ± 6.14 | +6.96 ± 7.85 [2.78, 11.14], 0.0908 | +0.20 ± 0.46 [-0.05, 0.44], 1 |
| VNM Q, square_last_90, uniform_both, 64 paths | 44.95 ± 4.49 / 9.08 ± 2.85 / 45.97 ± 2.68 | 80.27 ± 1.55 | 46.09 ± 4.27 / 7.81 ± 8.53 | +7.69 ± 4.11 [5.50, 9.88], 0.000129 | +0.05 ± 0.52 [-0.23, 0.33], 1 |
| VNM Q, signed_last_90, uniform_both, 8 paths | 44.70 ± 6.39 / 7.67 ± 2.39 / 47.63 ± 5.73 | 80.14 ± 1.49 | 45.65 ± 3.04 / 8.69 ± 6.07 | +7.93 ± 7.08 [4.16, 11.71], 0.0185 | +0.17 ± 0.54 [-0.12, 0.46], 1 |
| VNM V, signed final frequency, assisted opponent mixture | 40.77 ± 5.46 / 8.64 ± 3.62 / 50.59 ± 4.66 | 82.57 ± 1.00 | 46.31 ± 4.38 / 7.37 ± 8.77 | +11.87 ± 7.73 [7.74, 15.99], 0.00103 | -2.26 ± 1.41 [-3.02, -1.51], 0.000647 |
| VNM Q, signed_90, uniform_both, 64 paths | 41.16 ± 4.91 / 6.20 ± 2.20 / 52.64 ± 5.00 | 83.98 ± 1.04 | 47.02 ± 1.85 / 5.96 ± 3.71 | +11.47 ± 7.73 [7.36, 15.59], 0.00142 | -3.67 ± 1.95 [-4.71, -2.63], 0.000122 |
| Direct Q | 50.81 ± 3.58 / 4.93 ± 1.67 / 44.26 ± 3.21 | 86.51 ± 1.29 | 48.39 ± 0.97 / 3.22 ± 1.93 | +1.83 ± 5.55 [-1.13, 4.79], 1 | -6.19 ± 2.18 [-7.36, -5.03], 7.17e-07 |
| VNM Q, own_90, uniform_both, 64 paths | 49.24 ± 4.40 / 5.03 ± 1.34 / 45.73 ± 3.70 | 85.69 ± 1.55 | 48.12 ± 1.97 / 3.76 ± 3.93 | +3.39 ± 7.24 [-0.46, 7.25], 1 | -5.38 ± 2.09 [-6.50, -4.27], 2.56e-06 |
| Direct V | 44.48 ± 6.62 / 5.05 ± 2.21 / 50.46 ± 5.72 | 84.28 ± 1.42 | 47.97 ± 2.09 / 4.05 ± 4.18 | +8.15 ± 9.39 [3.15, 13.16], 0.102 | -3.97 ± 2.46 [-5.28, -2.66], 0.000609 |
| VNM Q, unsigned_90, uniform_both, 64 paths | 51.54 ± 3.19 / 5.05 ± 1.95 / 43.41 ± 3.57 | 83.72 ± 1.51 | 49.19 ± 0.75 / 1.61 ± 1.50 | +1.10 ± 4.61 [-1.36, 3.56], 1 | -3.41 ± 1.89 [-4.42, -2.40], 0.0002 |

## Equal games

| Rule | External wins / draws / losses (%) | Move agreement (%) | Exact-copy wins / draws (%) | Selected minus row in wins, points [95%], p | Selected minus row in agreement, points [95%], p |
|---|---|---|---|---|---|
| VNM V, supplied outcomes and frequencies, guarded mixture | 73.03 ± 1.89 / 10.35 ± 3.21 / 16.62 ± 3.21 | 64.59 ± 2.43 | 45.15 ± 4.31 / 9.70 ± 8.63 | -12.42 ± 3.50 [-14.15, -10.68], 2.48e-13 | -3.98 ± 2.13 [-5.05, -2.90], 7.8e-06 |
| VNM Q, sqrt_last_90, uniform_both, 64 paths, constant scores | 65.56 ± 1.32 / 10.27 ± 0.70 / 24.17 ± 1.13 | 59.30 ± 4.57 | 48.22 ± 1.22 / 3.56 ± 2.45 | -4.94 ± 3.35 [-6.69, -3.20], 0.000609 | +1.31 ± 3.63 [-0.58, 3.20], 1 |
| VNM Q, sqrt_last_90, uniform_both, 64 paths, constant scores and continuations | 65.56 ± 1.32 / 10.27 ± 0.70 / 24.17 ± 1.13 | 59.30 ± 4.57 | 48.22 ± 1.22 / 3.56 ± 2.45 | -4.94 ± 3.35 [-6.69, -3.20], 0.000609 | +1.31 ± 3.63 [-0.58, 3.20], 1 |
| VNM V, sqrt_last_90, uniform_both, 64 paths | 60.74 ± 3.34 / 11.26 ± 1.00 / 28.00 ± 3.23 | 61.85 ± 2.46 | 48.41 ± 1.26 / 3.17 ± 2.52 | -0.13 ± 1.96 [-1.14, 0.88], 1 | -1.24 ± 0.99 [-1.74, -0.74], 0.00117 |
| VNM Q, sqrt_last_90, uniform_both, 64 paths | 60.61 ± 3.46 / 11.27 ± 1.09 / 28.12 ± 3.25 | 60.61 ± 2.00 | 47.97 ± 1.85 / 4.06 ± 3.71 | reference | reference |
| VNM Q, signed_last_90, uniform_both, 64 paths | 58.11 ± 4.02 / 12.14 ± 1.45 / 29.75 ± 3.86 | 61.23 ± 1.98 | 46.38 ± 3.19 / 7.23 ± 6.37 | +2.50 ± 1.82 [1.58, 3.42], 0.000287 | -0.62 ± 1.10 [-1.20, -0.04], 0.892 |
| VNM Q, signed_last, uniform_both, 64 paths | 57.35 ± 3.94 / 12.39 ± 1.31 / 30.25 ± 3.84 | 59.57 ± 2.02 | 47.14 ± 2.91 / 5.71 ± 5.82 | +3.26 ± 2.05 [2.21, 4.31], 6.55e-05 | +1.04 ± 1.18 [0.42, 1.65], 0.0724 |
| Frozen original | 57.26 ± 4.36 / 19.20 ± 2.07 / 23.54 ± 4.34 | 100.00 ± 0.00 | 33.06 ± 0.27 / 33.87 ± 0.54 | +3.36 ± 4.06 [1.38, 5.33], 0.0568 | -39.39 ± 2.00 [-40.39, -38.39], 3.6e-28 |
| VNM Q, sqrt_last_90, uniform_both, 8 paths | 54.56 ± 4.00 / 11.21 ± 0.97 / 34.24 ± 3.77 | 58.65 ± 1.91 | 47.68 ± 1.89 / 4.64 ± 3.78 | +6.05 ± 3.10 [4.46, 7.65], 2.4e-06 | +1.96 ± 1.19 [1.33, 2.59], 0.000438 |
| VNM Q, square_last_90, uniform_both, 64 paths | 54.33 ± 3.68 / 11.69 ± 2.09 / 33.97 ± 3.86 | 61.55 ± 2.22 | 46.38 ± 3.14 / 7.24 ± 6.27 | +6.28 ± 1.87 [5.34, 7.22], 4.91e-13 | -0.94 ± 2.11 [-2.05, 0.18], 1 |
| VNM Q, signed_last_90, uniform_both, 8 paths | 52.98 ± 4.01 / 11.78 ± 1.26 / 35.23 ± 3.91 | 58.43 ± 1.47 | 46.22 ± 3.17 / 7.56 ± 6.34 | +7.63 ± 2.89 [6.15, 9.11], 3.59e-09 | +2.18 ± 1.32 [1.49, 2.88], 0.000337 |
| VNM V, signed final frequency, assisted opponent mixture | 51.53 ± 2.85 / 12.61 ± 1.75 / 35.86 ± 2.73 | 55.36 ± 1.78 | 46.57 ± 2.86 / 6.87 ± 5.72 | +9.09 ± 3.57 [7.30, 10.87], 5.95e-10 | +5.25 ± 1.73 [4.37, 6.12], 2.65e-09 |
| VNM Q, signed_90, uniform_both, 64 paths | 44.13 ± 2.82 / 7.56 ± 3.64 / 48.32 ± 3.79 | 79.13 ± 3.49 | 47.24 ± 2.88 / 5.51 ± 5.76 | +16.49 ± 3.98 [14.51, 18.47], 2.14e-16 | -18.52 ± 3.95 [-20.58, -16.46], 4.86e-11 |
| Direct Q | 43.86 ± 2.65 / 5.82 ± 3.25 / 50.32 ± 3.36 | 77.15 ± 3.80 | 47.64 ± 2.46 / 4.73 ± 4.91 | +16.75 ± 3.94 [14.83, 18.68], 1.42e-16 | -16.54 ± 4.28 [-18.75, -14.33], 4.58e-10 |
| VNM Q, own_90, uniform_both, 64 paths | 40.99 ± 3.29 / 11.93 ± 3.78 / 47.08 ± 4.92 | 74.14 ± 4.18 | 37.14 ± 5.61 / 25.73 ± 11.22 | +19.62 ± 3.99 [17.61, 21.64], 3.28e-17 | -13.53 ± 4.64 [-15.94, -11.12], 7.49e-08 |
| Direct V | 37.01 ± 4.79 / 6.64 ± 3.29 / 56.34 ± 5.43 | 69.90 ± 1.96 | 47.33 ± 2.55 / 5.34 ± 5.10 | +23.60 ± 4.52 [21.34, 25.86], 4.99e-19 | -9.29 ± 2.61 [-10.58, -8.01], 1.32e-12 |
| VNM Q, unsigned_90, uniform_both, 64 paths | 30.11 ± 3.32 / 18.08 ± 4.11 / 51.81 ± 5.98 | 67.69 ± 4.14 | 35.30 ± 5.65 / 29.40 ± 11.31 | +30.50 ± 2.84 [29.08, 31.93], 8.64e-27 | -7.08 ± 4.64 [-9.50, -4.67], 0.000574 |

## Scores on actual external games

Win minus loss subtracts each fitted player's mean utility on its lost games from the mean on its won games. Per turn is the within-fit least-squares change in utility per additional observed turn. 'unsigned_last' is the raw fitted frequency of the last played move. 'chooser_moved_last' is 1 if the evaluated seat made that move, otherwise 0. 'turns' counts the remaining turns including mandatory passes. Slopes and differences are averaged per original first. All diagnostic contrasts share a separate Holm adjustment. Missing outcomes are omitted from the corresponding fit-specific mean. Counts of observed wins, draws and losses and the fit-level values are in the path diagnostics file.

| Game (originals) | Measure | Outcome or association | Mean ± spread [95%], adjusted p for differences and slopes | Contributing fits |
|---|---|---|---|---|
| tictactoe (24) | chooser_moved_last | draw | +0.9938 ± 0.0198 [0.9854, 1.0021], — | 96 |
| tictactoe (24) | chooser_moved_last | loss | +0.0000 ± 0.0000 [0.0000, 0.0000], — | 96 |
| tictactoe (24) | chooser_moved_last | per_turn | +0.0105 ± 0.0200 [0.0021, 0.0190], 0.102 | 96 |
| tictactoe (24) | chooser_moved_last | win | +1.0000 ± 0.0000 [1.0000, 1.0000], — | 96 |
| tictactoe (24) | chooser_moved_last | win_minus_loss | +1.0000 ± 0.0000 [1.0000, 1.0000], — | 96 |
| tictactoe (24) | signed_90 | draw | -0.0464 ± 0.2143 [-0.1369, 0.0441], — | 96 |
| tictactoe (24) | signed_90 | loss | -0.0756 ± 0.1204 [-0.1265, -0.0248], — | 96 |
| tictactoe (24) | signed_90 | per_turn | -0.0848 ± 0.0576 [-0.1091, -0.0605], 4.36e-06 | 96 |
| tictactoe (24) | signed_90 | win | +0.2490 ± 0.1132 [0.2012, 0.2968], — | 96 |
| tictactoe (24) | signed_90 | win_minus_loss | +0.3247 ± 0.1505 [0.2611, 0.3882], 6.65e-09 | 96 |
| tictactoe (24) | signed_last | draw | +0.5375 ± 0.1454 [0.4761, 0.5989], — | 96 |
| tictactoe (24) | signed_last | loss | -0.2948 ± 0.0569 [-0.3188, -0.2708], — | 96 |
| tictactoe (24) | signed_last | per_turn | +0.0252 ± 0.0324 [0.0115, 0.0389], 0.00805 | 96 |
| tictactoe (24) | signed_last | win | +0.4578 ± 0.0405 [0.4407, 0.4749], — | 96 |
| tictactoe (24) | signed_last | win_minus_loss | +0.7526 ± 0.0626 [0.7262, 0.7790], 4.9e-25 | 96 |
| tictactoe (24) | sqrt_last_90 | draw | +0.3102 ± 0.0433 [0.2919, 0.3285], — | 96 |
| tictactoe (24) | sqrt_last_90 | loss | -0.2784 ± 0.0268 [-0.2897, -0.2671], — | 96 |
| tictactoe (24) | sqrt_last_90 | per_turn | -0.0250 ± 0.0129 [-0.0304, -0.0196], 4.38e-08 | 96 |
| tictactoe (24) | sqrt_last_90 | win | +0.4001 ± 0.0203 [0.3916, 0.4087], — | 96 |
| tictactoe (24) | sqrt_last_90 | win_minus_loss | +0.6786 ± 0.0313 [0.6653, 0.6918], 7e-31 | 96 |
| tictactoe (24) | turns | draw | +9.0000 ± 0.0000 [9.0000, 9.0000], — | 96 |
| tictactoe (24) | turns | loss | +6.8562 ± 0.2273 [6.7603, 6.9522], — | 96 |
| tictactoe (24) | turns | win | +5.8838 ± 0.1286 [5.8295, 5.9381], — | 96 |
| tictactoe (24) | turns | win_minus_loss | -0.9724 ± 0.2709 [-1.0868, -0.8580], 2.43e-13 | 96 |
| tictactoe (24) | unsigned_90 | draw | +1.3397 ± 0.1513 [1.2758, 1.4036], — | 96 |
| tictactoe (24) | unsigned_90 | loss | +0.8517 ± 0.1552 [0.7861, 0.9172], — | 96 |
| tictactoe (24) | unsigned_90 | per_turn | +0.1905 ± 0.0322 [0.1770, 0.2041], 4.41e-18 | 96 |
| tictactoe (24) | unsigned_90 | win | +0.7467 ± 0.1248 [0.6940, 0.7994], — | 96 |
| tictactoe (24) | unsigned_90 | win_minus_loss | -0.1050 ± 0.1309 [-0.1603, -0.0497], 0.00672 | 96 |
| tictactoe (24) | unsigned_last | draw | +0.5396 ± 0.1440 [0.4788, 0.6004], — | 96 |
| tictactoe (24) | unsigned_last | loss | +0.2948 ± 0.0569 [0.2708, 0.3188], — | 96 |
| tictactoe (24) | unsigned_last | per_turn | +0.0263 ± 0.0299 [0.0136, 0.0389], 0.00323 | 96 |
| tictactoe (24) | unsigned_last | win | +0.4578 ± 0.0405 [0.4407, 0.4749], — | 96 |
| tictactoe (24) | unsigned_last | win_minus_loss | +0.1631 ± 0.0764 [0.1308, 0.1953], 7.76e-09 | 96 |
| connect4 (6) | chooser_moved_last | draw | +0.7500 ± 0.4183 [0.3110, 1.1890], — | 7 |
| connect4 (16) | chooser_moved_last | loss | +0.0000 ± 0.0000 [0.0000, 0.0000], — | 63 |
| connect4 (16) | chooser_moved_last | per_turn | -0.0125 ± 0.0104 [-0.0181, -0.0070], 0.00299 | 64 |
| connect4 (16) | chooser_moved_last | win | +1.0000 ± 0.0000 [1.0000, 1.0000], — | 64 |
| connect4 (16) | chooser_moved_last | win_minus_loss | +1.0000 ± 0.0000 [1.0000, 1.0000], — | 63 |
| connect4 (6) | signed_90 | draw | -0.4024 ± 1.1320 [-1.5904, 0.7856], — | 7 |
| connect4 (16) | signed_90 | loss | -0.1651 ± 1.0475 [-0.7233, 0.3931], — | 63 |
| connect4 (16) | signed_90 | per_turn | -0.0508 ± 0.0397 [-0.0720, -0.0297], 0.00174 | 64 |
| connect4 (16) | signed_90 | win | +0.2430 ± 0.4724 [-0.0087, 0.4947], — | 64 |
| connect4 (16) | signed_90 | win_minus_loss | +0.4106 ± 0.6698 [0.0537, 0.7675], 0.135 | 63 |
| connect4 (6) | signed_last | draw | +0.3564 ± 0.4948 [-0.1629, 0.8756], — | 7 |
| connect4 (16) | signed_last | loss | -0.0902 ± 0.0858 [-0.1359, -0.0445], — | 63 |
| connect4 (16) | signed_last | per_turn | -0.0203 ± 0.0061 [-0.0236, -0.0170], 2.62e-08 | 64 |
| connect4 (16) | signed_last | win | +0.5816 ± 0.0695 [0.5446, 0.6186], — | 64 |
| connect4 (16) | signed_last | win_minus_loss | +0.6712 ± 0.0873 [0.6247, 0.7177], 1.84e-13 | 63 |
| connect4 (6) | sqrt_last_90 | draw | +0.0083 ± 0.0140 [-0.0065, 0.0230], — | 7 |
| connect4 (16) | sqrt_last_90 | loss | -0.0359 ± 0.0302 [-0.0520, -0.0198], — | 63 |
| connect4 (16) | sqrt_last_90 | per_turn | -0.0200 ± 0.0023 [-0.0213, -0.0188], 2.71e-14 | 64 |
| connect4 (16) | sqrt_last_90 | win | +0.3162 ± 0.0342 [0.2980, 0.3345], — | 64 |
| connect4 (16) | sqrt_last_90 | win_minus_loss | +0.3525 ± 0.0436 [0.3292, 0.3757], 9.21e-14 | 63 |
| connect4 (6) | turns | draw | +38.1333 ± 2.3381 [35.6797, 40.5870], — | 7 |
| connect4 (16) | turns | loss | +16.3985 ± 4.9744 [13.7479, 19.0492], — | 63 |
| connect4 (16) | turns | win | +10.3050 ± 0.8005 [9.8784, 10.7315], — | 64 |
| connect4 (16) | turns | win_minus_loss | -6.1228 ± 4.5959 [-8.5718, -3.6738], 0.00126 | 63 |
| connect4 (6) | unsigned_90 | draw | +2.5939 ± 1.2136 [1.3203, 3.8675], — | 7 |
| connect4 (16) | unsigned_90 | loss | +2.3683 ± 1.3591 [1.6440, 3.0925], — | 63 |
| connect4 (16) | unsigned_90 | per_turn | +0.0924 ± 0.0477 [0.0670, 0.1178], 2.03e-05 | 64 |
| connect4 (16) | unsigned_90 | win | +1.8424 ± 0.7158 [1.4609, 2.2238], — | 64 |
| connect4 (16) | unsigned_90 | win_minus_loss | -0.5308 ± 0.7261 [-0.9177, -0.1438], 0.082 | 63 |
| connect4 (6) | unsigned_last | draw | +0.5024 ± 0.3117 [0.1753, 0.8295], — | 7 |
| connect4 (16) | unsigned_last | loss | +0.0902 ± 0.0858 [0.0445, 0.1359], — | 63 |
| connect4 (16) | unsigned_last | per_turn | -0.0161 ± 0.0061 [-0.0194, -0.0129], 5.03e-07 | 64 |
| connect4 (16) | unsigned_last | win | +0.5816 ± 0.0695 [0.5446, 0.6186], — | 64 |
| connect4 (16) | unsigned_last | win_minus_loss | +0.4908 ± 0.1295 [0.4218, 0.5598], 4.33e-09 | 63 |
| reversi4 (16) | chooser_moved_last | draw | +0.9406 ± 0.0738 [0.9013, 0.9799], — | 63 |
| reversi4 (16) | chooser_moved_last | loss | +0.4846 ± 0.0989 [0.4319, 0.5373], — | 64 |
| reversi4 (16) | chooser_moved_last | per_turn | -0.0023 ± 0.0107 [-0.0079, 0.0034], 1 | 64 |
| reversi4 (16) | chooser_moved_last | win | +0.9900 ± 0.0087 [0.9854, 0.9946], — | 64 |
| reversi4 (16) | chooser_moved_last | win_minus_loss | +0.5054 ± 0.0978 [0.4533, 0.5575], 5.7e-11 | 64 |
| reversi4 (16) | signed_90 | draw | -0.0091 ± 0.1770 [-0.1034, 0.0853], — | 63 |
| reversi4 (16) | signed_90 | loss | -0.1648 ± 0.1012 [-0.2187, -0.1109], — | 64 |
| reversi4 (16) | signed_90 | per_turn | -0.0489 ± 0.0220 [-0.0606, -0.0372], 4.21e-06 | 64 |
| reversi4 (16) | signed_90 | win | -0.1312 ± 0.0993 [-0.1841, -0.0782], — | 64 |
| reversi4 (16) | signed_90 | win_minus_loss | +0.0337 ± 0.1792 [-0.0619, 0.1292], 1 | 64 |
| reversi4 (16) | signed_last | draw | +0.7486 ± 0.1208 [0.6842, 0.8130], — | 63 |
| reversi4 (16) | signed_last | loss | +0.0156 ± 0.1545 [-0.0667, 0.0979], — | 64 |
| reversi4 (16) | signed_last | per_turn | +0.0008 ± 0.0176 [-0.0086, 0.0102], 1 | 64 |
| reversi4 (16) | signed_last | win | +0.7580 ± 0.0407 [0.7363, 0.7797], — | 64 |
| reversi4 (16) | signed_last | win_minus_loss | +0.7424 ± 0.1572 [0.6586, 0.8262], 2.01e-10 | 64 |
| reversi4 (16) | sqrt_last_90 | draw | +0.3216 ± 0.0516 [0.2941, 0.3491], — | 63 |
| reversi4 (16) | sqrt_last_90 | loss | -0.0001 ± 0.0544 [-0.0291, 0.0289], — | 64 |
| reversi4 (16) | sqrt_last_90 | per_turn | -0.0195 ± 0.0062 [-0.0228, -0.0162], 5.3e-08 | 64 |
| reversi4 (16) | sqrt_last_90 | win | +0.3262 ± 0.0133 [0.3192, 0.3333], — | 64 |
| reversi4 (16) | sqrt_last_90 | win_minus_loss | +0.3263 ± 0.0498 [0.2998, 0.3529], 1.85e-12 | 64 |
| reversi4 (16) | turns | draw | +9.8342 ± 0.6561 [9.4846, 10.1838], — | 63 |
| reversi4 (16) | turns | loss | +10.4974 ± 0.2970 [10.3392, 10.6557], — | 64 |
| reversi4 (16) | turns | win | +10.3378 ± 0.2443 [10.2076, 10.4679], — | 64 |
| reversi4 (16) | turns | win_minus_loss | -0.1597 ± 0.4610 [-0.4053, 0.0860], 0.745 | 64 |
| reversi4 (16) | unsigned_90 | draw | +3.0446 ± 0.2562 [2.9081, 3.1812], — | 63 |
| reversi4 (16) | unsigned_90 | loss | +3.4333 ± 0.2278 [3.3119, 3.5547], — | 64 |
| reversi4 (16) | unsigned_90 | per_turn | +0.1650 ± 0.0354 [0.1461, 0.1838], 2.32e-10 | 64 |
| reversi4 (16) | unsigned_90 | win | +3.2318 ± 0.1656 [3.1435, 3.3200], — | 64 |
| reversi4 (16) | unsigned_90 | win_minus_loss | -0.2015 ± 0.2747 [-0.3479, -0.0551], 0.082 | 64 |
| reversi4 (16) | unsigned_last | draw | +0.8459 ± 0.0559 [0.8162, 0.8757], — | 63 |
| reversi4 (16) | unsigned_last | loss | +0.7393 ± 0.0358 [0.7202, 0.7583], — | 64 |
| reversi4 (16) | unsigned_last | per_turn | +0.0117 ± 0.0057 [0.0086, 0.0147], 1.15e-05 | 64 |
| reversi4 (16) | unsigned_last | win | +0.7706 ± 0.0384 [0.7502, 0.7911], — | 64 |
| reversi4 (16) | unsigned_last | win_minus_loss | +0.0314 ± 0.0281 [0.0164, 0.0464], 0.00494 | 64 |

## Network fitting

| Game (originals) | Network | Training mean squared frequency error | Fitting boards | Fitting legal candidates | Distinct observed decision boards |
|---|---|---|---|---|---|
| tictactoe (24) | V | 0.1676 ± 0.0048 | 2287.6 ± 113.9 | 4681.6 ± 318.5 | 4520.0 ± 0.0 |
| tictactoe (24) | Q | 0.1607 ± 0.0055 | 2942.7 ± 62.1 | 10543.4 ± 217.6 | 4520.0 ± 0.0 |
| connect4 (16) | V | 0.1155 ± 0.0051 | 2720.0 ± 523.6 | 14919.2 ± 2649.9 | 4187.5 ± 797.7 |
| connect4 (16) | Q | 0.0569 ± 0.0133 | 2720.0 ± 523.6 | 15263.3 ± 2640.3 | 4187.5 ± 797.7 |
| reversi4 (16) | V | 0.1279 ± 0.0090 | 727.7 ± 31.6 | 1370.0 ± 73.4 | 1188.4 ± 24.5 |
| reversi4 (16) | Q | 0.0605 ± 0.0137 | 776.0 ± 31.3 | 1473.8 ± 71.1 | 1188.4 ± 24.5 |
