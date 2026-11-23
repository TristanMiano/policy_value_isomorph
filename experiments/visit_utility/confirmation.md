# Confirmation results for visit-trained game scores

The frozen original takes its highest-scoring legal move. A network scoring the resulting board (V) or the board and move (Q) estimates the fraction of observed visits taking a move. Direct selection takes the largest immediate fit. Von Neumann–Morgenstern (VNM) selection maximizes expected complete-game score, replanning on each actual turn. Ties prefer the lowest move index. 'copies' projects highest fitted choices for both seats. The rule 'uniform_reply' projects greedy own moves against uniform legal replies. The rule 'mixed_reply' mixes those two opponents equally once per path. The rule 'uniform_both' projects uniform legal moves for both seats. Each random component gets the listed number of full paths, with exact expectations in tic-tac-toe. The assisted comparison includes an opponent considering tactics and one using positional priorities. The supplied-outcome reference adds actual game results and projects guarded own play against four opponent types.

The utility name specifies the score of a completed game from the current choosing seat. 'signed' counts own moves positively and opposing moves negatively, 'unsigned' counts both positively, and 'own' counts only own moves. 'last' counts the final fitted frequency. The rule 'sum' adds the whole path. The suffix 50, 75, 90 or 97 gives the discount per turn in percent, starting at weight one. 'sqrt_last' and 'square_last' apply the square root or square to the final frequency, signed by mover. 'last_plus_sum' adds the signed last frequency and 0.1 times the signed 0.9-discounted sum. A constant score uses 0.5 in the identical formula. Constant continuations also break all greedy choices by move index. Both constants induce the identical player when all continuing moves are uniformly random.

Win, draw and loss percentages are seat-balanced against four equally weighted external opponent groups. Agreement is the weighted percentage of withheld boards on which the reconstruction chose the original's favorite. Exact-copy self-play uses two identical deployed players and counts both seats. Values give means ± sample standard deviations across originals after averaging their four fits. The equal-game spread describes the average of one independent original per named game. The selected rule is VNM V, signed_last_90, mixed_reply, 8 paths.

Independent original counts for tic-tac-toe, Connect Four and 4×4 Reversi are 24, 16, 16.
For paired changes, brackets give individual 95% confidence intervals, followed by the two-sided p-value for a zero mean difference, Holm-adjusted across the table's win and agreement comparisons in all four populations. A dash means the across-original variance is exactly zero. Discovery contrasts are exploratory because the leader was chosen on those data. The compressed estimates and contrasts alongside this table contain all unrounded measurements, raw p-values, original-level differences, and intervals resampling the independent originals.

## External win rate (%)

| Rule | Tic-tac-toe | Connect Four | 4×4 Reversi | Equal games |
|---|---|---|---|---|
| VNM V, supplied outcomes and frequencies, guarded mixture | 52.03 ± 4.59 | 93.38 ± 2.65 | 75.93 ± 3.86 | 73.78 ± 2.18 |
| VNM V, signed_last_90, mixed_reply, 8 paths, constant scores | 47.64 ± 1.32 | 80.54 ± 7.34 | 48.93 ± 4.82 | 59.04 ± 2.96 |
| VNM V, signed final frequency, assisted opponent mixture | 49.57 ± 3.62 | 79.10 ± 13.63 | 47.24 ± 6.56 | 58.64 ± 5.18 |
| VNM V, signed_last_90, mixed_reply, 8 paths, constant scores and continuations | 44.79 ± 2.79 | 78.61 ± 3.69 | 51.46 ± 4.12 | 58.29 ± 2.06 |
| Frozen original | 45.71 ± 4.97 | 58.45 ± 12.27 | 69.24 ± 2.04 | 57.80 ± 4.47 |
| VNM V, signed_last_90, mixed_reply, 8 paths | 50.56 ± 3.62 | 72.19 ± 14.65 | 44.04 ± 5.69 | 55.60 ± 5.37 |
| VNM Q, signed_last_90, mixed_reply, 8 paths | 47.39 ± 2.24 | 74.58 ± 13.69 | 43.12 ± 3.80 | 55.03 ± 4.79 |
| VNM Q, signed_last_90, uniform_both, 8 paths | 46.76 ± 1.47 | 70.19 ± 9.69 | 46.07 ± 5.37 | 54.34 ± 3.73 |
| VNM V, signed_last_90, uniform_both, 8 paths | 46.46 ± 1.67 | 70.29 ± 9.11 | 44.80 ± 4.91 | 53.85 ± 3.50 |
| VNM V, signed_last, mixed_reply, 8 paths | 49.89 ± 3.82 | 66.50 ± 16.86 | 44.24 ± 5.23 | 53.55 ± 6.02 |
| VNM V, last_plus_sum, mixed_reply, 8 paths | 49.83 ± 4.03 | 66.82 ± 18.77 | 42.75 ± 4.97 | 53.13 ± 6.61 |
| VNM Q, signed_90, mixed_reply, 8 paths | 46.69 ± 3.56 | 63.94 ± 16.19 | 43.87 ± 6.84 | 51.50 ± 5.98 |
| Direct Q | 42.80 ± 2.86 | 57.47 ± 10.57 | 51.54 ± 2.89 | 50.60 ± 3.78 |
| VNM V, signed_90, mixed_reply, 8 paths | 47.20 ± 2.63 | 51.46 ± 32.04 | 37.01 ± 4.91 | 45.23 ± 10.84 |
| Direct V | 39.80 ± 3.41 | 44.12 ± 20.25 | 47.39 ± 3.49 | 43.77 ± 6.94 |
| VNM Q, own_90, mixed_reply, 8 paths | 40.12 ± 2.57 | 33.72 ± 22.59 | 52.17 ± 3.39 | 42.00 ± 7.66 |
| VNM V, own_90, mixed_reply, 8 paths | 37.42 ± 3.84 | 38.87 ± 24.94 | 45.12 ± 6.64 | 40.47 ± 8.70 |
| VNM Q, unsigned_90, uniform_both, 8 paths | 23.85 ± 2.60 | 18.75 ± 8.01 | 47.61 ± 3.80 | 30.07 ± 3.08 |
| VNM V, unsigned_90, mixed_reply, 8 paths | 19.21 ± 3.59 | 18.02 ± 7.43 | 47.41 ± 5.32 | 28.21 ± 3.27 |

## Tic-tac-toe

| Rule | External wins / draws / losses (%) | Move agreement (%) | Exact-copy wins / draws (%) | Selected minus row in wins, points [95%], p | Selected minus row in agreement, points [95%], p |
|---|---|---|---|---|---|
| VNM V, supplied outcomes and frequencies, guarded mixture | 52.03 ± 4.59 / 32.81 ± 7.25 / 15.17 ± 6.23 | 73.44 ± 2.13 | 19.79 ± 13.75 / 60.42 ± 27.50 | -1.47 ± 3.92 [-3.12, 0.19], 1 | -2.24 ± 1.90 [-3.05, -1.44], 0.000574 |
| VNM V, signed_last_90, mixed_reply, 8 paths, constant scores | 47.64 ± 1.32 / 32.91 ± 4.84 / 19.45 ± 4.67 | 65.38 ± 3.43 | 28.12 ± 11.80 / 43.75 ± 23.60 | +2.92 ± 3.54 [1.42, 4.41], 0.0289 | +5.81 ± 2.95 [4.57, 7.06], 1.58e-07 |
| VNM V, signed final frequency, assisted opponent mixture | 49.57 ± 3.62 / 25.42 ± 4.39 / 25.01 ± 2.78 | 67.77 ± 2.26 | 31.77 ± 11.05 / 36.46 ± 22.09 | +0.99 ± 2.94 [-0.25, 2.23], 1 | +3.43 ± 1.14 [2.95, 3.91], 3.84e-11 |
| VNM V, signed_last_90, mixed_reply, 8 paths, constant scores and continuations | 44.79 ± 2.79 / 39.27 ± 1.16 / 15.95 ± 2.44 | 60.66 ± 3.27 | 50.00 ± 0.00 / 0.00 ± 0.00 | +5.77 ± 5.25 [3.55, 7.99], 0.0014 | +10.54 ± 3.07 [9.24, 11.83], 2.58e-12 |
| Frozen original | 45.71 ± 4.97 / 54.29 ± 4.97 / 0.00 ± 0.00 | 100.00 ± 0.00 | 0.00 ± 0.00 / 100.00 ± 0.00 | +4.85 ± 7.04 [1.88, 7.82], 0.118 | -28.80 ± 2.47 [-29.85, -27.76], 4.02e-24 |
| VNM V, signed_last_90, mixed_reply, 8 paths | 50.56 ± 3.62 / 30.99 ± 5.85 / 18.45 ± 5.49 | 71.20 ± 2.47 | 24.48 ± 13.02 / 51.04 ± 26.04 | reference | reference |
| VNM Q, signed_last_90, mixed_reply, 8 paths | 47.39 ± 2.24 / 30.78 ± 7.35 / 21.83 ± 6.72 | 70.66 ± 2.50 | 36.98 ± 13.53 / 26.04 ± 27.07 | +3.17 ± 3.71 [1.61, 4.74], 0.0206 | +0.54 ± 2.01 [-0.31, 1.39], 1 |
| VNM Q, signed_last_90, uniform_both, 8 paths | 46.76 ± 1.47 / 31.05 ± 4.70 / 22.20 ± 4.25 | 66.56 ± 2.19 | 40.62 ± 11.80 / 18.75 ± 23.60 | +3.80 ± 4.11 [2.07, 5.53], 0.00953 | +4.64 ± 1.94 [3.82, 5.46], 4.23e-09 |
| VNM V, signed_last_90, uniform_both, 8 paths | 46.46 ± 1.67 / 31.13 ± 3.03 / 22.41 ± 3.37 | 66.90 ± 2.52 | 46.35 ± 7.80 / 7.29 ± 15.60 | +4.10 ± 4.21 [2.32, 5.88], 0.0057 | +4.29 ± 1.34 [3.73, 4.86], 1.07e-11 |
| VNM V, signed_last, mixed_reply, 8 paths | 49.89 ± 3.82 / 26.86 ± 4.90 / 23.25 ± 3.95 | 67.89 ± 2.38 | 26.56 ± 13.95 / 46.88 ± 27.89 | +0.66 ± 2.57 [-0.42, 1.75], 1 | +3.31 ± 0.94 [2.91, 3.70], 1.42e-12 |
| VNM V, last_plus_sum, mixed_reply, 8 paths | 49.83 ± 4.03 / 27.87 ± 6.20 / 22.30 ± 4.71 | 69.43 ± 2.31 | 22.92 ± 14.59 / 54.17 ± 29.18 | +0.73 ± 1.74 [-0.01, 1.46], 1 | +1.77 ± 0.82 [1.42, 2.12], 2.7e-08 |
| VNM Q, signed_90, mixed_reply, 8 paths | 46.69 ± 3.56 / 35.75 ± 5.97 / 17.56 ± 5.93 | 80.93 ± 1.65 | 31.25 ± 12.23 / 37.50 ± 24.45 | +3.87 ± 5.25 [1.65, 6.09], 0.072 | -9.73 ± 1.88 [-10.53, -8.94], 3.69e-16 |
| Direct Q | 42.80 ± 2.86 / 34.45 ± 8.53 / 22.76 ± 8.64 | 78.56 ± 1.78 | 28.65 ± 14.03 / 42.71 ± 28.05 | +7.76 ± 4.85 [5.71, 9.81], 5.83e-06 | -7.36 ± 2.23 [-8.31, -6.42], 6.04e-12 |
| VNM V, signed_90, mixed_reply, 8 paths | 47.20 ± 2.63 / 41.88 ± 4.71 / 10.91 ± 4.61 | 76.38 ± 1.93 | 16.15 ± 13.02 / 67.71 ± 26.04 | +3.36 ± 3.87 [1.72, 4.99], 0.0181 | -5.19 ± 1.28 [-5.73, -4.65], 7.84e-14 |
| Direct V | 39.80 ± 3.41 / 35.65 ± 6.08 / 24.55 ± 7.77 | 73.88 ± 2.09 | 16.15 ± 13.53 / 67.71 ± 27.07 | +10.76 ± 4.74 [8.76, 12.76], 1.12e-08 | -2.68 ± 1.88 [-3.48, -1.89], 3.56e-05 |
| VNM Q, own_90, mixed_reply, 8 paths | 40.12 ± 2.57 / 45.63 ± 5.31 / 14.25 ± 4.68 | 70.30 ± 1.96 | 18.23 ± 13.28 / 63.54 ± 26.56 | +10.44 ± 4.70 [8.46, 12.42], 1.67e-08 | +0.89 ± 2.52 [-0.17, 1.96], 1 |
| VNM V, own_90, mixed_reply, 8 paths | 37.42 ± 3.84 / 52.10 ± 4.94 / 10.48 ± 3.66 | 68.13 ± 1.93 | 4.17 ± 6.02 / 91.67 ± 12.04 | +13.14 ± 6.31 [10.48, 15.81], 5.55e-08 | +3.07 ± 2.66 [1.95, 4.19], 0.000759 |
| VNM Q, unsigned_90, uniform_both, 8 paths | 23.85 ± 2.60 / 50.49 ± 6.70 / 25.65 ± 6.46 | 49.59 ± 2.41 | 17.19 ± 14.19 / 65.62 ± 28.37 | +26.70 ± 4.81 [24.67, 28.73], 7.34e-17 | +21.61 ± 2.90 [20.38, 22.83], 9.86e-20 |
| VNM V, unsigned_90, mixed_reply, 8 paths | 19.21 ± 3.59 / 56.99 ± 6.74 / 23.80 ± 5.56 | 40.69 ± 1.50 | 10.94 ± 9.97 / 78.12 ± 19.94 | +31.35 ± 5.76 [28.92, 33.78], 1.15e-16 | +30.50 ± 2.69 [29.37, 31.64], 7.51e-24 |

## Connect Four

| Rule | External wins / draws / losses (%) | Move agreement (%) | Exact-copy wins / draws (%) | Selected minus row in wins, points [95%], p | Selected minus row in agreement, points [95%], p |
|---|---|---|---|---|---|
| VNM V, supplied outcomes and frequencies, guarded mixture | 93.38 ± 2.65 / 0.49 ± 0.60 / 6.13 ± 2.57 | 31.45 ± 7.35 | 49.10 ± 1.50 / 1.81 ± 3.00 | -21.19 ± 14.26 [-28.79, -13.59], 0.00199 | -1.69 ± 3.62 [-3.62, 0.24], 1 |
| VNM V, signed_last_90, mixed_reply, 8 paths, constant scores | 80.54 ± 7.34 / 0.49 ± 0.77 / 18.97 ± 7.40 | 26.34 ± 10.19 | 47.71 ± 2.87 / 4.59 ± 5.73 | -8.35 ± 9.05 [-13.17, -3.53], 0.105 | +3.42 ± 4.60 [0.97, 5.87], 0.369 |
| VNM V, signed final frequency, assisted opponent mixture | 79.10 ± 13.63 / 0.22 ± 0.43 / 20.68 ± 13.64 | 26.19 ± 6.07 | 49.24 ± 1.60 / 1.51 ± 3.20 | -6.91 ± 5.06 [-9.60, -4.21], 0.00456 | +3.56 ± 3.30 [1.81, 5.32], 0.0332 |
| VNM V, signed_last_90, mixed_reply, 8 paths, constant scores and continuations | 78.61 ± 3.69 / 0.17 ± 0.32 / 21.22 ± 3.78 | 25.25 ± 12.00 | 49.02 ± 1.70 / 1.95 ± 3.40 | -6.42 ± 12.90 [-13.30, 0.46], 1 | +4.50 ± 8.52 [-0.04, 9.04], 1 |
| Frozen original | 58.45 ± 12.27 / 0.68 ± 1.94 / 40.87 ± 13.28 | 100.00 ± 0.00 | 49.49 ± 0.86 / 1.03 ± 1.73 | +13.75 ± 10.35 [8.23, 19.26], 0.00581 | -70.24 ± 7.32 [-74.15, -66.34], 2.88e-14 |
| VNM V, signed_last_90, mixed_reply, 8 paths | 72.19 ± 14.65 / 0.27 ± 0.49 / 27.54 ± 14.63 | 29.76 ± 7.32 | 47.63 ± 2.84 / 4.74 ± 5.69 | reference | reference |
| VNM Q, signed_last_90, mixed_reply, 8 paths | 74.58 ± 13.69 / 0.34 ± 0.47 / 25.07 ± 13.91 | 38.33 ± 3.96 | 48.80 ± 2.54 / 2.39 ± 5.07 | -2.39 ± 6.22 [-5.70, 0.92], 1 | -8.57 ± 5.77 [-11.65, -5.50], 0.00199 |
| VNM Q, signed_last_90, uniform_both, 8 paths | 70.19 ± 9.69 / 0.05 ± 0.20 / 29.76 ± 9.70 | 32.52 ± 5.13 | 49.51 ± 1.55 / 0.98 ± 3.09 | +2.00 ± 8.25 [-2.40, 6.40], 1 | -2.76 ± 4.02 [-4.90, -0.62], 0.535 |
| VNM V, signed_last_90, uniform_both, 8 paths | 70.29 ± 9.11 / 0.12 ± 0.40 / 29.59 ± 9.22 | 34.56 ± 4.34 | 49.56 ± 1.56 / 0.88 ± 3.11 | +1.90 ± 8.23 [-2.48, 6.29], 1 | -4.80 ± 4.45 [-7.18, -2.43], 0.0332 |
| VNM V, signed_last, mixed_reply, 8 paths | 66.50 ± 16.86 / 0.34 ± 0.60 / 33.15 ± 16.77 | 24.01 ± 5.23 | 49.22 ± 0.55 / 1.56 ± 1.10 | +5.69 ± 4.50 [3.29, 8.08], 0.00917 | +5.74 ± 3.36 [3.95, 7.53], 0.000479 |
| VNM V, last_plus_sum, mixed_reply, 8 paths | 66.82 ± 18.77 / 0.46 ± 0.73 / 32.71 ± 18.66 | 28.71 ± 4.10 | 47.22 ± 4.12 / 5.57 ± 8.24 | +5.37 ± 6.14 [2.10, 8.64], 0.142 | +1.04 ± 4.72 [-1.47, 3.56], 1 |
| VNM Q, signed_90, mixed_reply, 8 paths | 63.94 ± 16.19 / 2.05 ± 3.79 / 34.01 ± 18.47 | 84.23 ± 11.82 | 46.75 ± 3.43 / 6.49 ± 6.85 | +8.25 ± 7.84 [4.08, 12.43], 0.0393 | -54.47 ± 17.98 [-64.05, -44.89], 3.84e-07 |
| Direct Q | 57.47 ± 10.57 / 1.20 ± 2.60 / 41.33 ± 12.09 | 92.97 ± 6.31 | 48.51 ± 4.08 / 2.98 ± 8.16 | +14.72 ± 9.41 [9.71, 19.74], 0.00122 | -63.21 ± 12.69 [-69.98, -56.45], 3.91e-10 |
| VNM V, signed_90, mixed_reply, 8 paths | 51.46 ± 32.04 / 1.59 ± 1.70 / 46.95 ± 32.96 | 57.85 ± 5.28 | 49.37 ± 0.71 / 1.27 ± 1.42 | +20.73 ± 19.31 [10.44, 31.02], 0.0339 | -28.10 ± 6.97 [-31.81, -24.38], 7.95e-09 |
| Direct V | 44.12 ± 20.25 / 0.24 ± 0.55 / 55.64 ± 20.50 | 76.63 ± 4.29 | 49.98 ± 0.10 / 0.05 ± 0.20 | +28.08 ± 9.70 [22.91, 33.25], 7.06e-07 | -46.88 ± 6.57 [-50.38, -43.37], 2.27e-12 |
| VNM Q, own_90, mixed_reply, 8 paths | 33.72 ± 22.59 / 7.13 ± 7.07 / 59.16 ± 28.84 | 71.32 ± 14.38 | 29.64 ± 13.99 / 40.72 ± 27.98 | +38.48 ± 10.81 [32.71, 44.24], 4.35e-08 | -41.56 ± 20.53 [-52.50, -30.62], 6.53e-05 |
| VNM V, own_90, mixed_reply, 8 paths | 38.87 ± 24.94 / 7.45 ± 5.70 / 53.69 ± 29.26 | 53.95 ± 4.04 | 38.06 ± 5.37 / 23.88 ± 10.74 | +33.33 ± 12.72 [26.55, 40.10], 2.62e-06 | -24.19 ± 9.16 [-29.07, -19.31], 2.36e-06 |
| VNM Q, unsigned_90, uniform_both, 8 paths | 18.75 ± 8.01 / 6.03 ± 5.56 / 75.22 ± 12.36 | 74.54 ± 9.07 | 28.10 ± 12.73 / 43.80 ± 25.47 | +53.44 ± 10.66 [47.76, 59.12], 3.58e-10 | -44.79 ± 15.28 [-52.93, -36.64], 6.02e-07 |
| VNM V, unsigned_90, mixed_reply, 8 paths | 18.02 ± 7.43 / 7.47 ± 4.51 / 74.51 ± 11.22 | 38.86 ± 4.72 | 32.18 ± 7.93 / 35.64 ± 15.86 | +54.17 ± 9.14 [49.31, 59.04], 3.28e-11 | -9.10 ± 11.10 [-15.01, -3.19], 0.207 |

## 4×4 Reversi

| Rule | External wins / draws / losses (%) | Move agreement (%) | Exact-copy wins / draws (%) | Selected minus row in wins, points [95%], p | Selected minus row in agreement, points [95%], p |
|---|---|---|---|---|---|
| VNM V, supplied outcomes and frequencies, guarded mixture | 75.93 ± 3.86 / 1.78 ± 0.84 / 22.29 ± 3.63 | 86.13 ± 1.09 | 49.73 ± 0.31 / 0.54 ± 0.62 | -31.88 ± 5.23 [-34.67, -29.10], 2.18e-11 | -3.43 ± 1.13 [-4.03, -2.83], 3.78e-07 |
| VNM V, signed_last_90, mixed_reply, 8 paths, constant scores | 48.93 ± 4.82 / 5.64 ± 1.96 / 45.43 ± 4.59 | 87.15 ± 0.91 | 46.17 ± 2.87 / 7.67 ± 5.74 | -4.88 ± 3.50 [-6.75, -3.02], 0.00375 | -4.45 ± 1.19 [-5.08, -3.82], 2.11e-08 |
| VNM V, signed final frequency, assisted opponent mixture | 47.24 ± 6.56 / 7.57 ± 2.63 / 45.19 ± 4.84 | 83.08 ± 1.32 | 46.56 ± 3.18 / 6.88 ± 6.37 | -3.20 ± 4.52 [-5.61, -0.79], 0.484 | -0.38 ± 0.47 [-0.63, -0.13], 0.215 |
| VNM V, signed_last_90, mixed_reply, 8 paths, constant scores and continuations | 51.46 ± 4.12 / 4.88 ± 1.71 / 43.65 ± 4.35 | 86.87 ± 0.92 | 47.73 ± 0.81 / 4.54 ± 1.62 | -7.42 ± 4.83 [-10.00, -4.85], 0.00143 | -4.17 ± 1.24 [-4.83, -3.51], 9.77e-08 |
| Frozen original | 69.24 ± 2.04 / 1.42 ± 0.62 / 29.35 ± 1.78 | 100.00 ± 0.00 | 49.78 ± 0.32 / 0.44 ± 0.64 | -25.20 ± 7.01 [-28.93, -21.46], 3.83e-08 | -17.30 ± 1.23 [-17.96, -16.64], 1.05e-16 |
| VNM V, signed_last_90, mixed_reply, 8 paths | 44.04 ± 5.69 / 7.67 ± 1.92 / 48.29 ± 4.93 | 82.70 ± 1.23 | 45.70 ± 4.37 / 8.59 ± 8.74 | reference | reference |
| VNM Q, signed_last_90, mixed_reply, 8 paths | 43.12 ± 3.80 / 8.42 ± 2.48 / 48.46 ± 3.46 | 80.09 ± 1.40 | 47.41 ± 3.07 / 5.18 ± 6.13 | +0.93 ± 5.13 [-1.80, 3.66], 1 | +2.62 ± 1.13 [2.01, 3.22], 1.33e-05 |
| VNM Q, signed_last_90, uniform_both, 8 paths | 46.07 ± 5.37 / 7.64 ± 2.33 / 46.29 ± 4.73 | 80.49 ± 1.51 | 46.51 ± 3.89 / 6.98 ± 7.78 | -2.03 ± 8.90 [-6.77, 2.72], 1 | +2.21 ± 1.35 [1.49, 2.93], 0.000759 |
| VNM V, signed_last_90, uniform_both, 8 paths | 44.80 ± 4.91 / 7.71 ± 1.92 / 47.49 ± 4.82 | 82.78 ± 1.37 | 45.29 ± 5.18 / 9.42 ± 10.36 | -0.76 ± 8.02 [-5.03, 3.51], 1 | -0.08 ± 0.78 [-0.49, 0.34], 1 |
| VNM V, signed_last, mixed_reply, 8 paths | 44.24 ± 5.23 / 7.45 ± 1.61 / 48.32 ± 4.62 | 82.91 ± 1.25 | 46.22 ± 3.56 / 7.57 ± 7.12 | -0.20 ± 1.71 [-1.11, 0.72], 1 | -0.21 ± 0.37 [-0.41, -0.02], 1 |
| VNM V, last_plus_sum, mixed_reply, 8 paths | 42.75 ± 4.97 / 7.79 ± 2.11 / 49.46 ± 4.42 | 81.41 ± 1.21 | 45.51 ± 4.61 / 8.98 ± 9.21 | +1.29 ± 2.60 [-0.09, 2.68], 1 | +1.30 ± 0.67 [0.94, 1.65], 0.00011 |
| VNM Q, signed_90, mixed_reply, 8 paths | 43.87 ± 6.84 / 7.45 ± 2.50 / 48.68 ± 5.70 | 83.79 ± 1.08 | 44.63 ± 5.26 / 10.74 ± 10.52 | +0.17 ± 8.44 [-4.33, 4.67], 1 | -1.09 ± 1.54 [-1.91, -0.27], 0.484 |
| Direct Q | 51.54 ± 2.89 / 4.71 ± 1.17 / 43.75 ± 2.97 | 85.71 ± 1.27 | 48.44 ± 0.78 / 3.12 ± 1.56 | -7.50 ± 6.03 [-10.71, -4.28], 0.0105 | -3.01 ± 1.70 [-3.92, -2.11], 0.000318 |
| VNM V, signed_90, mixed_reply, 8 paths | 37.01 ± 4.91 / 7.40 ± 1.87 / 55.59 ± 4.59 | 82.77 ± 1.05 | 44.29 ± 5.99 / 11.43 ± 11.98 | +7.03 ± 6.42 [3.61, 10.45], 0.0302 | -0.07 ± 1.22 [-0.72, 0.59], 1 |
| Direct V | 47.39 ± 3.49 / 5.18 ± 1.60 / 47.44 ± 4.13 | 83.72 ± 1.61 | 48.46 ± 0.71 / 3.08 ± 1.41 | -3.34 ± 6.27 [-6.69, -0.00], 1 | -1.02 ± 2.09 [-2.13, 0.09], 1 |
| VNM Q, own_90, mixed_reply, 8 paths | 52.17 ± 3.39 / 5.66 ± 2.21 / 42.16 ± 3.91 | 84.54 ± 1.45 | 48.07 ± 2.22 / 3.86 ± 4.43 | -8.13 ± 5.83 [-11.24, -5.02], 0.00377 | -1.84 ± 2.03 [-2.92, -0.76], 0.115 |
| VNM V, own_90, mixed_reply, 8 paths | 45.12 ± 6.64 / 5.30 ± 1.76 / 49.58 ± 6.82 | 83.39 ± 1.28 | 48.73 ± 0.71 / 2.54 ± 1.41 | -1.07 ± 8.55 [-5.63, 3.48], 1 | -0.69 ± 1.84 [-1.67, 0.29], 1 |
| VNM Q, unsigned_90, uniform_both, 8 paths | 47.61 ± 3.80 / 4.93 ± 1.52 / 47.46 ± 3.48 | 82.73 ± 1.26 | 47.24 ± 4.22 / 5.52 ± 8.44 | -3.56 ± 6.32 [-6.93, -0.20], 1 | -0.02 ± 1.65 [-0.90, 0.86], 1 |
| VNM V, unsigned_90, mixed_reply, 8 paths | 47.41 ± 5.32 / 4.20 ± 1.22 / 48.39 ± 5.74 | 81.26 ± 1.35 | 44.78 ± 5.58 / 10.45 ± 11.17 | -3.37 ± 8.26 [-7.77, 1.03], 1 | +1.44 ± 1.66 [0.56, 2.33], 0.146 |

## Equal games

| Rule | External wins / draws / losses (%) | Move agreement (%) | Exact-copy wins / draws (%) | Selected minus row in wins, points [95%], p | Selected minus row in agreement, points [95%], p |
|---|---|---|---|---|---|
| VNM V, supplied outcomes and frequencies, guarded mixture | 73.78 ± 2.18 / 11.69 ± 2.44 / 14.53 ± 2.55 | 63.67 ± 2.58 | 39.54 ± 4.61 / 20.92 ± 9.22 | -18.18 ± 5.23 [-20.87, -15.49], 5.59e-10 | -2.45 ± 1.42 [-3.16, -1.75], 1.95e-05 |
| VNM V, signed_last_90, mixed_reply, 8 paths, constant scores | 59.04 ± 2.96 / 13.01 ± 1.76 / 27.95 ± 3.29 | 59.63 ± 3.60 | 40.67 ± 4.16 / 18.67 ± 8.32 | -3.44 ± 3.44 [-5.18, -1.69], 0.0275 | +1.59 ± 1.86 [0.68, 2.51], 0.0699 |
| VNM V, signed final frequency, assisted opponent mixture | 58.64 ± 5.18 / 11.07 ± 1.71 / 30.29 ± 4.91 | 59.01 ± 2.20 | 42.52 ± 3.87 / 14.95 ± 7.74 | -3.04 ± 2.47 [-4.25, -1.82], 0.000933 | +2.20 ± 1.17 [1.60, 2.81], 4.11e-05 |
| VNM V, signed_last_90, mixed_reply, 8 paths, constant scores and continuations | 58.29 ± 2.06 / 14.77 ± 0.70 / 26.94 ± 2.09 | 57.59 ± 4.16 | 48.92 ± 0.63 / 2.16 ± 1.26 | -2.69 ± 4.92 [-5.18, -0.20], 1 | +3.62 ± 3.05 [2.06, 5.19], 0.00814 |
| Frozen original | 57.80 ± 4.47 / 18.80 ± 1.79 / 23.40 ± 4.47 | 100.00 ± 0.00 | 33.09 ± 0.31 / 33.82 ± 0.61 | -2.20 ± 4.78 [-4.52, 0.12], 1 | -38.78 ± 2.61 [-40.13, -37.44], 2.66e-20 |
| VNM V, signed_last_90, mixed_reply, 8 paths | 55.60 ± 5.37 / 12.98 ± 2.06 / 31.43 ± 5.46 | 61.22 ± 2.61 | 39.27 ± 4.68 / 21.46 ± 9.35 | reference | reference |
| VNM Q, signed_last_90, mixed_reply, 8 paths | 55.03 ± 4.79 / 13.18 ± 2.59 / 31.79 ± 5.28 | 63.02 ± 1.63 | 44.40 ± 4.70 / 11.20 ± 9.40 | +0.57 ± 2.96 [-0.88, 2.02], 1 | -1.81 ± 2.07 [-2.87, -0.74], 0.105 |
| VNM Q, signed_last_90, uniform_both, 8 paths | 54.34 ± 3.73 / 12.91 ± 1.75 / 32.75 ± 3.87 | 59.86 ± 1.92 | 45.55 ± 4.17 / 8.90 ± 8.35 | +1.26 ± 4.27 [-0.87, 3.39], 1 | +1.36 ± 1.55 [0.58, 2.14], 0.072 |
| VNM V, signed_last_90, uniform_both, 8 paths | 53.85 ± 3.50 / 12.99 ± 1.20 / 33.16 ± 3.65 | 61.41 ± 1.73 | 47.07 ± 3.16 / 5.86 ± 6.33 | +1.75 ± 4.08 [-0.28, 3.78], 1 | -0.20 ± 1.57 [-1.01, 0.62], 1 |
| VNM V, signed_last, mixed_reply, 8 paths | 53.55 ± 6.02 / 11.55 ± 1.73 / 34.91 ± 5.95 | 58.27 ± 1.96 | 40.67 ± 4.80 / 18.67 ± 9.60 | +2.05 ± 1.82 [1.15, 2.95], 0.00503 | +2.95 ± 1.17 [2.34, 3.56], 1.18e-06 |
| VNM V, last_plus_sum, mixed_reply, 8 paths | 53.13 ± 6.61 / 12.04 ± 2.20 / 34.83 ± 6.58 | 59.85 ± 1.62 | 38.55 ± 5.28 / 22.91 ± 10.56 | +2.46 ± 2.30 [1.29, 3.64], 0.0163 | +1.37 ± 1.61 [0.52, 2.22], 0.146 |
| VNM Q, signed_90, mixed_reply, 8 paths | 51.50 ± 5.98 / 15.08 ± 2.50 / 33.42 ± 6.74 | 82.98 ± 3.99 | 40.88 ± 4.58 / 18.25 ± 9.16 | +4.10 ± 4.22 [2.02, 6.17], 0.0173 | -21.76 ± 6.05 [-24.97, -18.56], 2.49e-08 |
| Direct Q | 50.60 ± 3.78 / 13.45 ± 3.00 / 35.95 ± 5.05 | 85.75 ± 2.23 | 41.86 ± 4.88 / 16.27 ± 9.75 | +5.00 ± 4.06 [2.98, 7.01], 0.00134 | -24.53 ± 4.33 [-26.81, -22.25], 1.31e-11 |
| VNM V, signed_90, mixed_reply, 8 paths | 45.23 ± 10.84 / 16.96 ± 1.78 / 37.82 ± 11.20 | 72.33 ± 1.91 | 36.60 ± 4.78 / 26.80 ± 9.57 | +10.37 ± 6.91 [6.78, 13.96], 0.000656 | -11.12 ± 2.40 [-12.38, -9.86], 1.68e-10 |
| Direct V | 43.77 ± 6.94 / 13.69 ± 2.10 / 42.54 ± 7.44 | 78.08 ± 1.68 | 38.19 ± 4.52 / 23.61 ± 9.03 | +11.83 ± 4.16 [9.76, 13.90], 7.76e-11 | -16.86 ± 2.38 [-18.09, -15.63], 1.92e-15 |
| VNM Q, own_90, mixed_reply, 8 paths | 42.00 ± 7.66 / 19.47 ± 3.04 / 38.52 ± 9.83 | 75.39 ± 4.86 | 31.98 ± 6.47 / 36.04 ± 12.94 | +13.60 ± 4.38 [11.40, 15.79], 6.5e-11 | -14.17 ± 6.93 [-17.84, -10.50], 4.28e-05 |
| VNM V, own_90, mixed_reply, 8 paths | 40.47 ± 8.70 / 21.62 ± 2.58 / 37.92 ± 10.09 | 68.49 ± 1.55 | 30.32 ± 2.70 / 39.36 ± 5.40 | +15.13 ± 5.53 [12.39, 17.88], 1.46e-10 | -7.27 ± 3.24 [-8.95, -5.59], 3.63e-06 |
| VNM Q, unsigned_90, uniform_both, 8 paths | 30.07 ± 3.08 / 20.49 ± 2.95 / 49.44 ± 4.79 | 68.95 ± 3.16 | 30.84 ± 6.51 / 38.31 ± 13.02 | +25.53 ± 4.43 [23.31, 27.74], 2.05e-18 | -7.73 ± 5.21 [-10.48, -4.99], 0.00145 |
| VNM V, unsigned_90, mixed_reply, 8 paths | 28.21 ± 3.27 / 22.89 ± 2.73 / 48.90 ± 4.59 | 53.60 ± 1.71 | 29.30 ± 4.64 / 41.41 ± 9.27 | +27.38 ± 4.53 [25.16, 29.61], 5.64e-23 | +7.61 ± 3.85 [5.60, 9.63], 3.56e-05 |

## Scores on actual external games

Win minus loss subtracts each fitted player's mean utility on its lost games from the mean on its won games. Per turn is the within-fit least-squares change in utility per additional observed turn. 'unsigned_last' is the raw fitted frequency of the last played move. 'chooser_moved_last' is 1 if the evaluated seat made that move, otherwise 0. 'turns' counts the remaining turns including mandatory passes. Slopes and differences are averaged per original first. All diagnostic contrasts share a separate Holm adjustment. Missing outcomes are omitted from the corresponding fit-specific mean. Counts of observed wins, draws and losses and the fit-level values are in the path diagnostics file.

| Game (originals) | Measure | Outcome or association | Mean ± spread [95%], adjusted p for differences and slopes | Contributing fits |
|---|---|---|---|---|
| tictactoe (24) | chooser_moved_last | draw | +0.7158 ± 0.1298 [0.6610, 0.7706], — | 96 |
| tictactoe (24) | chooser_moved_last | loss | +0.0000 ± 0.0000 [0.0000, 0.0000], — | 89 |
| tictactoe (24) | chooser_moved_last | per_turn | -0.0161 ± 0.0556 [-0.0396, 0.0074], 1 | 96 |
| tictactoe (24) | chooser_moved_last | win | +1.0000 ± 0.0000 [1.0000, 1.0000], — | 96 |
| tictactoe (24) | chooser_moved_last | win_minus_loss | +1.0000 ± 0.0000 [1.0000, 1.0000], — | 89 |
| tictactoe (24) | signed_90 | draw | +0.1013 ± 0.0949 [0.0612, 0.1413], — | 96 |
| tictactoe (24) | signed_90 | loss | -0.2881 ± 0.1473 [-0.3503, -0.2259], — | 89 |
| tictactoe (24) | signed_90 | per_turn | -0.0668 ± 0.0430 [-0.0850, -0.0487], 2.67e-06 | 96 |
| tictactoe (24) | signed_90 | win | +0.5800 ± 0.0699 [0.5504, 0.6095], — | 96 |
| tictactoe (24) | signed_90 | win_minus_loss | +0.8683 ± 0.1660 [0.7982, 0.9384], 6.99e-17 | 89 |
| tictactoe (24) | signed_last | draw | +0.2739 ± 0.1529 [0.2093, 0.3385], — | 96 |
| tictactoe (24) | signed_last | loss | -0.5159 ± 0.0633 [-0.5426, -0.4892], — | 89 |
| tictactoe (24) | signed_last | per_turn | -0.0470 ± 0.0730 [-0.0778, -0.0161], 0.0536 | 96 |
| tictactoe (24) | signed_last | win | +0.7876 ± 0.0246 [0.7773, 0.7980], — | 96 |
| tictactoe (24) | signed_last | win_minus_loss | +1.3035 ± 0.0661 [1.2755, 1.3314], 6.09e-30 | 89 |
| tictactoe (24) | signed_last_90 | draw | +0.1179 ± 0.0658 [0.0901, 0.1457], — | 96 |
| tictactoe (24) | signed_last_90 | loss | -0.2845 ± 0.0329 [-0.2984, -0.2706], — | 89 |
| tictactoe (24) | signed_last_90 | per_turn | -0.0465 ± 0.0378 [-0.0625, -0.0305], 8.05e-05 | 96 |
| tictactoe (24) | signed_last_90 | win | +0.4358 ± 0.0140 [0.4299, 0.4417], — | 96 |
| tictactoe (24) | signed_last_90 | win_minus_loss | +0.7202 ± 0.0394 [0.7036, 0.7368], 3.24e-29 | 89 |
| tictactoe (24) | turns | draw | +9.0000 ± 0.0000 [9.0000, 9.0000], — | 96 |
| tictactoe (24) | turns | loss | +6.5768 ± 0.3139 [6.4442, 6.7093], — | 89 |
| tictactoe (24) | turns | win | +6.6059 ± 0.2395 [6.5047, 6.7070], — | 96 |
| tictactoe (24) | turns | win_minus_loss | +0.0295 ± 0.4661 [-0.1673, 0.2263], 1 | 89 |
| tictactoe (24) | unsigned_90 | draw | +1.8458 ± 0.1995 [1.7616, 1.9300], — | 96 |
| tictactoe (24) | unsigned_90 | loss | +1.0803 ± 0.1777 [1.0053, 1.1554], — | 89 |
| tictactoe (24) | unsigned_90 | per_turn | +0.2105 ± 0.0398 [0.1937, 0.2273], 5.56e-17 | 96 |
| tictactoe (24) | unsigned_90 | win | +1.3114 ± 0.1139 [1.2633, 1.3595], — | 96 |
| tictactoe (24) | unsigned_90 | win_minus_loss | +0.2339 ± 0.1703 [0.1620, 0.3058], 1.68e-05 | 89 |
| tictactoe (24) | unsigned_last | draw | +0.5452 ± 0.0658 [0.5175, 0.5730], — | 96 |
| tictactoe (24) | unsigned_last | loss | +0.5159 ± 0.0633 [0.4892, 0.5426], — | 89 |
| tictactoe (24) | unsigned_last | per_turn | -0.0272 ± 0.0299 [-0.0398, -0.0145], 0.00274 | 96 |
| tictactoe (24) | unsigned_last | win | +0.7876 ± 0.0246 [0.7773, 0.7980], — | 96 |
| tictactoe (24) | unsigned_last | win_minus_loss | +0.2717 ± 0.0712 [0.2416, 0.3018], 6.95e-14 | 89 |
| connect4 (7) | chooser_moved_last | draw | +0.8571 ± 0.3780 [0.5076, 1.2067], — | 8 |
| connect4 (16) | chooser_moved_last | loss | +0.0000 ± 0.0000 [0.0000, 0.0000], — | 64 |
| connect4 (16) | chooser_moved_last | per_turn | -0.0101 ± 0.0171 [-0.0192, -0.0010], 0.292 | 64 |
| connect4 (16) | chooser_moved_last | win | +1.0000 ± 0.0000 [1.0000, 1.0000], — | 64 |
| connect4 (16) | chooser_moved_last | win_minus_loss | +1.0000 ± 0.0000 [1.0000, 1.0000], — | 64 |
| connect4 (7) | signed_90 | draw | +0.6345 ± 1.0092 [-0.2989, 1.5678], — | 8 |
| connect4 (16) | signed_90 | loss | -0.1239 ± 0.5758 [-0.4308, 0.1829], — | 64 |
| connect4 (16) | signed_90 | per_turn | -0.0276 ± 0.0144 [-0.0352, -0.0199], 3.17e-05 | 64 |
| connect4 (16) | signed_90 | win | +0.2607 ± 0.1908 [0.1590, 0.3623], — | 64 |
| connect4 (16) | signed_90 | win_minus_loss | +0.3846 ± 0.4439 [0.1481, 0.6211], 0.045 | 64 |
| connect4 (7) | signed_last | draw | +0.4969 ± 0.4268 [0.1021, 0.8916], — | 8 |
| connect4 (16) | signed_last | loss | -0.3785 ± 0.3317 [-0.5553, -0.2018], — | 64 |
| connect4 (16) | signed_last | per_turn | -0.0159 ± 0.0207 [-0.0269, -0.0049], 0.0854 | 64 |
| connect4 (16) | signed_last | win | +0.7397 ± 0.1765 [0.6457, 0.8338], — | 64 |
| connect4 (16) | signed_last | win_minus_loss | +1.1182 ± 0.5026 [0.8504, 1.3861], 5.67e-06 | 64 |
| connect4 (7) | signed_last_90 | draw | +0.0088 ± 0.0073 [0.0021, 0.0156], — | 8 |
| connect4 (16) | signed_last_90 | loss | -0.0778 ± 0.0679 [-0.1140, -0.0416], — | 64 |
| connect4 (16) | signed_last_90 | per_turn | -0.0143 ± 0.0085 [-0.0188, -0.0097], 0.00014 | 64 |
| connect4 (16) | signed_last_90 | win | +0.2652 ± 0.0829 [0.2210, 0.3094], — | 64 |
| connect4 (16) | signed_last_90 | win_minus_loss | +0.3430 ± 0.1459 [0.2652, 0.4207], 2.9e-06 | 64 |
| connect4 (7) | turns | draw | +38.4286 ± 1.9024 [36.6692, 40.1880], — | 8 |
| connect4 (16) | turns | loss | +17.3838 ± 4.3300 [15.0765, 19.6911], — | 64 |
| connect4 (16) | turns | win | +12.4943 ± 1.1639 [11.8741, 13.1145], — | 64 |
| connect4 (16) | turns | win_minus_loss | -4.8895 ± 5.2949 [-7.7109, -2.0681], 0.0303 | 64 |
| connect4 (7) | unsigned_90 | draw | +2.0197 ± 0.6250 [1.4417, 2.5977], — | 8 |
| connect4 (16) | unsigned_90 | loss | +1.6538 ± 0.5688 [1.3507, 1.9569], — | 64 |
| connect4 (16) | unsigned_90 | per_turn | +0.0408 ± 0.0144 [0.0331, 0.0485], 2.67e-07 | 64 |
| connect4 (16) | unsigned_90 | win | +1.5238 ± 0.2269 [1.4029, 1.6447], — | 64 |
| connect4 (16) | unsigned_90 | win_minus_loss | -0.1300 ± 0.4670 [-0.3789, 0.1188], 1 | 64 |
| connect4 (7) | unsigned_last | draw | +0.5459 ± 0.3500 [0.2222, 0.8696], — | 8 |
| connect4 (16) | unsigned_last | loss | +0.3785 ± 0.3317 [0.2018, 0.5553], — | 64 |
| connect4 (16) | unsigned_last | per_turn | -0.0003 ± 0.0093 [-0.0053, 0.0047], 1 | 64 |
| connect4 (16) | unsigned_last | win | +0.7397 ± 0.1765 [0.6457, 0.8338], — | 64 |
| connect4 (16) | unsigned_last | win_minus_loss | +0.3612 ± 0.1723 [0.2694, 0.4530], 1.15e-05 | 64 |
| reversi4 (16) | chooser_moved_last | draw | +0.8828 ± 0.1240 [0.8167, 0.9489], — | 64 |
| reversi4 (16) | chooser_moved_last | loss | +0.5549 ± 0.0807 [0.5119, 0.5980], — | 64 |
| reversi4 (16) | chooser_moved_last | per_turn | -0.0012 ± 0.0232 [-0.0136, 0.0112], 1 | 64 |
| reversi4 (16) | chooser_moved_last | win | +0.9813 ± 0.0155 [0.9731, 0.9896], — | 64 |
| reversi4 (16) | chooser_moved_last | win_minus_loss | +0.4264 ± 0.0771 [0.3853, 0.4675], 2.32e-11 | 64 |
| reversi4 (16) | signed_90 | draw | +0.0973 ± 0.1910 [-0.0045, 0.1991], — | 64 |
| reversi4 (16) | signed_90 | loss | -0.0762 ± 0.1125 [-0.1362, -0.0163], — | 64 |
| reversi4 (16) | signed_90 | per_turn | -0.0410 ± 0.0272 [-0.0554, -0.0265], 0.000412 | 64 |
| reversi4 (16) | signed_90 | win | -0.0380 ± 0.0881 [-0.0849, 0.0090], — | 64 |
| reversi4 (16) | signed_90 | win_minus_loss | +0.0383 ± 0.1341 [-0.0332, 0.1097], 1 | 64 |
| reversi4 (16) | signed_last | draw | +0.6855 ± 0.2399 [0.5576, 0.8133], — | 64 |
| reversi4 (16) | signed_last | loss | +0.1622 ± 0.1422 [0.0864, 0.2379], — | 64 |
| reversi4 (16) | signed_last | per_turn | +0.0010 ± 0.0363 [-0.0183, 0.0203], 1 | 64 |
| reversi4 (16) | signed_last | win | +0.8081 ± 0.0444 [0.7845, 0.8318], — | 64 |
| reversi4 (16) | signed_last | win_minus_loss | +0.6459 ± 0.1451 [0.5686, 0.7232], 5.06e-10 | 64 |
| reversi4 (16) | signed_last_90 | draw | +0.2620 ± 0.0984 [0.2095, 0.3145], — | 64 |
| reversi4 (16) | signed_last_90 | loss | +0.0554 ± 0.0495 [0.0290, 0.0817], — | 64 |
| reversi4 (16) | signed_last_90 | per_turn | -0.0164 ± 0.0120 [-0.0228, -0.0099], 0.00117 | 64 |
| reversi4 (16) | signed_last_90 | win | +0.3067 ± 0.0172 [0.2976, 0.3159], — | 64 |
| reversi4 (16) | signed_last_90 | win_minus_loss | +0.2514 ± 0.0531 [0.2231, 0.2797], 2.15e-10 | 64 |
| reversi4 (16) | turns | draw | +10.0928 ± 0.9266 [9.5991, 10.5866], — | 64 |
| reversi4 (16) | turns | loss | +10.7730 ± 0.2616 [10.6336, 10.9123], — | 64 |
| reversi4 (16) | turns | win | +10.3485 ± 0.3893 [10.1410, 10.5559], — | 64 |
| reversi4 (16) | turns | win_minus_loss | -0.4245 ± 0.5551 [-0.7203, -0.1287], 0.0854 | 64 |
| reversi4 (16) | unsigned_90 | draw | +3.0009 ± 0.2773 [2.8531, 3.1487], — | 64 |
| reversi4 (16) | unsigned_90 | loss | +3.2718 ± 0.1557 [3.1888, 3.3548], — | 64 |
| reversi4 (16) | unsigned_90 | per_turn | +0.0888 ± 0.0240 [0.0760, 0.1016], 6.8e-09 | 64 |
| reversi4 (16) | unsigned_90 | win | +3.0528 ± 0.1480 [2.9739, 3.1316], — | 64 |
| reversi4 (16) | unsigned_90 | win_minus_loss | -0.2190 ± 0.1697 [-0.3094, -0.1286], 0.00185 | 64 |
| reversi4 (16) | unsigned_last | draw | +0.8864 ± 0.0459 [0.8619, 0.9108], — | 64 |
| reversi4 (16) | unsigned_last | loss | +0.8270 ± 0.0416 [0.8049, 0.8492], — | 64 |
| reversi4 (16) | unsigned_last | per_turn | +0.0091 ± 0.0059 [0.0060, 0.0123], 0.000332 | 64 |
| reversi4 (16) | unsigned_last | win | +0.8403 ± 0.0306 [0.8240, 0.8566], — | 64 |
| reversi4 (16) | unsigned_last | win_minus_loss | +0.0133 ± 0.0384 [-0.0072, 0.0337], 1 | 64 |
