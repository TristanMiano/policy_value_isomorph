# Discovery results for visit-trained game scores

The frozen original takes its highest-scoring legal move. A network scoring the resulting board (V) or the board and move (Q) estimates the fraction of observed visits taking a move. Direct selection takes the largest immediate fit. Von Neumann–Morgenstern (VNM) selection maximizes expected complete-game score, replanning on each actual turn. Ties prefer the lowest move index. 'copies' projects highest fitted choices for both seats. The rule 'uniform_reply' projects greedy own moves against uniform legal replies. The rule 'mixed_reply' mixes those two opponents equally once per path. The rule 'uniform_both' projects uniform legal moves for both seats. Each random component gets the listed number of full paths, with exact expectations in tic-tac-toe. The assisted comparison includes an opponent considering tactics and one using positional priorities. The supplied-outcome reference adds actual game results and projects guarded own play against four opponent types.

The utility name specifies the score of a completed game from the current choosing seat. 'signed' counts own moves positively and opposing moves negatively, 'unsigned' counts both positively, and 'own' counts only own moves. 'last' counts the final fitted frequency. The rule 'sum' adds the whole path. The suffix 50, 75, 90 or 97 gives the discount per turn in percent, starting at weight one. 'sqrt_last' and 'square_last' apply the square root or square to the final frequency, signed by mover. 'last_plus_sum' adds the signed last frequency and 0.1 times the signed 0.9-discounted sum. A constant score uses 0.5 in the identical formula. Constant continuations also break all greedy choices by move index. Both constants induce the identical player when all continuing moves are uniformly random.

Win, draw and loss percentages are seat-balanced against four equally weighted external opponent groups. Agreement is the weighted percentage of withheld boards on which the reconstruction chose the original's favorite. Exact-copy self-play uses two identical deployed players and counts both seats. Values give means ± sample standard deviations across originals after averaging their four fits. The equal-game spread describes the average of one independent original per named game. The selected rule is VNM V, signed_last_90, mixed_reply, 8 paths.

Independent original counts for tic-tac-toe, Connect Four and 4×4 Reversi are 8, 8, 8.
For paired changes, brackets give individual 95% confidence intervals, followed by the two-sided p-value for a zero mean difference, Holm-adjusted across the table's win and agreement comparisons in all four populations. A dash means the across-original variance is exactly zero. Discovery contrasts are exploratory because the leader was chosen on those data. The compressed estimates and contrasts alongside this table contain all unrounded measurements, raw p-values, original-level differences, and intervals resampling the independent originals.

## External win rate (%)

| Rule | Tic-tac-toe | Connect Four | 4×4 Reversi | Equal games |
|---|---|---|---|---|
| VNM V, signed_last_90, mixed_reply, 8 paths | 50.42 ± 3.71 | 75.49 ± 13.78 | 41.02 ± 3.85 | 55.64 ± 4.93 |
| VNM V, signed_last_90, uniform_both, 8 paths | 47.38 ± 1.63 | 72.02 ± 9.95 | 45.70 ± 4.68 | 55.04 ± 3.71 |
| VNM Q, signed_last_90, uniform_both, 8 paths | 47.28 ± 1.62 | 70.46 ± 10.26 | 45.80 ± 4.46 | 54.51 ± 3.77 |
| VNM Q, signed_last_90, mixed_reply, 8 paths | 46.13 ± 3.00 | 72.66 ± 15.21 | 44.73 ± 4.02 | 54.50 ± 5.34 |
| VNM Q, signed_50, mixed_reply, 8 paths | 44.39 ± 5.04 | 63.43 ± 13.08 | 54.35 ± 4.01 | 54.05 ± 4.86 |
| VNM Q, signed_50, copies, 8 paths | 43.35 ± 5.16 | 63.57 ± 13.26 | 55.13 ± 3.45 | 54.02 ± 4.88 |
| VNM Q, signed_last_90, uniform_reply, 8 paths | 46.26 ± 2.92 | 69.78 ± 9.08 | 45.56 ± 5.42 | 53.87 ± 3.66 |
| VNM Q, signed_50, uniform_reply, 8 paths | 44.41 ± 4.07 | 62.11 ± 12.28 | 53.32 ± 4.29 | 53.28 ± 4.54 |
| VNM V, last_plus_sum, mixed_reply, 8 paths | 50.59 ± 5.76 | 67.48 ± 20.62 | 40.43 ± 4.16 | 52.83 ± 7.27 |
| VNM Q, signed_50, uniform_both, 8 paths | 44.57 ± 4.26 | 61.96 ± 12.21 | 51.17 ± 2.93 | 52.57 ± 4.42 |
| VNM Q, last_plus_sum, mixed_reply, 8 paths | 45.36 ± 3.70 | 65.72 ± 17.82 | 46.29 ± 6.09 | 52.46 ± 6.40 |
| VNM V, signed_last_90, uniform_reply, 8 paths | 47.59 ± 1.59 | 64.06 ± 11.60 | 45.56 ± 3.33 | 52.40 ± 4.06 |
| VNM Q, last_plus_sum, uniform_reply, 8 paths | 47.86 ± 4.16 | 64.26 ± 10.59 | 44.92 ± 4.94 | 52.35 ± 4.13 |
| VNM Q, signed_90, mixed_reply, 8 paths | 46.90 ± 4.36 | 62.84 ± 14.48 | 46.83 ± 5.65 | 52.19 ± 5.38 |
| VNM V, last_plus_sum, uniform_both, 8 paths | 45.83 ± 2.01 | 65.87 ± 12.26 | 44.73 ± 4.45 | 52.14 ± 4.40 |
| VNM V, signed_last, mixed_reply, 8 paths | 49.74 ± 5.58 | 64.79 ± 18.29 | 40.58 ± 3.06 | 51.70 ± 6.45 |
| VNM Q, signed_last, mixed_reply, 8 paths | 45.01 ± 3.37 | 64.70 ± 14.43 | 45.21 ± 4.49 | 51.64 ± 5.16 |
| VNM Q, last_plus_sum, uniform_both, 8 paths | 46.35 ± 1.53 | 62.94 ± 11.04 | 44.97 ± 2.87 | 51.42 ± 3.84 |
| VNM V, signed_last, uniform_both, 8 paths | 45.24 ± 2.35 | 64.79 ± 9.21 | 43.90 ± 3.25 | 51.31 ± 3.35 |
| VNM Q, signed_90, copies, 8 paths | 45.59 ± 5.54 | 64.11 ± 16.39 | 44.14 ± 5.57 | 51.28 ± 6.06 |
| VNM Q, signed_last, uniform_reply, 8 paths | 46.35 ± 4.33 | 60.74 ± 7.76 | 45.95 ± 4.91 | 51.01 ± 3.38 |
| VNM V, last_plus_sum, uniform_reply, 8 paths | 50.59 ± 4.61 | 58.11 ± 9.17 | 43.80 ± 3.21 | 50.83 ± 3.58 |
| VNM V, signed_50, mixed_reply, 8 paths | 42.07 ± 2.41 | 57.32 ± 27.34 | 53.08 ± 5.94 | 50.82 ± 9.36 |
| VNM Q, unsigned_50, uniform_both, 8 paths | 38.59 ± 5.16 | 59.72 ± 12.61 | 54.15 ± 3.76 | 50.82 ± 4.71 |
| VNM Q, signed_last, uniform_both, 8 paths | 45.30 ± 1.82 | 62.35 ± 8.92 | 44.58 ± 2.64 | 50.74 ± 3.16 |
| VNM Q, unsigned_50, uniform_reply, 8 paths | 38.55 ± 4.84 | 58.64 ± 11.89 | 54.83 ± 3.68 | 50.68 ± 4.45 |
| VNM V, signed_last, uniform_reply, 8 paths | 49.95 ± 5.79 | 56.98 ± 7.53 | 44.82 ± 3.61 | 50.58 ± 3.39 |
| VNM V, signed_50, copies, 8 paths | 41.92 ± 2.21 | 56.69 ± 28.10 | 52.93 ± 6.51 | 50.51 ± 9.64 |
| VNM V, signed_last_90, copies, 8 paths | 46.95 ± 3.89 | 69.97 ± 19.88 | 34.18 ± 2.87 | 50.37 ± 6.82 |
| VNM Q, signed_90, uniform_both, 8 paths | 45.10 ± 3.83 | 61.82 ± 12.31 | 42.63 ± 5.03 | 49.85 ± 4.61 |
| VNM Q, unsigned_50, mixed_reply, 8 paths | 37.88 ± 3.61 | 54.20 ± 17.48 | 54.83 ± 4.02 | 48.97 ± 6.10 |
| VNM V, signed_50, uniform_reply, 8 paths | 42.48 ± 2.48 | 51.56 ± 23.25 | 52.05 ± 4.72 | 48.70 ± 7.95 |
| VNM V, signed_50, uniform_both, 8 paths | 41.94 ± 2.61 | 51.12 ± 22.92 | 50.15 ± 6.25 | 47.74 ± 7.97 |
| VNM Q, signed_90, uniform_reply, 8 paths | 46.71 ± 5.47 | 53.76 ± 17.09 | 42.38 ± 3.22 | 47.62 ± 6.08 |
| VNM Q, own_90, uniform_both, 8 paths | 37.15 ± 4.11 | 56.54 ± 11.42 | 49.02 ± 4.73 | 47.57 ± 4.34 |
| VNM V, signed_90, copies, 8 paths | 47.47 ± 5.11 | 55.71 ± 34.94 | 37.01 ± 4.70 | 46.73 ± 11.87 |
| VNM Q, signed_last_90, copies, 8 paths | 41.68 ± 4.42 | 58.94 ± 20.42 | 38.09 ± 1.91 | 46.23 ± 6.99 |
| VNM V, last_plus_sum, copies, 8 paths | 45.68 ± 5.52 | 59.57 ± 23.68 | 32.23 ± 5.24 | 45.83 ± 8.29 |
| VNM V, signed_90, mixed_reply, 8 paths | 47.72 ± 5.50 | 50.54 ± 34.48 | 39.01 ± 5.65 | 45.76 ± 11.79 |
| VNM Q, signed_sum, mixed_reply, 8 paths | 47.48 ± 4.89 | 51.27 ± 19.99 | 38.13 ± 5.67 | 45.63 ± 7.12 |
| VNM Q, signed_sum, uniform_both, 8 paths | 45.01 ± 3.96 | 53.47 ± 12.28 | 35.84 ± 5.26 | 44.77 ± 4.64 |
| VNM V, signed_sum, mixed_reply, 8 paths | 50.04 ± 6.43 | 47.71 ± 27.59 | 35.74 ± 5.89 | 44.50 ± 9.64 |
| VNM V, signed_last, copies, 8 paths | 44.91 ± 4.85 | 55.57 ± 19.79 | 32.62 ± 2.25 | 44.37 ± 6.83 |
| VNM Q, signed_sum, copies, 8 paths | 43.83 ± 6.36 | 50.05 ± 20.92 | 38.13 ± 6.01 | 44.00 ± 7.56 |
| VNM V, signed_sum, copies, 8 paths | 47.09 ± 5.33 | 50.15 ± 29.93 | 33.98 ± 6.00 | 43.74 ± 10.33 |
| VNM Q, unsigned_50, copies, 8 paths | 36.12 ± 3.87 | 39.84 ± 29.90 | 53.61 ± 4.36 | 43.19 ± 10.16 |
| VNM Q, own_90, mixed_reply, 8 paths | 38.22 ± 3.40 | 35.11 ± 25.22 | 55.71 ± 2.82 | 43.01 ± 8.53 |
| VNM Q, own_90, uniform_reply, 8 paths | 37.51 ± 3.61 | 35.01 ± 15.39 | 55.62 ± 2.31 | 42.71 ± 5.33 |
| VNM Q, last_plus_sum, copies, 8 paths | 39.99 ± 5.10 | 50.59 ± 18.94 | 37.26 ± 3.49 | 42.61 ± 6.64 |
| VNM V, unsigned_50, uniform_reply, 8 paths | 32.86 ± 5.33 | 40.14 ± 23.90 | 54.00 ± 4.71 | 42.33 ± 8.31 |
| VNM V, signed_90, uniform_both, 8 paths | 43.51 ± 3.23 | 46.58 ± 25.01 | 36.57 ± 4.01 | 42.22 ± 8.51 |
| VNM Q, own_90, copies, 8 paths | 39.41 ± 2.99 | 33.25 ± 27.24 | 53.22 ± 4.38 | 41.96 ± 9.25 |
| VNM V, signed_90, uniform_reply, 8 paths | 43.58 ± 4.09 | 43.12 ± 24.09 | 38.43 ± 4.00 | 41.71 ± 8.25 |
| VNM Q, signed_last, copies, 8 paths | 40.10 ± 4.38 | 46.68 ± 18.16 | 37.50 ± 2.01 | 41.43 ± 6.26 |
| VNM V, unsigned_50, uniform_both, 8 paths | 32.33 ± 4.83 | 38.87 ± 20.90 | 52.93 ± 4.22 | 41.37 ± 7.29 |
| VNM Q, signed_sum, uniform_reply, 8 paths | 45.38 ± 3.38 | 39.11 ± 8.34 | 39.16 ± 4.31 | 41.22 ± 3.33 |
| VNM V, unsigned_50, mixed_reply, 8 paths | 33.29 ± 5.11 | 35.35 ± 24.53 | 53.42 ± 4.81 | 40.69 ± 8.50 |
| VNM V, own_90, mixed_reply, 8 paths | 37.48 ± 4.86 | 38.43 ± 25.83 | 45.31 ± 7.04 | 40.41 ± 9.07 |
| VNM V, signed_sum, uniform_both, 8 paths | 42.97 ± 3.04 | 42.09 ± 22.43 | 34.33 ± 5.49 | 39.80 ± 7.76 |
| VNM V, own_90, copies, 8 paths | 37.35 ± 4.42 | 39.40 ± 28.90 | 40.72 ± 8.97 | 39.16 ± 10.19 |
| VNM V, unsigned_50, copies, 8 paths | 31.72 ± 4.70 | 32.13 ± 25.99 | 52.49 ± 4.70 | 38.78 ± 8.94 |
| VNM V, signed_sum, uniform_reply, 8 paths | 43.08 ± 3.79 | 35.89 ± 13.28 | 35.89 ± 3.48 | 38.28 ± 4.75 |
| VNM V, own_90, uniform_reply, 8 paths | 35.64 ± 4.10 | 31.05 ± 14.86 | 47.85 ± 3.62 | 38.18 ± 5.28 |
| VNM V, own_90, uniform_both, 8 paths | 35.37 ± 4.57 | 35.89 ± 22.34 | 43.26 ± 5.89 | 38.17 ± 7.85 |
| VNM V, unsigned_last, uniform_reply, 8 paths | 33.30 ± 3.10 | 49.61 ± 7.25 | 31.40 ± 4.60 | 38.10 ± 3.04 |
| VNM Q, unsigned_last, uniform_reply, 8 paths | 29.23 ± 2.20 | 50.10 ± 6.09 | 33.25 ± 6.45 | 37.53 ± 3.04 |
| VNM V, unsigned_last, mixed_reply, 8 paths | 27.95 ± 3.12 | 41.70 ± 11.31 | 31.25 ± 4.03 | 33.63 ± 4.14 |
| VNM Q, unsigned_last, mixed_reply, 8 paths | 26.35 ± 2.44 | 42.92 ± 8.99 | 29.39 ± 5.59 | 32.89 ± 3.62 |
| VNM Q, unsigned_90, uniform_both, 8 paths | 23.62 ± 3.14 | 21.04 ± 9.31 | 51.03 ± 6.10 | 31.90 ± 3.85 |
| VNM Q, unsigned_90, uniform_reply, 8 paths | 23.11 ± 2.75 | 13.43 ± 3.52 | 55.66 ± 2.16 | 30.73 ± 1.65 |
| VNM Q, unsigned_90, mixed_reply, 8 paths | 24.29 ± 2.05 | 14.75 ± 8.00 | 52.69 ± 3.71 | 30.58 ± 3.02 |
| VNM Q, unsigned_90, copies, 8 paths | 24.47 ± 2.63 | 15.04 ± 8.37 | 50.20 ± 4.59 | 29.90 ± 3.30 |
| VNM Q, unsigned_last, uniform_both, 8 paths | 25.98 ± 3.26 | 35.69 ± 3.05 | 27.93 ± 3.62 | 29.87 ± 1.92 |
| VNM V, unsigned_last, uniform_both, 8 paths | 24.19 ± 2.71 | 34.38 ± 5.97 | 28.71 ± 5.95 | 29.09 ± 2.95 |
| VNM Q, unsigned_sum, uniform_reply, 8 paths | 19.25 ± 1.82 | 12.55 ± 3.53 | 54.83 ± 3.71 | 28.88 ± 1.81 |
| VNM V, unsigned_90, mixed_reply, 8 paths | 20.95 ± 3.73 | 18.85 ± 6.36 | 46.68 ± 5.13 | 28.83 ± 2.99 |
| VNM V, unsigned_90, uniform_reply, 8 paths | 18.53 ± 4.20 | 15.09 ± 3.17 | 51.42 ± 3.95 | 28.34 ± 2.19 |
| VNM V, unsigned_90, uniform_both, 8 paths | 20.62 ± 4.87 | 16.41 ± 6.19 | 47.36 ± 4.60 | 28.13 ± 3.04 |
| VNM Q, unsigned_sum, mixed_reply, 8 paths | 20.55 ± 1.81 | 14.21 ± 3.24 | 49.27 ± 2.23 | 28.01 ± 1.44 |
| VNM V, unsigned_last, copies, 8 paths | 21.61 ± 2.83 | 31.30 ± 11.44 | 29.98 ± 4.85 | 27.63 ± 4.25 |
| VNM V, unsigned_90, copies, 8 paths | 20.50 ± 3.01 | 18.16 ± 10.65 | 42.63 ± 5.05 | 27.10 ± 4.06 |
| VNM Q, unsigned_last, copies, 8 paths | 23.64 ± 2.31 | 27.78 ± 10.52 | 29.83 ± 5.81 | 27.09 ± 4.08 |
| VNM V, unsigned_sum, mixed_reply, 8 paths | 18.98 ± 3.25 | 16.55 ± 4.73 | 45.65 ± 3.72 | 27.06 ± 2.28 |
| VNM Q, unsigned_sum, uniform_both, 8 paths | 19.91 ± 3.77 | 14.75 ± 4.04 | 45.70 ± 5.95 | 26.79 ± 2.71 |
| VNM V, unsigned_sum, uniform_reply, 8 paths | 17.26 ± 3.08 | 12.45 ± 3.89 | 50.39 ± 5.96 | 26.70 ± 2.59 |
| VNM V, unsigned_sum, copies, 8 paths | 19.29 ± 3.13 | 19.38 ± 9.28 | 38.82 ± 4.77 | 25.83 ± 3.63 |
| VNM V, unsigned_sum, uniform_both, 8 paths | 19.49 ± 4.11 | 13.67 ± 3.68 | 42.87 ± 5.16 | 25.35 ± 2.52 |
| VNM Q, unsigned_sum, copies, 8 paths | 20.69 ± 2.91 | 12.99 ± 6.53 | 42.24 ± 4.46 | 25.31 ± 2.81 |
