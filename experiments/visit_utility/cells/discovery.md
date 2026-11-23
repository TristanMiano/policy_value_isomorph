# Discovery results for visit-trained game scores

The frozen original takes its highest-scoring legal move. A network scoring the resulting board (V) or the board and move (Q) estimates the fraction of observed visits taking a move. Direct selection takes the largest immediate fit. Von Neumann–Morgenstern (VNM) selection maximizes expected complete-game score, replanning on each actual turn. Ties prefer the lowest move index. 'copies' projects highest fitted choices for both seats. The rule 'uniform_reply' projects greedy own moves against uniform legal replies. The rule 'mixed_reply' mixes those two opponents equally once per path. The rule 'uniform_both' projects uniform legal moves for both seats. Each random component gets the listed number of full paths, with exact expectations in tic-tac-toe. The assisted comparison includes an opponent considering tactics and one using positional priorities. The supplied-outcome reference adds actual game results and projects guarded own play against four opponent types.

The utility name specifies the score of a completed game from the current choosing seat. 'signed' counts own moves positively and opposing moves negatively, 'unsigned' counts both positively, and 'own' counts only own moves. 'last' counts the final fitted frequency. The rule 'sum' adds the whole path. The suffix 50, 75, 90 or 97 gives the discount per turn in percent, starting at weight one. 'sqrt_last' and 'square_last' apply the square root or square to the final frequency, signed by mover. 'last_plus_sum' adds the signed last frequency and 0.1 times the signed 0.9-discounted sum. A constant score uses 0.5 in the identical formula. Constant continuations also break all greedy choices by move index. Both constants induce the identical player when all continuing moves are uniformly random.

Win, draw and loss percentages are seat-balanced against four equally weighted external opponent groups. Agreement is the weighted percentage of withheld boards on which the reconstruction chose the original's favorite. Exact-copy self-play uses two identical deployed players and counts both seats. Values give means ± sample standard deviations across originals after averaging their four fits. The equal-game spread describes the average of one independent original per named game. The selected rule is VNM Q, signed_last_90, uniform_both, 8 paths.

Independent original counts for tic-tac-toe, Connect Four and 4×4 Reversi are 8, 8, 8.
For paired changes, brackets give individual 95% confidence intervals, followed by the two-sided p-value for a zero mean difference, Holm-adjusted across the table's win and agreement comparisons in all four populations. A dash means the across-original variance is exactly zero. Discovery contrasts are exploratory because the leader was chosen on those data. The compressed estimates and contrasts alongside this table contain all unrounded measurements, raw p-values, original-level differences, and intervals resampling the independent originals.

## External win rate (%)

| Rule | Tic-tac-toe | Connect Four | 4×4 Reversi | Equal games |
|---|---|---|---|---|
| VNM Q, signed_last_90, uniform_both, 8 paths | 46.55 ± 1.41 | 67.92 ± 7.22 | 47.02 ± 4.02 | 53.83 ± 2.79 |
| VNM V, signed_last_90, uniform_both, 8 paths | 45.96 ± 1.73 | 67.68 ± 8.95 | 43.85 ± 2.61 | 52.49 ± 3.16 |
| VNM V, signed_last_90, mixed_reply, 8 paths | 44.16 ± 1.83 | 70.12 ± 11.58 | 41.75 ± 5.18 | 52.01 ± 4.27 |
| VNM Q, signed_last_90, mixed_reply, 8 paths | 43.38 ± 2.39 | 66.89 ± 8.57 | 41.89 ± 2.84 | 50.72 ± 3.11 |
| VNM V, last_plus_sum, uniform_both, 8 paths | 42.36 ± 1.46 | 65.43 ± 10.94 | 43.75 ± 3.33 | 50.51 ± 3.84 |
| VNM Q, last_plus_sum, uniform_both, 8 paths | 44.65 ± 2.27 | 60.64 ± 8.25 | 45.80 ± 2.41 | 50.37 ± 2.96 |
| VNM Q, signed_last_90, uniform_reply, 8 paths | 45.52 ± 3.49 | 61.08 ± 5.61 | 43.85 ± 2.91 | 50.15 ± 2.41 |
| VNM Q, signed_last, uniform_both, 8 paths | 45.14 ± 1.75 | 58.74 ± 8.31 | 45.90 ± 3.53 | 49.93 ± 3.06 |
| VNM V, signed_last_90, uniform_reply, 8 paths | 43.99 ± 2.33 | 62.35 ± 5.29 | 43.31 ± 3.99 | 49.88 ± 2.34 |
| VNM V, signed_last, uniform_both, 8 paths | 42.99 ± 1.59 | 62.21 ± 8.64 | 43.12 ± 3.22 | 49.44 ± 3.12 |
| VNM Q, signed_50, copies, 8 paths | 37.17 ± 8.08 | 52.54 ± 3.08 | 56.25 ± 4.91 | 48.65 ± 3.31 |
| VNM Q, signed_90, copies, 8 paths | 37.00 ± 5.41 | 53.81 ± 5.13 | 54.25 ± 6.32 | 48.35 ± 3.26 |
| VNM Q, signed_50, mixed_reply, 8 paths | 36.39 ± 8.01 | 51.61 ± 2.72 | 55.66 ± 4.68 | 47.89 ± 3.22 |
| VNM Q, signed_90, mixed_reply, 8 paths | 37.81 ± 6.14 | 53.52 ± 4.38 | 51.51 ± 4.43 | 47.61 ± 2.92 |
| VNM Q, last_plus_sum, uniform_reply, 8 paths | 43.42 ± 3.00 | 55.22 ± 6.73 | 44.14 ± 2.46 | 47.59 ± 2.59 |
| VNM V, last_plus_sum, uniform_reply, 8 paths | 41.94 ± 2.24 | 56.40 ± 7.28 | 43.60 ± 4.19 | 47.31 ± 2.90 |
| VNM Q, signed_last, uniform_reply, 8 paths | 42.90 ± 3.39 | 54.69 ± 7.07 | 44.14 ± 1.78 | 47.24 ± 2.68 |
| VNM V, signed_last, mixed_reply, 8 paths | 41.12 ± 2.55 | 58.64 ± 10.09 | 41.75 ± 5.02 | 47.17 ± 3.85 |
| VNM V, last_plus_sum, mixed_reply, 8 paths | 41.04 ± 2.07 | 58.01 ± 12.32 | 41.94 ± 3.84 | 47.00 ± 4.36 |
| VNM Q, signed_50, uniform_both, 8 paths | 34.99 ± 7.78 | 51.42 ± 2.82 | 54.54 ± 4.73 | 46.98 ± 3.18 |
| VNM Q, signed_50, uniform_reply, 8 paths | 34.52 ± 8.02 | 51.07 ± 2.91 | 55.18 ± 4.67 | 46.92 ± 3.24 |
| VNM Q, signed_last, mixed_reply, 8 paths | 41.67 ± 2.53 | 56.49 ± 7.22 | 42.43 ± 3.37 | 46.86 ± 2.79 |
| VNM Q, last_plus_sum, mixed_reply, 8 paths | 41.55 ± 2.08 | 55.18 ± 9.59 | 42.24 ± 2.69 | 46.32 ± 3.39 |
| VNM V, signed_last, uniform_reply, 8 paths | 41.86 ± 2.01 | 53.27 ± 3.89 | 43.80 ± 4.61 | 46.31 ± 2.12 |
| VNM V, signed_50, copies, 8 paths | 38.96 ± 5.09 | 45.46 ± 22.19 | 49.41 ± 6.13 | 44.61 ± 7.86 |
| VNM V, signed_50, mixed_reply, 8 paths | 39.38 ± 4.91 | 44.58 ± 19.18 | 49.66 ± 5.41 | 44.54 ± 6.84 |
| VNM Q, signed_90, uniform_both, 8 paths | 37.53 ± 5.28 | 51.76 ± 4.04 | 43.60 ± 5.39 | 44.30 ± 2.85 |
| VNM V, signed_50, uniform_reply, 8 paths | 39.87 ± 5.13 | 41.11 ± 14.89 | 48.78 ± 4.63 | 43.25 ± 5.47 |
| VNM V, signed_last_90, copies, 8 paths | 34.94 ± 3.08 | 59.52 ± 13.09 | 32.42 ± 3.21 | 42.29 ± 4.61 |
| VNM V, signed_50, uniform_both, 8 paths | 39.97 ± 5.31 | 39.26 ± 11.24 | 47.56 ± 6.40 | 42.26 ± 4.66 |
| VNM Q, signed_90, uniform_reply, 8 paths | 36.64 ± 4.74 | 44.68 ± 8.26 | 44.43 ± 3.19 | 41.92 ± 3.35 |
| VNM Q, signed_sum, copies, 8 paths | 37.33 ± 4.40 | 42.29 ± 7.63 | 44.04 ± 7.30 | 41.22 ± 3.81 |
| VNM Q, unsigned_50, uniform_both, 8 paths | 19.15 ± 3.42 | 48.83 ± 3.15 | 55.32 ± 2.92 | 41.10 ± 1.83 |
| VNM Q, unsigned_50, uniform_reply, 8 paths | 18.99 ± 3.80 | 48.63 ± 3.13 | 55.66 ± 3.46 | 41.09 ± 2.01 |
| VNM Q, signed_last_90, copies, 8 paths | 36.53 ± 2.06 | 49.95 ± 12.98 | 36.18 ± 6.11 | 40.89 ± 4.83 |
| VNM Q, signed_sum, mixed_reply, 8 paths | 37.39 ± 5.12 | 44.09 ± 7.74 | 41.16 ± 7.65 | 40.88 ± 4.01 |
| VNM Q, own_90, uniform_both, 8 paths | 24.80 ± 4.45 | 45.26 ± 7.39 | 51.61 ± 3.45 | 40.56 ± 3.10 |
| VNM Q, signed_sum, uniform_both, 8 paths | 37.77 ± 4.52 | 45.70 ± 4.21 | 35.84 ± 6.57 | 39.77 ± 3.01 |
| VNM Q, last_plus_sum, copies, 8 paths | 34.39 ± 1.92 | 39.65 ± 9.65 | 39.50 ± 7.40 | 37.85 ± 4.10 |
| VNM Q, unsigned_50, mixed_reply, 8 paths | 18.98 ± 3.38 | 38.82 ± 12.61 | 55.66 ± 3.68 | 37.82 ± 4.52 |
| VNM V, signed_90, copies, 8 paths | 35.99 ± 3.22 | 38.38 ± 23.62 | 38.57 ± 5.55 | 37.65 ± 8.16 |
| VNM V, signed_90, mixed_reply, 8 paths | 37.74 ± 3.31 | 35.16 ± 22.33 | 37.79 ± 4.41 | 36.90 ± 7.67 |
| VNM Q, signed_sum, uniform_reply, 8 paths | 36.62 ± 4.24 | 33.11 ± 7.17 | 40.04 ± 2.93 | 36.59 ± 2.94 |
| VNM V, last_plus_sum, copies, 8 paths | 31.35 ± 2.04 | 45.95 ± 12.68 | 32.23 ± 4.27 | 36.51 ± 4.51 |
| VNM V, signed_last, copies, 8 paths | 30.22 ± 2.67 | 46.44 ± 11.45 | 31.35 ± 3.42 | 36.00 ± 4.08 |
| VNM Q, own_90, uniform_reply, 8 paths | 25.03 ± 3.94 | 25.29 ± 8.97 | 57.37 ± 2.64 | 35.90 ± 3.38 |
| VNM Q, signed_last, copies, 8 paths | 32.97 ± 2.24 | 36.52 ± 5.37 | 37.55 ± 6.90 | 35.68 ± 3.01 |
| VNM V, signed_90, uniform_both, 8 paths | 37.51 ± 2.51 | 32.18 ± 13.38 | 36.13 ± 4.83 | 35.27 ± 4.81 |
| VNM Q, unsigned_last, uniform_reply, 8 paths | 27.17 ± 3.21 | 44.19 ± 3.32 | 34.23 ± 4.56 | 35.20 ± 2.16 |
| VNM V, signed_90, uniform_reply, 8 paths | 38.41 ± 2.68 | 28.27 ± 11.67 | 37.11 ± 5.19 | 34.60 ± 4.35 |
| VNM Q, own_90, mixed_reply, 8 paths | 25.87 ± 4.09 | 20.61 ± 11.62 | 57.08 ± 2.74 | 34.52 ± 4.21 |
| VNM V, signed_sum, copies, 8 paths | 35.05 ± 4.33 | 33.01 ± 17.69 | 35.25 ± 5.44 | 34.44 ± 6.33 |
| VNM V, signed_sum, mixed_reply, 8 paths | 36.59 ± 2.95 | 29.49 ± 13.23 | 35.16 ± 5.81 | 33.75 ± 4.92 |
| VNM V, signed_sum, uniform_both, 8 paths | 36.80 ± 2.80 | 30.76 ± 15.82 | 32.81 ± 3.61 | 33.46 ± 5.49 |
| VNM V, unsigned_last, uniform_reply, 8 paths | 24.22 ± 4.90 | 43.31 ± 5.41 | 32.23 ± 4.71 | 33.25 ± 2.90 |
| VNM Q, unsigned_50, copies, 8 paths | 17.95 ± 2.92 | 24.66 ± 15.41 | 55.22 ± 4.13 | 32.61 ± 5.41 |
| VNM V, signed_sum, uniform_reply, 8 paths | 36.52 ± 2.94 | 26.95 ± 7.82 | 34.08 ± 3.97 | 32.52 ± 3.08 |
| VNM Q, own_90, copies, 8 paths | 24.40 ± 3.85 | 17.43 ± 12.21 | 55.42 ± 3.44 | 32.42 ± 4.42 |
| VNM V, unsigned_50, uniform_reply, 8 paths | 16.42 ± 3.14 | 25.39 ± 10.32 | 52.98 ± 4.81 | 31.60 ± 3.94 |
| VNM V, own_90, uniform_reply, 8 paths | 26.33 ± 4.28 | 20.51 ± 6.53 | 44.82 ± 5.25 | 30.55 ± 3.14 |
| VNM V, unsigned_last, mixed_reply, 8 paths | 21.15 ± 2.77 | 39.21 ± 2.20 | 31.05 ± 2.93 | 30.47 ± 1.53 |
| VNM V, unsigned_50, uniform_both, 8 paths | 16.13 ± 3.40 | 23.73 ± 9.11 | 51.27 ± 5.54 | 30.38 ± 3.73 |
| VNM V, unsigned_50, mixed_reply, 8 paths | 16.28 ± 3.07 | 21.83 ± 13.48 | 52.54 ± 3.86 | 30.22 ± 4.79 |
| VNM V, own_90, mixed_reply, 8 paths | 25.73 ± 3.51 | 21.04 ± 10.41 | 43.65 ± 6.82 | 30.14 ± 4.31 |
| VNM Q, unsigned_90, uniform_reply, 8 paths | 17.64 ± 2.82 | 14.26 ± 3.57 | 57.91 ± 3.47 | 29.94 ± 1.91 |
| VNM V, own_90, uniform_both, 8 paths | 26.72 ± 4.91 | 21.83 ± 11.27 | 41.11 ± 6.55 | 29.89 ± 4.64 |
| VNM Q, unsigned_last, mixed_reply, 8 paths | 23.06 ± 2.01 | 34.57 ± 5.31 | 31.40 ± 4.11 | 29.68 ± 2.34 |
| VNM Q, unsigned_last, uniform_both, 8 paths | 22.14 ± 3.44 | 34.42 ± 2.16 | 29.59 ± 2.91 | 28.72 ± 1.67 |
| VNM V, own_90, copies, 8 paths | 24.93 ± 3.76 | 19.43 ± 13.21 | 41.65 ± 5.56 | 28.67 ± 4.94 |
| VNM Q, unsigned_90, uniform_both, 8 paths | 16.79 ± 2.83 | 16.06 ± 7.39 | 52.93 ± 4.62 | 28.59 ± 3.06 |
| VNM Q, unsigned_90, mixed_reply, 8 paths | 16.23 ± 2.49 | 12.35 ± 5.56 | 56.98 ± 2.67 | 28.52 ± 2.22 |
| VNM V, unsigned_50, copies, 8 paths | 15.92 ± 2.77 | 17.24 ± 11.02 | 52.29 ± 4.85 | 28.48 ± 4.12 |
| VNM V, unsigned_90, uniform_reply, 8 paths | 16.81 ± 3.38 | 15.87 ± 2.22 | 52.69 ± 4.18 | 28.46 ± 1.94 |
| VNM Q, unsigned_sum, uniform_reply, 8 paths | 16.68 ± 2.42 | 12.79 ± 3.02 | 55.52 ± 2.26 | 28.33 ± 1.49 |
| VNM Q, unsigned_sum, uniform_both, 8 paths | 16.32 ± 3.47 | 15.53 ± 4.67 | 52.44 ± 4.65 | 28.10 ± 2.48 |
| VNM V, unsigned_sum, uniform_reply, 8 paths | 17.01 ± 3.56 | 14.01 ± 1.75 | 51.37 ± 4.34 | 27.46 ± 1.96 |
| VNM Q, unsigned_sum, mixed_reply, 8 paths | 16.41 ± 2.54 | 12.06 ± 3.84 | 53.03 ± 3.07 | 27.17 ± 1.85 |
| VNM V, unsigned_last, uniform_both, 8 paths | 22.35 ± 3.11 | 31.74 ± 3.91 | 27.29 ± 7.64 | 27.13 ± 3.04 |
| VNM V, unsigned_90, uniform_both, 8 paths | 17.83 ± 3.01 | 14.99 ± 4.95 | 46.97 ± 3.63 | 26.60 ± 2.28 |
| VNM V, unsigned_90, mixed_reply, 8 paths | 15.98 ± 2.74 | 13.72 ± 3.99 | 49.76 ± 4.27 | 26.49 ± 2.15 |
| VNM V, unsigned_sum, mixed_reply, 8 paths | 15.84 ± 2.49 | 14.11 ± 3.66 | 49.17 ± 4.06 | 26.37 ± 2.00 |
| VNM Q, unsigned_90, copies, 8 paths | 14.96 ± 2.43 | 9.13 ± 3.67 | 53.32 ± 5.43 | 25.80 ± 2.33 |
| VNM V, unsigned_90, copies, 8 paths | 15.07 ± 1.56 | 12.55 ± 5.05 | 46.09 ± 2.79 | 24.57 ± 1.99 |
| VNM V, unsigned_sum, uniform_both, 8 paths | 17.95 ± 3.38 | 11.91 ± 4.59 | 42.92 ± 3.83 | 24.26 ± 2.29 |
| VNM Q, unsigned_sum, copies, 8 paths | 14.68 ± 2.14 | 9.08 ± 2.94 | 48.63 ± 5.34 | 24.13 ± 2.15 |
| VNM V, unsigned_last, copies, 8 paths | 18.23 ± 1.52 | 21.92 ± 3.40 | 29.64 ± 2.25 | 23.26 ± 1.45 |
| VNM Q, unsigned_last, copies, 8 paths | 18.16 ± 2.90 | 18.16 ± 2.80 | 29.93 ± 4.11 | 22.08 ± 1.92 |
| VNM V, unsigned_sum, copies, 8 paths | 14.77 ± 1.39 | 11.28 ± 3.73 | 39.94 ± 3.37 | 22.00 ± 1.74 |
