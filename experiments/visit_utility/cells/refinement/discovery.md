# Discovery results for visit-trained game scores

The frozen original takes its highest-scoring legal move. A network scoring the resulting board (V) or the board and move (Q) estimates the fraction of observed visits taking a move. Direct selection takes the largest immediate fit. Von Neumann–Morgenstern (VNM) selection maximizes expected complete-game score, replanning on each actual turn. Ties prefer the lowest move index. 'copies' projects highest fitted choices for both seats. The rule 'uniform_reply' projects greedy own moves against uniform legal replies. The rule 'mixed_reply' mixes those two opponents equally once per path. The rule 'uniform_both' projects uniform legal moves for both seats. Each random component gets the listed number of full paths, with exact expectations in tic-tac-toe. The assisted comparison includes an opponent considering tactics and one using positional priorities. The supplied-outcome reference adds actual game results and projects guarded own play against four opponent types.

The utility name specifies the score of a completed game from the current choosing seat. 'signed' counts own moves positively and opposing moves negatively, 'unsigned' counts both positively, and 'own' counts only own moves. 'last' counts the final fitted frequency. The rule 'sum' adds the whole path. The suffix 50, 75, 90 or 97 gives the discount per turn in percent, starting at weight one. 'sqrt_last' and 'square_last' apply the square root or square to the final frequency, signed by mover. 'last_plus_sum' adds the signed last frequency and 0.1 times the signed 0.9-discounted sum. A constant score uses 0.5 in the identical formula. Constant continuations also break all greedy choices by move index. Both constants induce the identical player when all continuing moves are uniformly random.

Win, draw and loss percentages are seat-balanced against four equally weighted external opponent groups. Agreement is the weighted percentage of withheld boards on which the reconstruction chose the original's favorite. Exact-copy self-play uses two identical deployed players and counts both seats. Values give means ± sample standard deviations across originals after averaging their four fits. The equal-game spread describes the average of one independent original per named game. The selected rule is VNM Q, sqrt_last_90, uniform_both, 64 paths.

Independent original counts for tic-tac-toe, Connect Four and 4×4 Reversi are 8, 8, 8.
For paired changes, brackets give individual 95% confidence intervals, followed by the two-sided p-value for a zero mean difference, Holm-adjusted across the table's win and agreement comparisons in all four populations. A dash means the across-original variance is exactly zero. Discovery contrasts are exploratory because the leader was chosen on those data. The compressed estimates and contrasts alongside this table contain all unrounded measurements, raw p-values, original-level differences, and intervals resampling the independent originals.

## External win rate (%)

| Rule | Tic-tac-toe | Connect Four | 4×4 Reversi | Equal games |
|---|---|---|---|---|
| VNM Q, sqrt_last_90, uniform_both, 64 paths | 47.62 ± 0.89 | 79.64 ± 10.48 | 54.59 ± 3.40 | 60.62 ± 3.69 |
| VNM V, sqrt_last_90, uniform_both, 64 paths | 47.43 ± 0.86 | 81.84 ± 10.24 | 51.61 ± 3.32 | 60.29 ± 3.60 |
| VNM V, sqrt_last_90, uniform_both, 32 paths | 47.43 ± 0.86 | 80.08 ± 8.23 | 49.22 ± 4.34 | 58.91 ± 3.11 |
| VNM Q, sqrt_last_90, uniform_both, 32 paths | 47.62 ± 0.89 | 77.59 ± 11.65 | 50.49 ± 3.44 | 58.57 ± 4.06 |
| VNM Q, signed_last_90, uniform_both, 64 paths | 46.70 ± 1.02 | 76.37 ± 11.51 | 50.93 ± 3.94 | 58.00 ± 4.07 |
| VNM V, signed_last_75, uniform_both, 64 paths | 47.49 ± 1.17 | 79.54 ± 9.88 | 46.63 ± 2.66 | 57.89 ± 3.43 |
| VNM Q, signed_last_75, uniform_both, 64 paths | 47.31 ± 1.03 | 76.12 ± 11.61 | 49.71 ± 4.40 | 57.71 ± 4.15 |
| VNM V, signed_last_90, uniform_both, 64 paths | 45.88 ± 1.30 | 78.37 ± 10.53 | 47.66 ± 3.44 | 57.30 ± 3.72 |
| VNM V, signed_last_75, uniform_both, 32 paths | 47.49 ± 1.17 | 78.91 ± 10.28 | 44.58 ± 4.17 | 56.99 ± 3.72 |
| VNM Q, signed_last_90, uniform_both, 32 paths | 46.70 ± 1.02 | 75.20 ± 10.64 | 49.02 ± 4.32 | 56.97 ± 3.84 |
| VNM Q, signed_last_97, uniform_both, 64 paths | 45.51 ± 0.77 | 73.44 ± 11.75 | 51.76 ± 4.24 | 56.90 ± 4.17 |
| VNM Q, signed_last_75, uniform_both, 32 paths | 47.31 ± 1.03 | 75.34 ± 12.48 | 47.71 ± 3.60 | 56.79 ± 4.34 |
| VNM V, signed_last_97, uniform_both, 64 paths | 43.30 ± 1.28 | 78.22 ± 9.16 | 47.71 ± 3.36 | 56.41 ± 3.28 |
| VNM V, signed_last_90, uniform_both, 32 paths | 45.88 ± 1.30 | 77.00 ± 9.56 | 45.61 ± 4.68 | 56.16 ± 3.57 |
| VNM Q, signed_last_97, uniform_both, 32 paths | 45.51 ± 0.77 | 72.71 ± 9.68 | 49.41 ± 3.81 | 55.88 ± 3.48 |
| VNM Q, sqrt_last_90, mixed_reply, 64 paths | 44.20 ± 2.27 | 75.39 ± 8.28 | 45.36 ± 4.84 | 54.99 ± 3.28 |
| VNM Q, sqrt_last_90, uniform_both, 8 paths | 47.62 ± 0.89 | 69.92 ± 11.59 | 47.07 ± 5.90 | 54.87 ± 4.34 |
| VNM V, signed_last_97, uniform_both, 32 paths | 43.30 ± 1.28 | 74.37 ± 12.26 | 46.83 ± 2.56 | 54.83 ± 4.20 |
| VNM V, sqrt_last_90, uniform_both, 8 paths | 47.43 ± 0.86 | 71.34 ± 11.17 | 45.36 ± 5.23 | 54.71 ± 4.12 |
| VNM Q, sqrt_last_90, mixed_reply, 32 paths | 44.20 ± 2.27 | 74.51 ± 9.62 | 44.14 ± 4.08 | 54.29 ± 3.56 |
| VNM Q, signed_last_75, uniform_both, 8 paths | 47.31 ± 1.03 | 70.90 ± 10.55 | 44.63 ± 4.59 | 54.28 ± 3.85 |
| VNM V, signed_last_75, uniform_both, 8 paths | 47.49 ± 1.17 | 70.31 ± 9.93 | 45.02 ± 5.62 | 54.27 ± 3.82 |
| VNM V, sqrt_last_90, mixed_reply, 64 paths | 43.99 ± 0.93 | 72.95 ± 12.22 | 45.75 ± 9.17 | 54.23 ± 5.10 |
| VNM V, sqrt_last_90, mixed_reply, 32 paths | 43.99 ± 0.93 | 73.34 ± 12.50 | 44.48 ± 8.68 | 53.94 ± 5.08 |
| VNM Q, square_last_90, uniform_both, 64 paths | 44.44 ± 0.88 | 70.26 ± 9.32 | 45.90 ± 4.98 | 53.53 ± 3.53 |
| VNM Q, square_last_90, uniform_both, 32 paths | 44.44 ± 0.88 | 69.58 ± 9.47 | 45.80 ± 5.26 | 53.27 ± 3.62 |
| VNM V, signed_last_75, mixed_reply, 64 paths | 44.11 ± 1.25 | 74.41 ± 13.77 | 41.11 ± 6.33 | 53.21 ± 5.07 |
| VNM V, sqrt_last_90, mixed_reply, 8 paths | 43.99 ± 0.93 | 72.07 ± 11.36 | 43.46 ± 6.34 | 53.17 ± 4.35 |
| VNM V, signed_last_75, mixed_reply, 32 paths | 44.11 ± 1.25 | 73.10 ± 12.50 | 42.14 ± 6.92 | 53.12 ± 4.78 |
| VNM Q, signed_last_90, uniform_both, 8 paths | 46.70 ± 1.02 | 66.06 ± 10.74 | 46.24 ± 5.16 | 53.00 ± 3.99 |
| VNM V, signed_last_90, uniform_both, 8 paths | 45.88 ± 1.30 | 69.24 ± 11.33 | 43.80 ± 5.19 | 52.97 ± 4.18 |
| VNM Q, signed_last_90, mixed_reply, 64 paths | 43.46 ± 2.20 | 70.31 ± 9.75 | 43.95 ± 3.74 | 52.57 ± 3.56 |
| VNM Q, sqrt_last_90, mixed_reply, 8 paths | 44.20 ± 2.27 | 70.07 ± 9.57 | 43.21 ± 4.27 | 52.49 ± 3.57 |
| VNM V, signed_last_75, mixed_reply, 8 paths | 44.11 ± 1.25 | 72.85 ± 12.83 | 40.43 ± 7.59 | 52.46 ± 4.99 |
| VNM V, signed_last_90, mixed_reply, 64 paths | 43.07 ± 1.73 | 71.14 ± 12.91 | 42.68 ± 8.41 | 52.30 ± 5.17 |
| VNM V, signed_last_90, mixed_reply, 32 paths | 43.07 ± 1.73 | 70.07 ± 12.23 | 43.12 ± 7.52 | 52.08 ± 4.82 |
| VNM Q, signed_last_75, mixed_reply, 64 paths | 43.59 ± 3.12 | 69.24 ± 9.12 | 42.82 ± 3.07 | 51.88 ± 3.37 |
| VNM V, square_last_90, uniform_both, 64 paths | 42.84 ± 3.00 | 72.12 ± 9.84 | 40.14 ± 4.18 | 51.70 ± 3.70 |
| VNM Q, signed_last_90, mixed_reply, 32 paths | 43.46 ± 2.20 | 67.92 ± 10.35 | 42.48 ± 4.48 | 51.29 ± 3.83 |
| VNM V, signed_last_90, mixed_reply, 8 paths | 43.07 ± 1.73 | 69.04 ± 12.68 | 41.55 ± 7.74 | 51.22 ± 4.99 |
| VNM V, signed_last_97, uniform_both, 8 paths | 43.30 ± 1.28 | 66.89 ± 10.18 | 43.07 ± 4.42 | 51.09 ± 3.72 |
| VNM V, square_last_90, uniform_both, 32 paths | 42.84 ± 3.00 | 68.99 ± 11.51 | 40.87 ± 4.19 | 50.90 ± 4.20 |
| VNM V, signed_last_97, mixed_reply, 32 paths | 42.15 ± 2.56 | 67.77 ± 11.93 | 42.53 ± 8.09 | 50.82 ± 4.88 |
| VNM Q, signed_last_75, mixed_reply, 32 paths | 43.59 ± 3.12 | 68.55 ± 9.42 | 40.09 ± 3.62 | 50.74 ± 3.52 |
| VNM V, signed_last_97, mixed_reply, 64 paths | 42.15 ± 2.56 | 67.14 ± 11.14 | 42.68 ± 8.28 | 50.66 ± 4.70 |
| VNM Q, signed_last_97, uniform_both, 8 paths | 45.51 ± 0.77 | 61.28 ± 10.74 | 45.17 ± 4.41 | 50.65 ± 3.88 |
| VNM Q, signed_last_90, mixed_reply, 8 paths | 43.46 ± 2.20 | 65.77 ± 8.35 | 42.48 ± 3.90 | 50.57 ± 3.16 |
| VNM Q, signed_last_97, mixed_reply, 64 paths | 42.43 ± 2.34 | 64.79 ± 7.68 | 44.38 ± 3.83 | 50.54 ± 2.96 |
| VNM Q, signed_last_75, mixed_reply, 8 paths | 43.59 ± 3.12 | 66.16 ± 10.38 | 40.82 ± 4.18 | 50.19 ± 3.87 |
| VNM V, square_last_90, mixed_reply, 32 paths | 41.34 ± 2.24 | 67.48 ± 14.56 | 39.65 ± 6.18 | 49.49 ± 5.32 |
| VNM Q, signed_last_97, mixed_reply, 32 paths | 42.43 ± 2.34 | 62.99 ± 8.24 | 42.82 ± 3.91 | 49.41 ± 3.14 |
| VNM Q, square_last_90, uniform_both, 8 paths | 44.44 ± 0.88 | 59.03 ± 9.42 | 42.38 ± 5.04 | 48.62 ± 3.57 |
| VNM V, signed_last_97, mixed_reply, 8 paths | 42.15 ± 2.56 | 61.67 ± 12.06 | 41.85 ± 7.75 | 48.56 ± 4.86 |
| VNM V, square_last_90, mixed_reply, 64 paths | 41.34 ± 2.24 | 66.26 ± 15.11 | 37.89 ± 7.28 | 48.50 ± 5.64 |
| VNM V, square_last_90, uniform_both, 8 paths | 42.84 ± 3.00 | 62.79 ± 12.26 | 39.60 ± 5.32 | 48.41 ± 4.57 |
| VNM Q, signed_last_97, mixed_reply, 8 paths | 42.43 ± 2.34 | 58.59 ± 9.09 | 42.92 ± 3.96 | 47.98 ± 3.39 |
| VNM Q, square_last_90, mixed_reply, 64 paths | 40.90 ± 2.84 | 62.45 ± 9.14 | 40.33 ± 2.86 | 47.89 ± 3.33 |
| VNM Q, square_last_90, mixed_reply, 32 paths | 40.90 ± 2.84 | 63.09 ± 8.87 | 39.55 ± 3.44 | 47.85 ± 3.31 |
| VNM V, square_last_90, mixed_reply, 8 paths | 41.34 ± 2.24 | 62.65 ± 13.72 | 38.33 ± 9.10 | 47.44 ± 5.54 |
| VNM Q, square_last_90, mixed_reply, 8 paths | 40.90 ± 2.84 | 59.57 ± 6.85 | 40.23 ± 3.79 | 46.90 ± 2.78 |
