# Tuning win rates

Each rule is named by its network, its continuing players and its whole-game utility. V sees the board after a move. Q sees the board and move. Copies take the largest immediate fitted frequency. Guarded copies first consider immediate wins and threats. A mixture draws an opponent type once per simulated path. Original uses the frozen source on both seats. Signed last, sum and discounted scores use learned visit frequencies. Win, points and quarter-draw give draws weights 0, 0.5 and 0.25. Hybrid adds a bounded learned term to quarter-draw. Fit ties prefer the higher immediate fitted score. Paths gives the sample count for the uniform-reply part in the larger games. Each VNM row compares complete games following the candidate, then chooses the highest expectation. Direct chooses the highest immediate score. [Full definitions and precision plan](DESIGN.md)

Values are win percentages, mean ± sample standard deviation across originals. Each repeat averages two partitions with two fitted initializations each. The combined column weights each game by one third. Its spread describes the average of one independent original per game. Intervals are individual 95% intervals on paired differences, in percentage points. P-values use a two-sided zero-difference model and Holm correction over the win comparisons within the study stage.

The independent original counts in table order are [8, 8, 8].

| Method | Tic-tac-toe | Connect Four | 4×4 Reversi | Equal games | Selected minus row [95% interval] | Adjusted p |
|---|---:|---:|---:|---:|---:|---:|
| VNM V/guarded_mixture/hybrid/paths16 | 53.17 ± 4.07 | 92.29 ± 2.85 | 76.90 ± 3.17 | 74.12 ± 1.96 | reference | — |
| VNM V/guarded_mixture/hybrid/fit_ties/paths16 | 53.17 ± 4.07 | 92.29 ± 2.85 | 76.90 ± 3.17 | 74.12 ± 1.96 | +0.00 [0.00, 0.00] | — |
| VNM V/guarded_mixture/hybrid | 53.17 ± 4.07 | 88.72 ± 3.75 | 75.93 ± 3.46 | 72.61 ± 2.17 | +1.51 [0.81, 2.22] | 0.0141 |
| VNM V/guarded_mixture/hybrid/fit_ties | 53.17 ± 4.07 | 88.72 ± 3.75 | 75.93 ± 3.46 | 72.61 ± 2.17 | +1.51 [0.81, 2.22] | 0.0141 |
| VNM V/guarded_mixture/hybrid/paths1 | 53.17 ± 4.07 | 86.67 ± 4.55 | 72.27 ± 4.05 | 70.70 ± 2.44 | +3.42 [2.30, 4.54] | 0.000956 |
| VNM V/guarded_mixture/hybrid/fit_ties/paths1 | 53.17 ± 4.07 | 86.67 ± 4.55 | 72.27 ± 4.05 | 70.70 ± 2.44 | +3.42 [2.30, 4.54] | 0.000956 |
