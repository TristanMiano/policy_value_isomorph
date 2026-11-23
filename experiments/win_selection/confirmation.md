# Confirmation win rates

Each rule is named by its network, its continuing players and its whole-game utility. V sees the board after a move. Q sees the board and move. Copies take the largest immediate fitted frequency. Guarded copies first consider immediate wins and threats. A mixture draws an opponent type once per simulated path. Original uses the frozen source on both seats. Signed last, sum and discounted scores use learned visit frequencies. Win, points and quarter-draw give draws weights 0, 0.5 and 0.25. Hybrid adds a bounded learned term to quarter-draw. Fit ties prefer the higher immediate fitted score. Paths gives the sample count for the uniform-reply part in the larger games. Each VNM row compares complete games following the candidate, then chooses the highest expectation. Direct chooses the highest immediate score. [Full definitions and precision plan](DESIGN.md)

Values are win percentages, mean ± sample standard deviation across originals. Each repeat averages two partitions with two fitted initializations each. The combined column weights each game by one third. Its spread describes the average of one independent original per game. Intervals are individual 95% intervals on paired differences, in percentage points. P-values use a two-sided zero-difference model and Holm correction over the win comparisons within the study stage.

The independent original counts in table order are [24, 16, 16].

| Method | Tic-tac-toe | Connect Four | 4×4 Reversi | Equal games | Selected minus row [95% interval] | Adjusted p |
|---|---:|---:|---:|---:|---:|---:|
| VNM V/guarded_mixture/hybrid/paths16 | 51.69 ± 4.64 | 92.29 ± 3.22 | 75.83 ± 3.04 | 73.27 ± 2.14 | reference | — |
| VNM V/guarded_mixture/hybrid/fit_ties/paths16 | 51.69 ± 4.64 | 92.29 ± 3.22 | 75.83 ± 3.04 | 73.27 ± 2.14 | +0.00 [0.00, 0.00] | — |
| VNM V/guarded_mixture/points/paths16 | 50.82 ± 4.90 | 91.04 ± 1.99 | 76.37 ± 3.34 | 72.74 ± 2.09 | +0.53 [-0.10, 1.15] | 1 |
| VNM V/guarded_mixture/quarter_draw/paths16 | 51.09 ± 5.08 | 90.94 ± 2.25 | 76.03 ± 3.16 | 72.69 ± 2.13 | +0.58 [-0.05, 1.21] | 1 |
| VNM Q/guarded_mixture/hybrid/paths16 | 52.18 ± 5.84 | 90.84 ± 2.76 | 74.02 ± 2.81 | 72.35 ± 2.35 | +0.92 [-0.12, 1.96] | 1 |
| VNM V/guarded_mixture/hybrid | 51.69 ± 4.64 | 90.01 ± 4.88 | 73.95 ± 3.50 | 71.88 ± 2.53 | +1.38 [0.63, 2.13] | 0.0247 |
| VNM V/guarded_mixture/win/paths16 | 51.36 ± 4.90 | 90.77 ± 2.43 | 73.14 ± 3.92 | 71.76 ± 2.24 | +1.51 [0.74, 2.28] | 0.0102 |
| VNM V/guarded_mixture/points | 50.82 ± 4.90 | 86.87 ± 3.05 | 76.10 ± 2.80 | 71.26 ± 2.14 | +2.01 [1.06, 2.95] | 0.00848 |
| VNM V/guarded_mixture/hybrid/paths1 | 51.69 ± 4.64 | 87.45 ± 5.47 | 72.39 ± 3.98 | 70.51 ± 2.73 | +2.76 [1.87, 3.65] | 4.58e-05 |
| VNM V/mixture/hybrid/paths16 | 51.07 ± 4.81 | 88.94 ± 6.01 | 70.00 ± 2.50 | 70.00 ± 2.70 | +3.27 [2.19, 4.34] | 9.49e-05 |
| VNM Q/guarded_mixture/win | 51.18 ± 5.15 | 84.33 ± 3.12 | 70.39 ± 3.33 | 68.63 ± 2.29 | +4.64 [3.40, 5.87] | 1.23e-07 |
| VNM V/guarded/hybrid/paths16 | 46.85 ± 4.39 | 84.35 ± 7.66 | 62.55 ± 5.75 | 64.58 ± 3.51 | +8.68 [7.08, 10.29] | 7.12e-11 |
| VNM V/mixture/last | 48.84 ± 4.67 | 77.91 ± 13.66 | 46.97 ± 5.51 | 57.91 ± 5.15 | +15.36 [12.94, 17.79] | 3.37e-10 |
| VNM V/guarded_mixture/last/paths16 | 48.23 ± 4.66 | 81.64 ± 8.16 | 43.53 ± 4.96 | 57.80 ± 3.54 | +15.47 [13.85, 17.08] | 5.76e-19 |
| VNM V/original/hybrid/paths16 | 44.62 ± 2.56 | 57.47 ± 28.75 | 68.63 ± 3.82 | 56.91 ± 9.70 | +16.36 [11.36, 21.37] | 0.000148 |
| VNM V/copies/hybrid/paths16 | 47.35 ± 4.47 | 61.52 ± 25.45 | 55.10 ± 5.62 | 54.66 ± 8.81 | +18.61 [14.18, 23.03] | 4.45e-06 |
| Direct Q/copies/direct | 42.80 ± 2.53 | 58.28 ± 10.16 | 49.68 ± 4.01 | 50.25 ± 3.74 | +23.02 [21.17, 24.86] | 4.12e-19 |
| VNM V/guarded_mixture/discounted/paths16 | 46.70 ± 3.21 | 62.28 ± 21.45 | 33.91 ± 4.78 | 47.63 ± 7.40 | +25.64 [21.89, 29.39] | 1.2e-09 |
| VNM V/guarded_mixture/sum/paths16 | 47.25 ± 3.82 | 58.37 ± 17.26 | 33.67 ± 4.42 | 46.43 ± 6.07 | +26.84 [23.84, 29.84] | 5.87e-12 |
| VNM V/copies/last | 44.62 ± 4.85 | 56.45 ± 18.21 | 33.45 ± 6.04 | 44.84 ± 6.60 | +28.43 [25.23, 31.63] | 5.08e-12 |
| VNM Q/copies/last | 41.29 ± 3.02 | 52.00 ± 17.61 | 36.06 ± 4.42 | 43.12 ± 6.13 | +30.15 [27.04, 33.26] | 7.37e-13 |
| Direct V/copies/direct | 39.04 ± 3.39 | 44.21 ± 20.69 | 45.46 ± 5.15 | 42.91 ± 7.20 | +30.36 [26.70, 34.03] | 3.68e-11 |

## Selected VNM rule against external opponents and exact copies

| Game | External wins | Draws | Losses | Same move as original on withheld boards | Mirror wins | Mirror draws | First seat wins in mirror |
|---|---:|---:|---:|---:|---:|---:|---:|
| tictactoe | 51.69 ± 4.64 | 32.75 ± 6.72 | 15.56 ± 6.61 | 73.44 ± 2.13 | 19.79 ± 13.75 | 60.42 ± 27.50 | 38.54 ± 28.53 |
| connect4 | 92.29 ± 3.22 | 0.49 ± 0.60 | 7.23 ± 2.86 | 31.13 ± 7.11 | 48.83 ± 2.23 | 2.34 ± 4.46 | 47.17 ± 12.16 |
| reversi4 | 75.83 ± 3.04 | 1.76 ± 1.10 | 22.41 ± 3.06 | 86.15 ± 1.10 | 49.78 ± 0.28 | 0.44 ± 0.57 | 22.07 ± 4.26 |
| equal_games | 73.27 ± 2.14 | 11.67 ± 2.28 | 15.07 ± 2.61 | 63.57 ± 2.50 | 39.47 ± 4.64 | 21.07 ± 9.29 | 35.93 ± 10.43 |
