# Discovery win rates

Each rule is named by its network, its continuing players and its whole-game utility. V sees the board after a move. Q sees the board and move. Copies take the largest immediate fitted frequency. Guarded copies first consider immediate wins and threats. A mixture draws an opponent type once per simulated path. Original uses the frozen source on both seats. Signed last, sum and discounted scores use learned visit frequencies. Win, points and quarter-draw give draws weights 0, 0.5 and 0.25. Hybrid adds a bounded learned term to quarter-draw. Fit ties prefer the higher immediate fitted score. Paths gives the sample count for the uniform-reply part in the larger games. Each VNM row compares complete games following the candidate, then chooses the highest expectation. Direct chooses the highest immediate score. [Full definitions and precision plan](DESIGN.md)

Values are win percentages, mean ± sample standard deviation across originals. Each repeat averages two partitions with two fitted initializations each. The combined column weights each game by one third. Its spread describes the average of one independent original per game. Intervals are individual 95% intervals on paired differences, in percentage points. P-values use a two-sided zero-difference model and Holm correction over the win comparisons within the study stage.

The independent original counts in table order are [8, 8, 8].

| Method | Tic-tac-toe | Connect Four | 4×4 Reversi | Equal games | Selected minus row [95% interval] | Adjusted p |
|---|---:|---:|---:|---:|---:|---:|
| VNM V/guarded_mixture/hybrid | 53.17 ± 4.07 | 88.72 ± 3.75 | 75.93 ± 3.46 | 72.61 ± 2.17 | reference | — |
| VNM V/guarded_mixture/points | 50.62 ± 5.40 | 87.30 ± 3.98 | 76.86 ± 2.42 | 71.59 ± 2.38 | +1.01 [-0.49, 2.51] | 1 |
| VNM V/guarded_mixture/quarter_draw | 50.87 ± 4.88 | 87.26 ± 3.79 | 76.17 ± 3.21 | 71.43 ± 2.32 | +1.17 [-0.24, 2.58] | 1 |
| VNM Q/guarded_mixture/quarter_draw | 53.03 ± 4.64 | 85.30 ± 3.58 | 74.02 ± 2.84 | 70.78 ± 2.17 | +1.82 [0.64, 3.00] | 0.363 |
| VNM Q/guarded_mixture/points | 51.18 ± 5.35 | 85.74 ± 2.82 | 74.46 ± 2.82 | 70.46 ± 2.22 | +2.14 [1.14, 3.15] | 0.0297 |
| VNM V/guarded_mixture/win | 51.03 ± 4.76 | 86.62 ± 3.53 | 72.61 ± 3.64 | 70.09 ± 2.32 | +2.52 [0.97, 4.07] | 0.298 |
| VNM Q/mixture/quarter_draw | 51.87 ± 2.94 | 86.28 ± 3.22 | 71.29 ± 3.39 | 69.81 ± 1.84 | +2.79 [0.61, 4.97] | 0.857 |
| VNM Q/mixture/win | 52.69 ± 4.35 | 85.60 ± 2.66 | 71.00 ± 3.01 | 69.76 ± 1.97 | +2.85 [0.78, 4.92] | 0.628 |
| VNM Q/guarded_mixture/hybrid | 51.78 ± 3.39 | 84.72 ± 4.76 | 72.61 ± 3.31 | 69.70 ± 2.24 | +2.91 [1.07, 4.74] | 0.416 |
| VNM Q/guarded_mixture/win | 53.29 ± 5.21 | 84.57 ± 2.73 | 71.19 ± 3.19 | 69.68 ± 2.23 | +2.92 [1.60, 4.25] | 0.0267 |
| VNM Q/mixture/points | 51.31 ± 3.83 | 86.04 ± 3.45 | 71.34 ± 2.85 | 69.56 ± 1.96 | +3.05 [0.95, 5.15] | 0.568 |
| VNM V/mixture/quarter_draw | 50.24 ± 5.16 | 84.77 ± 6.74 | 71.53 ± 1.99 | 68.85 ± 2.90 | +3.76 [1.53, 5.99] | 0.261 |
| VNM Q/mixture/hybrid | 51.31 ± 2.72 | 85.21 ± 2.67 | 69.97 ± 2.49 | 68.83 ± 1.52 | +3.78 [1.91, 5.65] | 0.0687 |
| VNM V/mixture/hybrid | 50.61 ± 5.02 | 85.50 ± 8.51 | 70.12 ± 2.00 | 68.74 ± 3.36 | +3.86 [1.56, 6.17] | 0.31 |
| VNM V/mixture/points | 49.77 ± 5.67 | 84.67 ± 6.77 | 70.90 ± 2.25 | 68.44 ± 3.04 | +4.16 [1.93, 6.39] | 0.151 |
| VNM V/mixture/win | 50.90 ± 4.89 | 83.74 ± 6.86 | 70.65 ± 2.69 | 68.43 ± 2.95 | +4.18 [1.90, 6.45] | 0.169 |
| VNM V/guarded/hybrid | 47.69 ± 5.73 | 82.67 ± 7.62 | 65.38 ± 4.40 | 65.25 ± 3.50 | +7.36 [5.18, 9.54] | 0.000333 |
| VNM Q/guarded/hybrid | 47.67 ± 4.24 | 75.15 ± 10.08 | 63.38 ± 3.45 | 62.07 ± 3.82 | +10.54 [7.37, 13.71] | 0.00328 |
| VNM V/guarded/points | 48.29 ± 4.93 | 67.38 ± 5.55 | 70.41 ± 4.85 | 62.03 ± 2.96 | +10.58 [8.08, 13.09] | 1.03e-05 |
| VNM V/guarded/quarter_draw | 48.29 ± 4.93 | 67.38 ± 5.55 | 70.41 ± 4.85 | 62.03 ± 2.96 | +10.58 [8.08, 13.09] | 1.03e-05 |
| VNM Q/guarded/points | 45.92 ± 5.05 | 66.36 ± 6.83 | 72.56 ± 1.84 | 61.61 ± 2.90 | +11.00 [8.84, 13.15] | 1.48e-06 |
| VNM Q/guarded/quarter_draw | 45.92 ± 5.05 | 66.36 ± 6.83 | 72.56 ± 1.84 | 61.61 ± 2.90 | +11.00 [8.84, 13.15] | 1.48e-06 |
| VNM Q/guarded/win | 43.46 ± 5.11 | 62.74 ± 7.72 | 72.36 ± 1.76 | 59.52 ± 3.14 | +13.09 [10.55, 15.62] | 4.95e-06 |
| VNM V/mixture/last | 50.85 ± 3.34 | 78.17 ± 14.61 | 49.41 ± 5.08 | 59.48 ± 5.28 | +13.13 [9.13, 17.13] | 0.00501 |
| VNM V/guarded/win | 45.01 ± 5.82 | 61.82 ± 3.57 | 70.36 ± 4.47 | 59.06 ± 2.72 | +13.54 [11.28, 15.81] | 1.53e-08 |
| VNM V/guarded_mixture/last | 50.18 ± 3.03 | 80.13 ± 9.95 | 43.41 ± 6.16 | 57.91 ± 4.03 | +14.70 [11.68, 17.72] | 2.32e-05 |
| VNM Q/mixture/last | 46.50 ± 2.45 | 77.88 ± 10.24 | 48.14 ± 6.39 | 57.51 ± 4.11 | +15.10 [11.66, 18.54] | 7.87e-06 |
| VNM Q/guarded_mixture/last | 46.99 ± 2.44 | 76.90 ± 9.60 | 44.38 ± 3.00 | 56.09 ± 3.45 | +16.51 [13.68, 19.35] | 1.81e-06 |
| VNM Q/copies/hybrid | 46.29 ± 5.14 | 61.18 ± 16.85 | 59.81 ± 3.35 | 55.76 ± 5.98 | +16.85 [12.03, 21.66] | 0.0028 |
| VNM V/copies/points | 47.56 ± 3.87 | 59.28 ± 6.94 | 59.96 ± 3.17 | 55.60 ± 2.85 | +17.01 [15.20, 18.82] | 7.5e-12 |
| VNM V/copies/quarter_draw | 47.56 ± 3.87 | 59.28 ± 6.94 | 59.96 ± 3.17 | 55.60 ± 2.85 | +17.01 [15.20, 18.82] | 7.5e-12 |
| VNM V/copies/hybrid | 47.77 ± 5.04 | 61.13 ± 28.14 | 55.32 ± 5.51 | 54.74 ± 9.71 | +17.87 [10.24, 25.50] | 0.0798 |
| VNM V/copies/win | 45.17 ± 3.44 | 59.23 ± 6.95 | 59.57 ± 2.85 | 54.65 ± 2.75 | +17.95 [16.24, 19.67] | 4.6e-12 |
| VNM Q/copies/points | 43.03 ± 3.60 | 57.23 ± 13.47 | 60.55 ± 1.42 | 53.60 ± 4.67 | +19.00 [15.81, 22.20] | 3.56e-05 |
| VNM Q/copies/quarter_draw | 43.03 ± 3.60 | 57.23 ± 13.47 | 60.55 ± 1.42 | 53.60 ± 4.67 | +19.00 [15.81, 22.20] | 3.56e-05 |
| Direct Q/copies/direct | 45.80 ± 4.02 | 60.94 ± 11.50 | 52.25 ± 4.26 | 52.99 ± 4.30 | +19.61 [16.13, 23.10] | 3.32e-06 |
| VNM Q/copies/win | 41.99 ± 3.19 | 56.69 ± 12.79 | 59.18 ± 2.17 | 52.62 ± 4.45 | +19.99 [17.01, 22.97] | 1.61e-05 |
| VNM Q/mixture/discounted | 47.75 ± 4.14 | 65.04 ± 14.72 | 42.87 ± 7.27 | 51.89 ± 5.64 | +20.72 [16.59, 24.85] | 4.39e-05 |
| VNM Q/copies/discounted | 46.76 ± 5.48 | 64.26 ± 17.23 | 44.38 ± 5.34 | 51.80 ± 6.28 | +20.81 [15.82, 25.79] | 0.000423 |
| VNM Q/guarded_mixture/discounted | 47.70 ± 3.82 | 64.99 ± 12.84 | 36.87 ± 4.72 | 49.85 ± 4.73 | +22.76 [19.18, 26.33] | 2.26e-05 |
| VNM V/guarded/last | 45.07 ± 6.87 | 68.85 ± 8.43 | 35.60 ± 6.65 | 49.84 ± 4.25 | +22.77 [19.99, 25.56] | 3.7e-10 |
| VNM Q/guarded/last | 38.74 ± 5.27 | 68.36 ± 9.50 | 40.33 ± 4.49 | 49.14 ± 3.92 | +23.46 [20.85, 26.08] | 6.84e-11 |
| VNM Q/mixture/sum | 48.56 ± 4.85 | 61.04 ± 17.57 | 36.38 ± 4.21 | 48.66 ± 6.23 | +23.95 [18.87, 29.03] | 0.000289 |
| VNM V/guarded_mixture/discounted | 49.73 ± 5.48 | 62.21 ± 21.08 | 33.74 ± 3.94 | 48.56 ± 7.38 | +24.05 [18.31, 29.79] | 0.0013 |
| VNM Q/guarded_mixture/sum | 47.91 ± 5.30 | 59.77 ± 9.38 | 36.28 ± 3.43 | 47.98 ± 3.77 | +24.62 [21.65, 27.60] | 8.23e-09 |
| VNM V/guarded_mixture/sum | 51.07 ± 5.21 | 55.76 ± 19.94 | 36.67 ± 2.64 | 47.83 ± 6.92 | +24.77 [19.56, 29.99] | 0.000546 |
| VNM V/mixture/sum | 51.42 ± 5.11 | 57.57 ± 24.97 | 34.33 ± 4.84 | 47.77 ± 8.65 | +24.84 [18.09, 31.58] | 0.00488 |
| VNM Q/guarded/discounted | 46.73 ± 4.58 | 61.77 ± 13.47 | 33.84 ± 4.77 | 47.45 ± 5.00 | +25.16 [21.47, 28.85] | 7.2e-06 |
| VNM V/mixture/discounted | 49.70 ± 4.71 | 57.13 ± 30.69 | 34.28 ± 3.77 | 47.03 ± 10.42 | +25.57 [17.23, 33.92] | 0.0182 |
| VNM V/guarded/discounted | 47.21 ± 4.90 | 61.28 ± 25.30 | 32.08 ± 2.67 | 46.86 ± 8.64 | +25.75 [19.05, 32.46] | 0.0041 |
| VNM V/copies/discounted | 47.47 ± 5.04 | 55.86 ± 34.81 | 36.62 ± 4.34 | 46.65 ± 11.81 | +25.96 [16.47, 35.44] | 0.0368 |
| VNM V/guarded/sum | 47.44 ± 5.73 | 57.37 ± 21.92 | 32.91 ± 3.34 | 45.91 ± 7.63 | +26.70 [20.77, 32.63] | 0.00121 |
| Direct V/copies/direct | 40.22 ± 3.42 | 47.46 ± 20.43 | 48.44 ± 5.74 | 45.37 ± 7.17 | +27.24 [21.71, 32.76] | 0.000162 |
| VNM V/copies/last | 45.38 ± 6.46 | 56.05 ± 18.22 | 34.47 ± 2.98 | 45.30 ± 6.52 | +27.30 [22.23, 32.38] | 0.000125 |
| VNM Q/copies/sum | 45.62 ± 5.30 | 50.10 ± 19.60 | 40.19 ± 4.47 | 45.30 ± 6.93 | +27.31 [21.78, 32.83] | 0.000123 |
| VNM Q/guarded/sum | 45.30 ± 4.75 | 55.76 ± 17.65 | 33.54 ± 3.97 | 44.87 ± 6.23 | +27.74 [22.85, 32.63] | 6.92e-05 |
| VNM V/copies/sum | 47.01 ± 4.71 | 48.19 ± 28.36 | 34.81 ± 6.29 | 43.34 ± 9.81 | +29.27 [21.53, 37.01] | 0.0041 |
| VNM Q/copies/last | 40.72 ± 3.67 | 47.17 ± 16.89 | 38.33 ± 3.98 | 42.07 ± 5.91 | +30.53 [25.82, 35.25] | 4.34e-05 |
