# Changing selection on the fitting positions

The original is the frozen neural player's favorite legal move. Q scores a board and move. V scores only the board resulting from the move. Direct chooses the largest immediate score. Von Neumann–Morgenstern (VNM) selection compares the score of complete games from each candidate, with both seats continuing by fitted direct play. Terminal VNM counts only the final move’s frequency, signed for its mover. Centered VNM sums the played score minus the best available score at each turn, signed for the mover. Each position evaluated here contributed its actual original move to training. All original visits in the collection were pooled across the old split labels. These rows measure in-sample behavior.

For Q, original visits fit the observed board-and-move frequencies. In the all-paths fit we also observe both copies of the original playing out every initial alternative before freezing the fit. A forced first alternative contributes zero source observations. On every following freely chosen move its selected candidate gets 1 and the others get 0. Tic-tac-toe already includes every decision. For V, multiple incoming candidates can share a successor. V pooled assigns each successor its mean training target. V disjoint keeps the first board in fixed order whose successor inputs are all unused, using only legal transitions to choose the set. This directly fits the visit fractions on every selected board. Unobserved inputs use independent random background scores. A covered board has every legal alternative and all subsequent direct decisions scored entirely with fitted inputs.

Values are means ± sample standard deviations across independently frozen original-player blocks. Average the two collections and two initializations first. The fixed tic-tac-toe and Reversi choices can coincide between repetitions. Differences are VNM minus direct on the same frozen fit, boards and paired games. Rates are percentages and their differences are percentage points. The 95% intervals and two-sided p-values refer to the zero-mean paired-difference model. Holm adjustment covers this training-control family. A dash for a p-value denotes a zero standard error. Games use the Phase V seat-balanced opponent panel. [Full diagnosis](../../AGREEMENT_DIAGNOSIS.md)

| Game | Originals | Training population | Fitted roots evaluated | All alternatives covered (%) | Direct agreement (%) | Terminal VNM agreement (%) | Change ± spread [95% interval] | Raw / adjusted p | Centered VNM agreement (%) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Tic-tac-toe | 32 | Q, all alternative paths fitted | 4520.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 64.77 ± 5.56 | -35.23 ± 5.56 [-37.23, -33.22] | 8.45e-27 / 2.46e-24 | 100.00 ± 0.00 |
| Tic-tac-toe | 32 | V, all observed parents | 4520.00 ± 0.00 | 100.00 ± 0.00 | 95.29 ± 0.80 | 61.03 ± 4.07 | -34.26 ± 3.54 [-35.53, -32.98] | 2.07e-32 / 6.42e-30 | 95.29 ± 0.80 |
| Tic-tac-toe | 32 | V, disjoint successors | 929.00 ± 0.00 | 45.60 ± 0.19 | 100.00 ± 0.00 | 64.83 ± 3.35 | -35.17 ± 3.35 [-36.37, -33.96] | 1.71e-33 / 5.47e-31 | 100.00 ± 0.00 |
| Connect Four | 24 | Q, original visits | 4237.08 ± 792.86 | 1.22 ± 0.63 | 100.00 ± 0.00 | 51.74 ± 0.76 | -48.26 ± 0.76 [-48.58, -47.94] | 3.15e-43 / 1.06e-40 | 100.00 ± 0.00 |
| Connect Four | 24 | Q, all alternative paths fitted | 4237.08 ± 792.86 | 100.00 ± 0.00 | 100.00 ± 0.00 | 23.52 ± 9.34 | -76.48 ± 9.34 [-80.43, -72.54] | 8.51e-23 / 2.39e-20 | 100.00 ± 0.00 |
| Connect Four | 24 | V, all observed parents | 4237.08 ± 792.86 | 1.22 ± 0.63 | 99.98 ± 0.02 | 50.00 ± 1.18 | -49.98 ± 1.17 [-50.48, -49.49] | 3.35e-39 / 1.1e-36 | 99.98 ± 0.02 |
| Connect Four | 24 | V, disjoint successors | 3985.58 ± 775.31 | 1.27 ± 0.66 | 100.00 ± 0.00 | 49.20 ± 0.87 | -50.80 ± 0.87 [-51.17, -50.43] | 2.66e-42 / 8.91e-40 | 100.00 ± 0.00 |
| 4×4 Reversi | 24 | Q, original visits | 1186.81 ± 26.07 | 51.53 ± 0.84 | 100.00 ± 0.00 | 79.50 ± 0.77 | -20.50 ± 0.77 [-20.82, -20.17] | 1.81e-34 / 5.84e-32 | 100.00 ± 0.00 |
| 4×4 Reversi | 24 | Q, all alternative paths fitted | 1186.81 ± 26.07 | 100.00 ± 0.00 | 100.00 ± 0.00 | 87.65 ± 0.66 | -12.35 ± 0.66 [-12.63, -12.07] | 5.97e-31 / 1.85e-28 | 100.00 ± 0.00 |
| 4×4 Reversi | 24 | V, all observed parents | 1186.81 ± 26.07 | 52.29 ± 0.81 | 100.00 ± 0.01 | 80.65 ± 0.81 | -19.35 ± 0.81 [-19.69, -19.01] | 2.05e-33 / 6.53e-31 | 100.00 ± 0.01 |
| 4×4 Reversi | 24 | V, disjoint successors | 1113.48 ± 24.89 | 50.01 ± 1.02 | 100.00 ± 0.00 | 80.69 ± 0.86 | -19.31 ± 0.86 [-19.67, -18.95] | 7.43e-33 / 2.34e-30 | 100.00 ± 0.00 |

## Which whole-game utility preserves the initial fit?

Sum counts every signed move frequency. Discounted sum gives successive moves weights 1, 0.9, 0.81 and so on. All columns use the same frozen fit and continuations.

| Game | Originals | Training population | Direct agreement (%) | Final-move VNM (%) | All moves VNM (%) | Discounted VNM (%) | Centered VNM (%) |
|---|---:|---|---:|---:|---:|---:|---:|
| Tic-tac-toe | 32 | Q, all alternative paths fitted | 100.00 ± 0.00 | 64.77 ± 5.56 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Tic-tac-toe | 32 | V, all observed parents | 95.29 ± 0.80 | 61.03 ± 4.07 | 83.99 ± 0.95 | 89.90 ± 0.54 | 95.29 ± 0.80 |
| Tic-tac-toe | 32 | V, disjoint successors | 100.00 ± 0.00 | 64.83 ± 3.35 | 91.76 ± 0.95 | 96.24 ± 0.43 | 100.00 ± 0.00 |
| Connect Four | 24 | Q, original visits | 100.00 ± 0.00 | 51.74 ± 0.76 | 75.49 ± 2.29 | 99.92 ± 0.04 | 100.00 ± 0.00 |
| Connect Four | 24 | Q, all alternative paths fitted | 100.00 ± 0.00 | 23.52 ± 9.34 | 77.83 ± 6.86 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Connect Four | 24 | V, all observed parents | 99.98 ± 0.02 | 50.00 ± 1.18 | 75.44 ± 2.08 | 98.39 ± 0.64 | 99.98 ± 0.02 |
| Connect Four | 24 | V, disjoint successors | 100.00 ± 0.00 | 49.20 ± 0.87 | 78.01 ± 2.16 | 99.85 ± 0.06 | 100.00 ± 0.00 |
| 4×4 Reversi | 24 | Q, original visits | 100.00 ± 0.00 | 79.50 ± 0.77 | 94.80 ± 0.39 | 99.79 ± 0.10 | 100.00 ± 0.00 |
| 4×4 Reversi | 24 | Q, all alternative paths fitted | 100.00 ± 0.00 | 87.65 ± 0.66 | 99.61 ± 0.13 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| 4×4 Reversi | 24 | V, all observed parents | 100.00 ± 0.01 | 80.65 ± 0.81 | 93.70 ± 0.45 | 98.71 ± 0.23 | 100.00 ± 0.01 |
| 4×4 Reversi | 24 | V, disjoint successors | 100.00 ± 0.00 | 80.69 ± 0.86 | 94.24 ± 0.48 | 99.04 ± 0.25 | 100.00 ± 0.00 |

## Games played from the panel openings

| Game | Originals | Fit | Rule | Wins (%) | Draws (%) | Losses (%) | Points/game |
|---|---:|---|---|---:|---:|---:|---:|
| Tic-tac-toe | 32 | Q, all alternative paths fitted | Direct | 45.55 ± 6.01 | 54.45 ± 6.01 | 0.00 ± 0.00 | 0.46 ± 0.06 |
| Tic-tac-toe | 32 | Q, all alternative paths fitted | Terminal VNM | 38.90 ± 2.68 | 27.82 ± 2.00 | 33.28 ± 2.27 | 0.06 ± 0.05 |
| Tic-tac-toe | 32 | Q, all alternative paths fitted | All signed moves VNM | 45.55 ± 6.01 | 54.45 ± 6.01 | 0.00 ± 0.00 | 0.46 ± 0.06 |
| Tic-tac-toe | 32 | Q, all alternative paths fitted | Discounted signed moves VNM | 45.55 ± 6.01 | 54.45 ± 6.01 | 0.00 ± 0.00 | 0.46 ± 0.06 |
| Tic-tac-toe | 32 | Q, all alternative paths fitted | Centered VNM | 45.55 ± 6.01 | 54.45 ± 6.01 | 0.00 ± 0.00 | 0.46 ± 0.06 |
| Tic-tac-toe | 32 | V, all observed parents | Direct | 44.74 ± 5.87 | 52.45 ± 7.21 | 2.81 ± 6.57 | 0.42 ± 0.10 |
| Tic-tac-toe | 32 | V, all observed parents | Terminal VNM | 38.61 ± 3.73 | 30.05 ± 3.17 | 31.34 ± 3.10 | 0.07 ± 0.06 |
| Tic-tac-toe | 32 | V, all observed parents | All signed moves VNM | 44.16 ± 2.62 | 52.00 ± 5.38 | 3.84 ± 5.44 | 0.40 ± 0.07 |
| Tic-tac-toe | 32 | V, all observed parents | Discounted signed moves VNM | 44.49 ± 2.73 | 52.92 ± 5.23 | 2.59 ± 5.02 | 0.42 ± 0.06 |
| Tic-tac-toe | 32 | V, all observed parents | Centered VNM | 44.74 ± 5.87 | 52.45 ± 7.21 | 2.81 ± 6.57 | 0.42 ± 0.10 |
| Tic-tac-toe | 32 | V, disjoint successors | Direct | 32.70 ± 4.08 | 29.93 ± 12.09 | 37.36 ± 13.85 | -0.05 ± 0.16 |
| Tic-tac-toe | 32 | V, disjoint successors | Terminal VNM | 37.22 ± 2.69 | 16.89 ± 7.63 | 45.89 ± 7.83 | -0.09 ± 0.09 |
| Tic-tac-toe | 32 | V, disjoint successors | All signed moves VNM | 39.63 ± 3.39 | 24.24 ± 6.09 | 36.13 ± 7.06 | 0.04 ± 0.09 |
| Tic-tac-toe | 32 | V, disjoint successors | Discounted signed moves VNM | 39.59 ± 3.32 | 25.06 ± 7.29 | 35.35 ± 7.77 | 0.04 ± 0.09 |
| Tic-tac-toe | 32 | V, disjoint successors | Centered VNM | 32.70 ± 4.08 | 29.93 ± 12.09 | 37.36 ± 13.85 | -0.05 ± 0.16 |
| Connect Four | 24 | Q, original visits | Direct | 33.61 ± 5.18 | 0.07 ± 0.19 | 66.32 ± 5.21 | -0.33 ± 0.10 |
| Connect Four | 24 | Q, original visits | Terminal VNM | 36.21 ± 4.42 | 0.23 ± 0.38 | 63.56 ± 4.48 | -0.27 ± 0.09 |
| Connect Four | 24 | Q, original visits | All signed moves VNM | 39.00 ± 4.06 | 0.67 ± 0.64 | 60.34 ± 4.15 | -0.21 ± 0.08 |
| Connect Four | 24 | Q, original visits | Discounted signed moves VNM | 55.01 ± 6.35 | 0.15 ± 0.34 | 44.84 ± 6.42 | 0.10 ± 0.13 |
| Connect Four | 24 | Q, original visits | Centered VNM | 33.61 ± 5.18 | 0.07 ± 0.19 | 66.32 ± 5.21 | -0.33 ± 0.10 |
| Connect Four | 24 | Q, all alternative paths fitted | Direct | 31.97 ± 4.85 | 0.15 ± 0.32 | 67.89 ± 4.94 | -0.36 ± 0.10 |
| Connect Four | 24 | Q, all alternative paths fitted | Terminal VNM | 36.21 ± 4.15 | 0.18 ± 0.23 | 63.61 ± 4.12 | -0.27 ± 0.08 |
| Connect Four | 24 | Q, all alternative paths fitted | All signed moves VNM | 38.41 ± 5.03 | 0.78 ± 0.63 | 60.81 ± 4.99 | -0.22 ± 0.10 |
| Connect Four | 24 | Q, all alternative paths fitted | Discounted signed moves VNM | 55.88 ± 4.70 | 0.31 ± 0.77 | 43.82 ± 4.97 | 0.12 ± 0.10 |
| Connect Four | 24 | Q, all alternative paths fitted | Centered VNM | 31.97 ± 4.85 | 0.15 ± 0.32 | 67.89 ± 4.94 | -0.36 ± 0.10 |
| Connect Four | 24 | V, all observed parents | Direct | 34.26 ± 5.96 | 0.31 ± 0.55 | 65.43 ± 6.23 | -0.31 ± 0.12 |
| Connect Four | 24 | V, all observed parents | Terminal VNM | 40.09 ± 5.18 | 0.26 ± 0.61 | 59.65 ± 5.18 | -0.20 ± 0.10 |
| Connect Four | 24 | V, all observed parents | All signed moves VNM | 42.81 ± 5.45 | 0.83 ± 0.69 | 56.36 ± 5.37 | -0.14 ± 0.11 |
| Connect Four | 24 | V, all observed parents | Discounted signed moves VNM | 58.40 ± 5.43 | 0.31 ± 0.64 | 41.29 ± 5.47 | 0.17 ± 0.11 |
| Connect Four | 24 | V, all observed parents | Centered VNM | 34.26 ± 5.96 | 0.31 ± 0.55 | 65.43 ± 6.23 | -0.31 ± 0.12 |
| Connect Four | 24 | V, disjoint successors | Direct | 34.29 ± 5.94 | 0.37 ± 0.57 | 65.33 ± 6.23 | -0.31 ± 0.12 |
| Connect Four | 24 | V, disjoint successors | Terminal VNM | 39.81 ± 5.48 | 0.18 ± 0.30 | 60.01 ± 5.48 | -0.20 ± 0.11 |
| Connect Four | 24 | V, disjoint successors | All signed moves VNM | 42.58 ± 5.03 | 0.72 ± 0.61 | 56.71 ± 4.87 | -0.14 ± 0.10 |
| Connect Four | 24 | V, disjoint successors | Discounted signed moves VNM | 57.60 ± 5.24 | 0.33 ± 0.73 | 42.07 ± 5.14 | 0.16 ± 0.10 |
| Connect Four | 24 | V, disjoint successors | Centered VNM | 34.29 ± 5.94 | 0.37 ± 0.57 | 65.33 ± 6.23 | -0.31 ± 0.12 |
| 4×4 Reversi | 24 | Q, original visits | Direct | 54.93 ± 3.40 | 4.57 ± 1.66 | 40.49 ± 2.78 | 0.14 ± 0.06 |
| 4×4 Reversi | 24 | Q, original visits | Terminal VNM | 50.52 ± 3.45 | 5.24 ± 2.11 | 44.24 ± 3.09 | 0.06 ± 0.06 |
| 4×4 Reversi | 24 | Q, original visits | All signed moves VNM | 58.90 ± 1.99 | 3.53 ± 0.85 | 37.57 ± 2.01 | 0.21 ± 0.04 |
| 4×4 Reversi | 24 | Q, original visits | Discounted signed moves VNM | 59.83 ± 2.40 | 3.74 ± 1.10 | 36.43 ± 1.99 | 0.23 ± 0.04 |
| 4×4 Reversi | 24 | Q, original visits | Centered VNM | 54.93 ± 3.40 | 4.57 ± 1.66 | 40.49 ± 2.78 | 0.14 ± 0.06 |
| 4×4 Reversi | 24 | Q, all alternative paths fitted | Direct | 62.52 ± 2.61 | 2.41 ± 1.04 | 35.07 ± 2.53 | 0.27 ± 0.05 |
| 4×4 Reversi | 24 | Q, all alternative paths fitted | Terminal VNM | 51.51 ± 3.04 | 6.84 ± 2.22 | 41.65 ± 2.90 | 0.10 ± 0.06 |
| 4×4 Reversi | 24 | Q, all alternative paths fitted | All signed moves VNM | 63.75 ± 1.68 | 2.51 ± 0.90 | 33.74 ± 1.70 | 0.30 ± 0.03 |
| 4×4 Reversi | 24 | Q, all alternative paths fitted | Discounted signed moves VNM | 64.96 ± 2.05 | 2.02 ± 0.82 | 33.02 ± 1.98 | 0.32 ± 0.04 |
| 4×4 Reversi | 24 | Q, all alternative paths fitted | Centered VNM | 62.52 ± 2.61 | 2.41 ± 1.04 | 35.07 ± 2.53 | 0.27 ± 0.05 |
| 4×4 Reversi | 24 | V, all observed parents | Direct | 56.38 ± 2.61 | 4.02 ± 1.77 | 39.60 ± 2.17 | 0.17 ± 0.04 |
| 4×4 Reversi | 24 | V, all observed parents | Terminal VNM | 51.74 ± 3.63 | 5.11 ± 1.71 | 43.15 ± 3.02 | 0.09 ± 0.06 |
| 4×4 Reversi | 24 | V, all observed parents | All signed moves VNM | 59.52 ± 2.45 | 3.27 ± 1.16 | 37.21 ± 2.12 | 0.22 ± 0.04 |
| 4×4 Reversi | 24 | V, all observed parents | Discounted signed moves VNM | 59.77 ± 2.62 | 3.66 ± 1.93 | 36.57 ± 2.18 | 0.23 ± 0.04 |
| 4×4 Reversi | 24 | V, all observed parents | Centered VNM | 56.38 ± 2.61 | 4.02 ± 1.77 | 39.60 ± 2.17 | 0.17 ± 0.04 |
| 4×4 Reversi | 24 | V, disjoint successors | Direct | 56.30 ± 2.55 | 4.12 ± 1.72 | 39.58 ± 2.12 | 0.17 ± 0.04 |
| 4×4 Reversi | 24 | V, disjoint successors | Terminal VNM | 53.39 ± 3.36 | 4.85 ± 1.62 | 41.76 ± 3.04 | 0.12 ± 0.06 |
| 4×4 Reversi | 24 | V, disjoint successors | All signed moves VNM | 59.39 ± 2.32 | 3.19 ± 1.23 | 37.42 ± 1.94 | 0.22 ± 0.04 |
| 4×4 Reversi | 24 | V, disjoint successors | Discounted signed moves VNM | 59.88 ± 2.50 | 3.56 ± 1.62 | 36.56 ± 2.36 | 0.23 ± 0.05 |
| 4×4 Reversi | 24 | V, disjoint successors | Centered VNM | 56.30 ± 2.55 | 4.12 ± 1.72 | 39.58 ± 2.12 | 0.17 ± 0.04 |

Per-fit board choices, alternative-path coverage, counts and means are retained in the individual source directories. [All estimates](training_results.jsonl.gz) · [Paired changes, source-level differences, resampling intervals and raw and adjusted p-values](training_contrasts.jsonl.gz)
