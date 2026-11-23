# Matched measurements of move recovery

The original is the frozen neural player. V scores only the resulting board. Q also sees the current board and move. Direct selection takes the highest immediate score. Von Neumann–Morgenstern (VNM) selection compares expected scores of complete plays. The final-move rule uses the last mover’s signed frequency. A local shortfall is the played score minus the mover’s best available score. The alternative VNM utility sums these differences, positive sign for the first player's terms and negative for the second. Both seats use the specified continuing player. Exact frequencies give the original’s move a 1 and every alternative a 0. For each comparison using exact frequencies, the V or Q prefix identifies the original fit being controlled. Each game uses two board splits and two independently initialized fits per original. Values show mean ± sample standard deviation after averaging fits per original. A dash denotes a measurement outside that intervention. [Design](DESIGN.md) · [Report and method definitions](../../AGREEMENT_DIAGNOSIS.md)

| Game | Originals | Method | Same move (%) | Strictly beaten favorite (%) | Favorite lost a tie (%) | Wins (%) | Draws (%) | Losses (%) | Points/game |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Tic-tac-toe | 32 | Exact original frequency, direct (privileged) | 100.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | — | — | — | — |
| Tic-tac-toe | 32 | Mean exact incoming frequency, direct V (privileged) | 95.43 ± 0.88 | 0.76 ± 0.29 | 3.81 ± 0.89 | — | — | — | — |
| Tic-tac-toe | 32 | Action preferences plus game consequences | 100.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 45.55 ± 6.01 | 54.45 ± 6.01 | 0.00 ± 0.00 | 0.46 ± 0.06 |
| Tic-tac-toe | 32 | Action preferences alone | 54.83 ± 3.98 | 45.17 ± 3.98 | 0.00 ± 0.00 | 25.97 ± 5.33 | 13.29 ± 13.15 | 60.74 ± 12.26 | -0.35 ± 0.14 |
| Tic-tac-toe | 32 | Game consequences, signed local shortfalls VNM | 100.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 45.55 ± 6.01 | 54.45 ± 6.01 | 0.00 ± 0.00 | 0.46 ± 0.06 |
| Tic-tac-toe | 32 | Q direct | 78.39 ± 2.14 | 21.61 ± 2.14 | 0.00 ± 0.00 | 44.04 ± 2.90 | 33.66 ± 7.89 | 22.30 ± 7.96 | 0.22 ± 0.09 |
| Tic-tac-toe | 32 | Q direct, before bounding scores | 78.39 ± 2.14 | 21.61 ± 2.14 | 0.00 ± 0.00 | — | — | — | — |
| Tic-tac-toe | 32 | Q final-move VNM, fitted continuations | 67.10 ± 2.50 | 32.23 ± 2.34 | 0.68 ± 0.28 | 40.99 ± 4.11 | 30.17 ± 4.99 | 28.84 ± 6.18 | 0.12 ± 0.09 |
| Tic-tac-toe | 32 | Q final-move VNM, fitted scores and original continuations | 68.76 ± 2.55 | 30.52 ± 2.41 | 0.71 ± 0.33 | — | — | — | — |
| Tic-tac-toe | 32 | Q final-move VNM, exact frequencies and fitted continuations | 63.83 ± 5.30 | 8.70 ± 1.43 | 27.47 ± 5.06 | — | — | — | — |
| Tic-tac-toe | 32 | Q final-move VNM, exact frequencies and original continuations | 64.76 ± 6.33 | 0.00 ± 0.00 | 35.24 ± 6.33 | — | — | — | — |
| Tic-tac-toe | 32 | Q signed local shortfalls VNM, fitted continuations | 78.39 ± 2.14 | 21.61 ± 2.14 | 0.00 ± 0.00 | 44.04 ± 2.90 | 33.66 ± 7.89 | 22.30 ± 7.96 | 0.22 ± 0.09 |
| Tic-tac-toe | 32 | Q signed local shortfalls VNM, original continuations | 78.05 ± 2.08 | 21.95 ± 2.08 | 0.00 ± 0.00 | — | — | — | — |
| Tic-tac-toe | 32 | Q direct, half training boards | 77.87 ± 2.17 | — | — | 43.49 ± 3.44 | — | — | — |
| Tic-tac-toe | 32 | Q final frequency with last mover's sign removed | 27.25 ± 2.19 | — | — | — | — | — | — |
| Tic-tac-toe | 32 | V direct | 73.95 ± 2.56 | 25.92 ± 2.53 | 0.12 ± 0.10 | 40.00 ± 3.20 | 35.00 ± 6.01 | 25.00 ± 7.25 | 0.15 ± 0.09 |
| Tic-tac-toe | 32 | V direct, before bounding scores | 73.99 ± 2.54 | 26.01 ± 2.54 | 0.00 ± 0.00 | — | — | — | — |
| Tic-tac-toe | 32 | V final-move VNM, fitted continuations | 68.56 ± 2.94 | 30.12 ± 2.70 | 1.32 ± 0.55 | 44.19 ± 4.69 | 29.88 ± 6.96 | 25.93 ± 6.89 | 0.18 ± 0.10 |
| Tic-tac-toe | 32 | V final-move VNM, fitted scores and original continuations | 70.18 ± 3.01 | 28.54 ± 2.78 | 1.28 ± 0.56 | — | — | — | — |
| Tic-tac-toe | 32 | V final-move VNM, exact frequencies and fitted continuations | 63.41 ± 5.39 | 11.10 ± 1.62 | 25.49 ± 5.05 | — | — | — | — |
| Tic-tac-toe | 32 | V final-move VNM, exact frequencies and original continuations | 64.76 ± 6.33 | 0.00 ± 0.00 | 35.24 ± 6.33 | — | — | — | — |
| Tic-tac-toe | 32 | V signed local shortfalls VNM, fitted continuations | 73.95 ± 2.56 | 25.92 ± 2.53 | 0.12 ± 0.10 | 40.00 ± 3.20 | 35.00 ± 6.01 | 25.00 ± 7.25 | 0.15 ± 0.09 |
| Tic-tac-toe | 32 | V signed local shortfalls VNM, original continuations | 70.01 ± 3.06 | 29.94 ± 3.04 | 0.05 ± 0.06 | — | — | — | — |
| Tic-tac-toe | 32 | V direct, half training boards | 72.37 ± 2.54 | — | — | 38.47 ± 3.61 | — | — | — |
| Tic-tac-toe | 32 | V final frequency with last mover's sign removed | 32.50 ± 2.79 | — | — | — | — | — | — |
| Tic-tac-toe | 32 | Game outcome plus elicited ordering, low-index fallback | 100.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 45.55 ± 6.01 | 54.45 ± 6.01 | 0.00 ± 0.00 | 0.46 ± 0.06 |
| Tic-tac-toe | 32 | Game outcome plus elicited ordering, high-index fallback | 100.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 45.55 ± 6.01 | 54.45 ± 6.01 | 0.00 ± 0.00 | 0.46 ± 0.06 |
| Tic-tac-toe | 32 | V final-move VNM, direct breaks ties | 69.14 ± 2.76 | — | — | — | — | — | — |
| Tic-tac-toe | 32 | V exact-frequency final-move VNM, exact direct breaks ties (privileged) | 100.00 ± 0.00 | — | — | — | — | — | — |
| Tic-tac-toe | 32 | Q final-move VNM, direct breaks ties | 67.48 ± 2.42 | — | — | — | — | — | — |
| Tic-tac-toe | 32 | Q exact-frequency final-move VNM, exact direct breaks ties (privileged) | 100.00 ± 0.00 | — | — | — | — | — | — |
| Connect Four | 24 | Exact original frequency, direct (privileged) | 100.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | — | — | — | — |
| Connect Four | 24 | Mean exact incoming frequency, direct V (privileged) | 100.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | — | — | — | — |
| Connect Four | 24 | Action preferences plus game consequences | 97.21 ± 2.63 | 2.79 ± 2.63 | 0.00 ± 0.00 | 56.30 ± 10.94 | 0.63 ± 2.16 | 43.07 ± 12.05 | 0.13 ± 0.23 |
| Connect Four | 24 | Action preferences alone | 88.49 ± 11.70 | 11.51 ± 11.70 | 0.00 ± 0.00 | 49.43 ± 4.86 | 0.00 ± 0.00 | 50.57 ± 4.86 | -0.01 ± 0.10 |
| Connect Four | 24 | Game consequences, signed local shortfalls VNM | 97.21 ± 2.63 | 2.79 ± 2.63 | 0.00 ± 0.00 | 56.30 ± 10.94 | 0.63 ± 2.16 | 43.07 ± 12.05 | 0.13 ± 0.23 |
| Connect Four | 24 | Q normalized fit, original visits | 93.58 ± 4.30 | 6.42 ± 4.30 | 0.00 ± 0.00 | 56.07 ± 12.00 | 1.06 ± 2.00 | 42.87 ± 13.50 | 0.13 ± 0.25 |
| Connect Four | 24 | Q normalized fit, added visits | 98.23 ± 1.42 | 1.77 ± 1.42 | 0.00 ± 0.00 | 57.71 ± 11.39 | 0.65 ± 1.52 | 41.63 ± 12.26 | 0.16 ± 0.24 |
| Connect Four | 24 | Q normalized fit, added visits, signed local shortfalls VNM | 98.23 ± 1.42 | 1.77 ± 1.42 | 0.00 ± 0.00 | 57.71 ± 11.39 | 0.65 ± 1.52 | 41.63 ± 12.26 | 0.16 ± 0.24 |
| Connect Four | 24 | Q normalized fit, original visits, signed local shortfalls VNM | 93.58 ± 4.30 | 6.42 ± 4.30 | 0.00 ± 0.00 | 56.07 ± 12.00 | 1.06 ± 2.00 | 42.87 ± 13.50 | 0.13 ± 0.25 |
| Connect Four | 24 | Q direct | 92.98 ± 6.24 | 6.76 ± 6.03 | 0.25 ± 0.39 | 57.45 ± 11.30 | 1.04 ± 2.07 | 41.50 ± 12.55 | 0.16 ± 0.24 |
| Connect Four | 24 | Q direct, before bounding scores | 93.05 ± 6.16 | 6.95 ± 6.16 | 0.00 ± 0.00 | — | — | — | — |
| Connect Four | 24 | Q final-move VNM, fitted continuations | 24.86 ± 4.38 | 69.59 ± 5.30 | 5.55 ± 5.73 | 49.37 ± 17.69 | 0.41 ± 0.63 | 50.23 ± 17.98 | -0.01 ± 0.36 |
| Connect Four | 24 | Q final-move VNM, fitted scores and original continuations | 25.98 ± 4.44 | 69.96 ± 5.50 | 4.06 ± 3.81 | — | — | — | — |
| Connect Four | 24 | Q final-move VNM, exact frequencies and fitted continuations | 22.32 ± 11.10 | 40.66 ± 9.93 | 37.02 ± 13.96 | — | — | — | — |
| Connect Four | 24 | Q final-move VNM, exact frequencies and original continuations | 21.29 ± 13.06 | 34.27 ± 6.80 | 44.44 ± 14.20 | — | — | — | — |
| Connect Four | 24 | Q signed local shortfalls VNM, fitted continuations | 92.98 ± 6.24 | 6.76 ± 6.03 | 0.25 ± 0.39 | 57.45 ± 11.30 | 1.04 ± 2.07 | 41.50 ± 12.55 | 0.16 ± 0.24 |
| Connect Four | 24 | Q signed local shortfalls VNM, original continuations | 84.44 ± 8.54 | 15.40 ± 8.40 | 0.16 ± 0.26 | — | — | — | — |
| Connect Four | 24 | Q direct, added visits | 95.51 ± 4.35 | 4.45 ± 4.34 | 0.03 ± 0.09 | 56.84 ± 10.35 | 0.52 ± 1.41 | 42.64 ± 11.02 | 0.14 ± 0.21 |
| Connect Four | 24 | Q final frequency with last mover's sign removed | 18.96 ± 7.83 | — | — | — | — | — | — |
| Connect Four | 24 | V direct | 76.89 ± 4.43 | 21.92 ± 4.32 | 1.19 ± 0.59 | 43.99 ± 20.40 | 0.26 ± 0.70 | 55.75 ± 20.68 | -0.12 ± 0.41 |
| Connect Four | 24 | V direct, before bounding scores | 77.47 ± 4.38 | 22.42 ± 4.33 | 0.10 ± 0.20 | — | — | — | — |
| Connect Four | 24 | V final-move VNM, fitted continuations | 23.81 ± 7.06 | 72.25 ± 7.95 | 3.95 ± 2.80 | 53.73 ± 18.27 | 0.21 ± 0.33 | 46.06 ± 18.46 | 0.08 ± 0.37 |
| Connect Four | 24 | V final-move VNM, fitted scores and original continuations | 25.89 ± 5.81 | 71.46 ± 7.23 | 2.65 ± 2.56 | — | — | — | — |
| Connect Four | 24 | V final-move VNM, exact frequencies and fitted continuations | 21.22 ± 12.96 | 36.58 ± 6.31 | 42.20 ± 12.94 | — | — | — | — |
| Connect Four | 24 | V final-move VNM, exact frequencies and original continuations | 21.29 ± 13.06 | 34.27 ± 6.80 | 44.44 ± 14.20 | — | — | — | — |
| Connect Four | 24 | V signed local shortfalls VNM, fitted continuations | 76.89 ± 4.43 | 21.92 ± 4.32 | 1.19 ± 0.59 | 43.99 ± 20.40 | 0.26 ± 0.70 | 55.75 ± 20.68 | -0.12 ± 0.41 |
| Connect Four | 24 | V signed local shortfalls VNM, original continuations | 53.93 ± 8.13 | 45.64 ± 8.27 | 0.44 ± 0.35 | — | — | — | — |
| Connect Four | 24 | V direct, added visits | 81.22 ± 4.29 | 18.33 ± 4.15 | 0.45 ± 0.43 | 51.12 ± 14.54 | 0.16 ± 0.43 | 48.71 ± 14.67 | 0.02 ± 0.29 |
| Connect Four | 24 | V final frequency with last mover's sign removed | 19.00 ± 4.66 | — | — | — | — | — | — |
| Connect Four | 24 | V final-move VNM, direct breaks ties | 26.56 ± 7.87 | — | — | — | — | — | — |
| Connect Four | 24 | V exact-frequency final-move VNM, exact direct breaks ties (privileged) | 65.73 ± 6.80 | — | — | — | — | — | — |
| Connect Four | 24 | Q final-move VNM, direct breaks ties | 29.95 ± 5.01 | — | — | — | — | — | — |
| Connect Four | 24 | Q exact-frequency final-move VNM, exact direct breaks ties (privileged) | 65.73 ± 6.80 | — | — | — | — | — | — |
| 4×4 Reversi | 24 | Exact original frequency, direct (privileged) | 100.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | — | — | — | — |
| 4×4 Reversi | 24 | Mean exact incoming frequency, direct V (privileged) | 99.68 ± 0.26 | 0.28 ± 0.22 | 0.03 ± 0.08 | — | — | — | — |
| 4×4 Reversi | 24 | Action preferences plus game consequences | 99.84 ± 0.20 | 0.16 ± 0.20 | 0.00 ± 0.00 | 67.29 ± 2.09 | 1.37 ± 0.78 | 31.35 ± 1.94 | 0.36 ± 0.04 |
| 4×4 Reversi | 24 | Action preferences alone | 92.06 ± 0.97 | 7.94 ± 0.97 | 0.00 ± 0.00 | 46.68 ± 3.92 | 5.57 ± 2.16 | 47.75 ± 3.45 | -0.01 ± 0.07 |
| 4×4 Reversi | 24 | Game consequences, signed local shortfalls VNM | 99.84 ± 0.20 | 0.16 ± 0.20 | 0.00 ± 0.00 | 67.29 ± 2.09 | 1.37 ± 0.78 | 31.35 ± 1.94 | 0.36 ± 0.04 |
| 4×4 Reversi | 24 | Q normalized fit, original visits | 87.89 ± 1.07 | 12.11 ± 1.07 | 0.00 ± 0.00 | 54.75 ± 3.79 | 3.86 ± 1.50 | 41.39 ± 4.05 | 0.13 ± 0.08 |
| 4×4 Reversi | 24 | Q normalized fit, added visits | 90.96 ± 1.13 | 9.04 ± 1.13 | 0.00 ± 0.00 | 56.48 ± 4.62 | 3.47 ± 1.24 | 40.06 ± 4.74 | 0.16 ± 0.09 |
| 4×4 Reversi | 24 | Q normalized fit, added visits, signed local shortfalls VNM | 90.96 ± 1.13 | 9.04 ± 1.13 | 0.00 ± 0.00 | 56.48 ± 4.62 | 3.47 ± 1.24 | 40.06 ± 4.74 | 0.16 ± 0.09 |
| 4×4 Reversi | 24 | Q normalized fit, original visits, signed local shortfalls VNM | 87.89 ± 1.07 | 12.11 ± 1.07 | 0.00 ± 0.00 | 54.75 ± 3.79 | 3.86 ± 1.50 | 41.39 ± 4.05 | 0.13 ± 0.08 |
| 4×4 Reversi | 24 | Q direct | 85.86 ± 1.26 | 14.14 ± 1.26 | 0.00 ± 0.00 | 49.87 ± 4.78 | 5.47 ± 2.26 | 44.66 ± 4.00 | 0.05 ± 0.09 |
| 4×4 Reversi | 24 | Q direct, before bounding scores | 85.86 ± 1.25 | 14.14 ± 1.25 | 0.00 ± 0.00 | — | — | — | — |
| 4×4 Reversi | 24 | Q final-move VNM, fitted continuations | 79.89 ± 1.67 | 19.95 ± 1.74 | 0.16 ± 0.24 | 36.87 ± 5.02 | 9.68 ± 3.33 | 53.45 ± 4.58 | -0.17 ± 0.09 |
| 4×4 Reversi | 24 | Q final-move VNM, fitted scores and original continuations | 80.78 ± 1.42 | 19.07 ± 1.47 | 0.15 ± 0.22 | — | — | — | — |
| 4×4 Reversi | 24 | Q final-move VNM, exact frequencies and fitted continuations | 90.59 ± 0.99 | 6.06 ± 0.89 | 3.35 ± 0.55 | — | — | — | — |
| 4×4 Reversi | 24 | Q final-move VNM, exact frequencies and original continuations | 92.29 ± 1.05 | 4.80 ± 0.87 | 2.91 ± 0.48 | — | — | — | — |
| 4×4 Reversi | 24 | Q signed local shortfalls VNM, fitted continuations | 85.86 ± 1.26 | 14.14 ± 1.26 | 0.00 ± 0.00 | 49.87 ± 4.78 | 5.47 ± 2.26 | 44.66 ± 4.00 | 0.05 ± 0.09 |
| 4×4 Reversi | 24 | Q signed local shortfalls VNM, original continuations | 84.25 ± 1.23 | 15.75 ± 1.23 | 0.00 ± 0.00 | — | — | — | — |
| 4×4 Reversi | 24 | Q direct, added visits | 90.66 ± 1.18 | 9.34 ± 1.18 | 0.00 ± 0.00 | 47.18 ± 5.07 | 5.83 ± 1.87 | 46.99 ± 4.56 | 0.00 ± 0.09 |
| 4×4 Reversi | 24 | Q final frequency with last mover's sign removed | 76.69 ± 1.74 | — | — | — | — | — | — |
| 4×4 Reversi | 24 | V direct | 83.91 ± 1.49 | 16.09 ± 1.49 | 0.01 ± 0.02 | 46.27 ± 5.90 | 5.03 ± 1.56 | 48.70 ± 5.37 | -0.02 ± 0.11 |
| 4×4 Reversi | 24 | V direct, before bounding scores | 83.88 ± 1.47 | 16.12 ± 1.47 | 0.00 ± 0.00 | — | — | — | — |
| 4×4 Reversi | 24 | V final-move VNM, fitted continuations | 82.70 ± 1.60 | 16.97 ± 1.68 | 0.33 ± 0.28 | 34.78 ± 5.39 | 10.01 ± 2.75 | 55.21 ± 5.09 | -0.20 ± 0.10 |
| 4×4 Reversi | 24 | V final-move VNM, fitted scores and original continuations | 83.45 ± 1.35 | 16.39 ± 1.40 | 0.15 ± 0.21 | — | — | — | — |
| 4×4 Reversi | 24 | V final-move VNM, exact frequencies and fitted continuations | 90.62 ± 1.05 | 6.11 ± 1.04 | 3.27 ± 0.53 | — | — | — | — |
| 4×4 Reversi | 24 | V final-move VNM, exact frequencies and original continuations | 92.29 ± 1.05 | 4.80 ± 0.87 | 2.91 ± 0.48 | — | — | — | — |
| 4×4 Reversi | 24 | V signed local shortfalls VNM, fitted continuations | 83.91 ± 1.49 | 16.09 ± 1.49 | 0.01 ± 0.02 | 46.27 ± 5.90 | 5.03 ± 1.56 | 48.70 ± 5.37 | -0.02 ± 0.11 |
| 4×4 Reversi | 24 | V signed local shortfalls VNM, original continuations | 82.09 ± 1.45 | 17.91 ± 1.45 | 0.00 ± 0.00 | — | — | — | — |
| 4×4 Reversi | 24 | V direct, added visits | 88.66 ± 1.27 | 11.33 ± 1.27 | 0.01 ± 0.03 | 47.92 ± 5.02 | 4.23 ± 2.11 | 47.85 ± 4.19 | 0.00 ± 0.09 |
| 4×4 Reversi | 24 | V final frequency with last mover's sign removed | 79.69 ± 1.43 | — | — | — | — | — | — |
| 4×4 Reversi | 24 | Game outcome plus elicited ordering, low-index fallback | 100.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 67.35 ± 2.05 | 1.35 ± 0.77 | 31.30 ± 1.97 | 0.36 ± 0.04 |
| 4×4 Reversi | 24 | Game outcome plus elicited ordering, high-index fallback | 99.95 ± 0.11 | 0.05 ± 0.11 | 0.00 ± 0.00 | 67.35 ± 2.05 | 1.35 ± 0.77 | 31.30 ± 1.97 | 0.36 ± 0.04 |
| 4×4 Reversi | 24 | V final-move VNM, direct breaks ties | 80.85 ± 1.57 | — | — | — | — | — | — |
| 4×4 Reversi | 24 | V exact-frequency final-move VNM, exact direct breaks ties (privileged) | 95.20 ± 0.87 | — | — | — | — | — | — |
| 4×4 Reversi | 24 | Q final-move VNM, direct breaks ties | 79.59 ± 1.65 | — | — | — | — | — | — |
| 4×4 Reversi | 24 | Q exact-frequency final-move VNM, exact direct breaks ties (privileged) | 95.20 ± 0.87 | — | — | — | — | — | — |

## Paired changes

Each difference uses the same original and board or seat cases. Rate differences are percentage points. The 95% intervals describe uncertainty across independent original blocks. Raw p is the two-sided probability of a result at least as extreme under zero mean difference. Adjusted p uses Holm’s procedure over this diagnostic family. Exact identities have zero spread and an undefined Student p-value. The machine-readable data also retain every individual difference.

| Game | Originals | Method minus comparator | Measurement | Difference ± spread | 95% interval | Raw p | Adjusted p |
|---|---:|---|---|---:|---:|---:|---:|
| Tic-tac-toe | 32 | Action preferences plus game consequences minus Action preferences alone | match | 45.17 ± 3.98 | [43.74, 46.61] | 1.59e-34 | 4.64e-32 |
| Tic-tac-toe | 32 | Action preferences plus game consequences minus Action preferences alone | win | 19.58 ± 7.39 | [16.92, 22.24] | 9.31e-16 | 2.29e-13 |
| Tic-tac-toe | 32 | Action preferences plus game consequences minus Action preferences alone | draw | 41.16 ± 13.79 | [36.19, 46.14] | 3.52e-17 | 8.79e-15 |
| Tic-tac-toe | 32 | Action preferences plus game consequences minus Action preferences alone | loss | -60.74 ± 12.26 | [-65.16, -56.32] | 1.39e-23 | 3.76e-21 |
| Tic-tac-toe | 32 | Action preferences plus game consequences minus Action preferences alone | points | 0.80 ± 0.15 | [0.75, 0.86] | 9.31e-25 | 2.56e-22 |
| Tic-tac-toe | 32 | Action preferences plus game consequences minus Q direct | match | 21.61 ± 2.14 | [20.84, 22.39] | 5.67e-33 | 1.64e-30 |
| Tic-tac-toe | 32 | Action preferences plus game consequences minus Q direct | win | 1.51 ± 4.96 | [-0.27, 3.30] | 0.0942 | 1 |
| Tic-tac-toe | 32 | Action preferences plus game consequences minus Q direct | draw | 20.79 ± 8.56 | [17.70, 23.88] | 9.91e-15 | 2.4e-12 |
| Tic-tac-toe | 32 | Action preferences plus game consequences minus Q direct | loss | -22.30 ± 7.96 | [-25.17, -19.43] | 2.06e-16 | 5.09e-14 |
| Tic-tac-toe | 32 | Action preferences plus game consequences minus Q direct | points | 0.24 ± 0.10 | [0.20, 0.27] | 2.41e-14 | 5.78e-12 |
| Tic-tac-toe | 32 | Action preferences plus game consequences minus Q direct | multi_match | 22.77 ± 2.21 | [21.98, 23.57] | 2.82e-33 | 8.2e-31 |
| Tic-tac-toe | 32 | Game consequences, signed local shortfalls VNM minus Action preferences plus game consequences | match | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | Game consequences, signed local shortfalls VNM minus Action preferences plus game consequences | win | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | Game consequences, signed local shortfalls VNM minus Action preferences plus game consequences | draw | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | Game consequences, signed local shortfalls VNM minus Action preferences plus game consequences | loss | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | Game consequences, signed local shortfalls VNM minus Action preferences plus game consequences | points | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | Game outcome plus elicited ordering, low-index fallback minus Game outcome plus elicited ordering, high-index fallback | match | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | Game outcome plus elicited ordering, low-index fallback minus Game outcome plus elicited ordering, high-index fallback | win | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | Game outcome plus elicited ordering, low-index fallback minus Game outcome plus elicited ordering, high-index fallback | draw | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | Game outcome plus elicited ordering, low-index fallback minus Game outcome plus elicited ordering, high-index fallback | loss | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | Game outcome plus elicited ordering, low-index fallback minus Game outcome plus elicited ordering, high-index fallback | points | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | Game outcome plus elicited ordering, low-index fallback minus Action preferences plus game consequences | match | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | Game outcome plus elicited ordering, low-index fallback minus Action preferences plus game consequences | win | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | Game outcome plus elicited ordering, low-index fallback minus Action preferences plus game consequences | draw | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | Game outcome plus elicited ordering, low-index fallback minus Action preferences plus game consequences | loss | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | Game outcome plus elicited ordering, low-index fallback minus Action preferences plus game consequences | points | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | V direct, before bounding scores minus V direct | match | 0.03 ± 0.07 | [0.01, 0.06] | 0.0176 | 1 |
| Tic-tac-toe | 32 | V final-move VNM, fitted continuations minus V direct | match | -5.39 ± 2.54 | [-6.30, -4.47] | 3.36e-13 | 7.74e-11 |
| Tic-tac-toe | 32 | V final-move VNM, fitted continuations minus V direct | win | 4.19 ± 5.14 | [2.33, 6.04] | 6.58e-05 | 0.0108 |
| Tic-tac-toe | 32 | V final-move VNM, fitted continuations minus V direct | draw | -5.12 ± 9.23 | [-8.45, -1.79] | 0.00373 | 0.493 |
| Tic-tac-toe | 32 | V final-move VNM, fitted continuations minus V direct | loss | 0.93 ± 7.27 | [-1.69, 3.55] | 0.475 | 1 |
| Tic-tac-toe | 32 | V final-move VNM, fitted continuations minus V direct | points | 0.03 ± 0.09 | [0.00, 0.06] | 0.0395 | 1 |
| Tic-tac-toe | 32 | V signed local shortfalls VNM, fitted continuations minus V direct | match | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | V signed local shortfalls VNM, fitted continuations minus V direct | win | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | V signed local shortfalls VNM, fitted continuations minus V direct | draw | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | V signed local shortfalls VNM, fitted continuations minus V direct | loss | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | V signed local shortfalls VNM, fitted continuations minus V direct | points | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | V signed local shortfalls VNM, original continuations minus V direct | match | -3.94 ± 1.05 | [-4.32, -3.57] | 4.52e-20 | 1.17e-17 |
| Tic-tac-toe | 32 | V direct, half training boards minus V direct | match | -1.59 ± 0.68 | [-1.83, -1.34] | 3.49e-14 | 8.31e-12 |
| Tic-tac-toe | 32 | V direct, half training boards minus V direct | win | -1.54 ± 1.70 | [-2.15, -0.92] | 1.54e-05 | 0.00271 |
| Tic-tac-toe | 32 | Exact original frequency, direct (privileged) minus V direct | match | 26.05 ± 2.56 | [25.13, 26.97] | 4.27e-33 | 1.24e-30 |
| Tic-tac-toe | 32 | Mean exact incoming frequency, direct V (privileged) minus V direct | match | 21.48 ± 2.53 | [20.57, 22.40] | 1.2e-30 | 3.46e-28 |
| Tic-tac-toe | 32 | V final-move VNM, exact frequencies and original continuations minus Exact original frequency, direct (privileged) | match | -35.24 ± 6.33 | [-37.52, -32.96] | 4.22e-25 | 1.18e-22 |
| Tic-tac-toe | 32 | V final-move VNM, exact frequencies and fitted continuations minus V final-move VNM, fitted continuations | match | -5.16 ± 4.26 | [-6.69, -3.62] | 1.13e-07 | 2.28e-05 |
| Tic-tac-toe | 32 | V final-move VNM, fitted scores and original continuations minus V final-move VNM, fitted continuations | match | 1.61 ± 0.91 | [1.29, 1.94] | 2.96e-11 | 6.62e-09 |
| Tic-tac-toe | 32 | V final-move VNM, exact frequencies and original continuations minus V final-move VNM, exact frequencies and fitted continuations | match | 1.36 ± 1.51 | [0.81, 1.90] | 1.64e-05 | 0.00287 |
| Tic-tac-toe | 32 | V final-move VNM, exact frequencies and original continuations minus V final-move VNM, fitted scores and original continuations | match | -5.42 ± 5.42 | [-7.37, -3.46] | 3.3e-06 | 0.000623 |
| Tic-tac-toe | 32 | V signed local shortfalls VNM, fitted continuations minus V final-move VNM, fitted continuations | match | 5.39 ± 2.54 | [4.47, 6.30] | 3.36e-13 | 7.74e-11 |
| Tic-tac-toe | 32 | V signed local shortfalls VNM, fitted continuations minus V final-move VNM, fitted continuations | win | -4.19 ± 5.14 | [-6.04, -2.33] | 6.58e-05 | 0.0108 |
| Tic-tac-toe | 32 | V signed local shortfalls VNM, fitted continuations minus V final-move VNM, fitted continuations | draw | 5.12 ± 9.23 | [1.79, 8.45] | 0.00373 | 0.493 |
| Tic-tac-toe | 32 | V signed local shortfalls VNM, fitted continuations minus V final-move VNM, fitted continuations | loss | -0.93 ± 7.27 | [-3.55, 1.69] | 0.475 | 1 |
| Tic-tac-toe | 32 | V signed local shortfalls VNM, fitted continuations minus V final-move VNM, fitted continuations | points | -0.03 ± 0.09 | [-0.06, -0.00] | 0.0395 | 1 |
| Tic-tac-toe | 32 | V final frequency with last mover's sign removed minus V final-move VNM, fitted continuations | match | -36.07 ± 2.77 | [-37.07, -35.07] | 2.13e-36 | 6.24e-34 |
| Tic-tac-toe | 32 | V final-move VNM, direct breaks ties minus V final-move VNM, fitted continuations | match | 0.57 ± 0.49 | [0.40, 0.75] | 1.83e-07 | 3.63e-05 |
| Tic-tac-toe | 32 | V exact-frequency final-move VNM, exact direct breaks ties (privileged) minus V final-move VNM, exact frequencies and original continuations | match | 35.24 ± 6.33 | [32.96, 37.52] | 4.22e-25 | 1.18e-22 |
| Tic-tac-toe | 32 | Q direct, before bounding scores minus Q direct | match | -0.00 ± 0.01 | [-0.00, 0.00] | 0.961 | 1 |
| Tic-tac-toe | 32 | Q final-move VNM, fitted continuations minus Q direct | match | -11.29 ± 2.72 | [-12.27, -10.31] | 2.78e-21 | 7.34e-19 |
| Tic-tac-toe | 32 | Q final-move VNM, fitted continuations minus Q direct | win | -3.04 ± 3.62 | [-4.35, -1.74] | 4.29e-05 | 0.00725 |
| Tic-tac-toe | 32 | Q final-move VNM, fitted continuations minus Q direct | draw | -3.49 ± 9.07 | [-6.76, -0.22] | 0.037 | 1 |
| Tic-tac-toe | 32 | Q final-move VNM, fitted continuations minus Q direct | loss | 6.54 ± 9.14 | [3.24, 9.83] | 0.000321 | 0.0484 |
| Tic-tac-toe | 32 | Q final-move VNM, fitted continuations minus Q direct | points | -0.10 ± 0.11 | [-0.13, -0.06] | 1.42e-05 | 0.00256 |
| Tic-tac-toe | 32 | Q final-move VNM, fitted continuations minus Q direct | multi_match | -11.90 ± 2.84 | [-12.92, -10.87] | 2.05e-21 | 5.43e-19 |
| Tic-tac-toe | 32 | Q signed local shortfalls VNM, fitted continuations minus Q direct | match | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | Q signed local shortfalls VNM, fitted continuations minus Q direct | win | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | Q signed local shortfalls VNM, fitted continuations minus Q direct | draw | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | Q signed local shortfalls VNM, fitted continuations minus Q direct | loss | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | Q signed local shortfalls VNM, fitted continuations minus Q direct | points | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Tic-tac-toe | 32 | Q signed local shortfalls VNM, original continuations minus Q direct | match | -0.34 ± 0.49 | [-0.51, -0.16] | 0.000519 | 0.0748 |
| Tic-tac-toe | 32 | Q direct, half training boards minus Q direct | match | -0.52 ± 0.40 | [-0.66, -0.37] | 3.17e-08 | 6.47e-06 |
| Tic-tac-toe | 32 | Q direct, half training boards minus Q direct | win | -0.54 ± 1.60 | [-1.12, 0.03] | 0.0634 | 1 |
| Tic-tac-toe | 32 | Exact original frequency, direct (privileged) minus Q direct | match | 21.61 ± 2.14 | [20.84, 22.39] | 5.67e-33 | 1.64e-30 |
| Tic-tac-toe | 32 | Q final-move VNM, exact frequencies and original continuations minus Exact original frequency, direct (privileged) | match | -35.24 ± 6.33 | [-37.52, -32.96] | 4.22e-25 | 1.18e-22 |
| Tic-tac-toe | 32 | Q final-move VNM, exact frequencies and fitted continuations minus Q final-move VNM, fitted continuations | match | -3.27 ± 4.41 | [-4.86, -1.68] | 0.000212 | 0.0326 |
| Tic-tac-toe | 32 | Q final-move VNM, fitted scores and original continuations minus Q final-move VNM, fitted continuations | match | 1.67 ± 0.72 | [1.41, 1.93] | 4.21e-14 | 9.99e-12 |
| Tic-tac-toe | 32 | Q final-move VNM, exact frequencies and original continuations minus Q final-move VNM, exact frequencies and fitted continuations | match | 0.94 ± 1.44 | [0.42, 1.45] | 0.000895 | 0.128 |
| Tic-tac-toe | 32 | Q final-move VNM, exact frequencies and original continuations minus Q final-move VNM, fitted scores and original continuations | match | -4.00 ± 5.64 | [-6.03, -1.97] | 0.00035 | 0.0522 |
| Tic-tac-toe | 32 | Q signed local shortfalls VNM, fitted continuations minus Q final-move VNM, fitted continuations | match | 11.29 ± 2.72 | [10.31, 12.27] | 2.78e-21 | 7.34e-19 |
| Tic-tac-toe | 32 | Q signed local shortfalls VNM, fitted continuations minus Q final-move VNM, fitted continuations | win | 3.04 ± 3.62 | [1.74, 4.35] | 4.29e-05 | 0.00725 |
| Tic-tac-toe | 32 | Q signed local shortfalls VNM, fitted continuations minus Q final-move VNM, fitted continuations | draw | 3.49 ± 9.07 | [0.22, 6.76] | 0.037 | 1 |
| Tic-tac-toe | 32 | Q signed local shortfalls VNM, fitted continuations minus Q final-move VNM, fitted continuations | loss | -6.54 ± 9.14 | [-9.83, -3.24] | 0.000321 | 0.0484 |
| Tic-tac-toe | 32 | Q signed local shortfalls VNM, fitted continuations minus Q final-move VNM, fitted continuations | points | 0.10 ± 0.11 | [0.06, 0.13] | 1.42e-05 | 0.00256 |
| Tic-tac-toe | 32 | Q final frequency with last mover's sign removed minus Q final-move VNM, fitted continuations | match | -39.85 ± 2.50 | [-40.75, -38.95] | 4.21e-39 | 1.24e-36 |
| Tic-tac-toe | 32 | Q final-move VNM, direct breaks ties minus Q final-move VNM, fitted continuations | match | 0.38 ± 0.23 | [0.30, 0.47] | 1.72e-10 | 3.75e-08 |
| Tic-tac-toe | 32 | Q exact-frequency final-move VNM, exact direct breaks ties (privileged) minus Q final-move VNM, exact frequencies and original continuations | match | 35.24 ± 6.33 | [32.96, 37.52] | 4.22e-25 | 1.18e-22 |
| Connect Four | 24 | Action preferences plus game consequences minus Action preferences alone | match | 8.72 ± 9.22 | [4.83, 12.62] | 0.000115 | 0.0184 |
| Connect Four | 24 | Action preferences plus game consequences minus Action preferences alone | win | 6.87 ± 9.81 | [2.72, 11.01] | 0.00229 | 0.312 |
| Connect Four | 24 | Action preferences plus game consequences minus Action preferences alone | draw | 0.63 ± 2.16 | [-0.28, 1.55] | 0.164 | 1 |
| Connect Four | 24 | Action preferences plus game consequences minus Action preferences alone | loss | -7.50 ± 10.92 | [-12.11, -2.89] | 0.00266 | 0.356 |
| Connect Four | 24 | Action preferences plus game consequences minus Action preferences alone | points | 0.14 ± 0.21 | [0.06, 0.23] | 0.0024 | 0.324 |
| Connect Four | 24 | Action preferences plus game consequences minus Q direct | match | 4.23 ± 3.99 | [2.55, 5.92] | 2.9e-05 | 0.00497 |
| Connect Four | 24 | Action preferences plus game consequences minus Q direct | win | -1.16 ± 6.25 | [-3.80, 1.48] | 0.375 | 1 |
| Connect Four | 24 | Action preferences plus game consequences minus Q direct | draw | -0.41 ± 1.87 | [-1.20, 0.38] | 0.297 | 1 |
| Connect Four | 24 | Action preferences plus game consequences minus Q direct | loss | 1.56 ± 6.09 | [-1.01, 4.13] | 0.221 | 1 |
| Connect Four | 24 | Action preferences plus game consequences minus Q direct | points | -0.03 ± 0.12 | [-0.08, 0.02] | 0.286 | 1 |
| Connect Four | 24 | Action preferences plus game consequences minus Q direct | multi_match | 4.23 ± 3.99 | [2.55, 5.92] | 2.89e-05 | 0.00497 |
| Connect Four | 24 | Game consequences, signed local shortfalls VNM minus Action preferences plus game consequences | match | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Connect Four | 24 | Game consequences, signed local shortfalls VNM minus Action preferences plus game consequences | win | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Connect Four | 24 | Game consequences, signed local shortfalls VNM minus Action preferences plus game consequences | draw | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Connect Four | 24 | Game consequences, signed local shortfalls VNM minus Action preferences plus game consequences | loss | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Connect Four | 24 | Game consequences, signed local shortfalls VNM minus Action preferences plus game consequences | points | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Connect Four | 24 | V direct, before bounding scores minus V direct | match | 0.59 ± 0.51 | [0.37, 0.80] | 9.27e-06 | 0.00172 |
| Connect Four | 24 | V final-move VNM, fitted continuations minus V direct | match | -53.08 ± 5.11 | [-55.24, -50.92] | 3.89e-25 | 1.1e-22 |
| Connect Four | 24 | V final-move VNM, fitted continuations minus V direct | win | 9.73 ± 8.63 | [6.09, 13.38] | 1.28e-05 | 0.00236 |
| Connect Four | 24 | V final-move VNM, fitted continuations minus V direct | draw | -0.05 ± 0.74 | [-0.36, 0.26] | 0.75 | 1 |
| Connect Four | 24 | V final-move VNM, fitted continuations minus V direct | loss | -9.68 ± 8.79 | [-13.40, -5.97] | 1.75e-05 | 0.00304 |
| Connect Four | 24 | V final-move VNM, fitted continuations minus V direct | points | 0.19 ± 0.17 | [0.12, 0.27] | 1.48e-05 | 0.00264 |
| Connect Four | 24 | V signed local shortfalls VNM, fitted continuations minus V direct | match | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Connect Four | 24 | V signed local shortfalls VNM, fitted continuations minus V direct | win | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Connect Four | 24 | V signed local shortfalls VNM, fitted continuations minus V direct | draw | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Connect Four | 24 | V signed local shortfalls VNM, fitted continuations minus V direct | loss | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Connect Four | 24 | V signed local shortfalls VNM, fitted continuations minus V direct | points | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Connect Four | 24 | V signed local shortfalls VNM, original continuations minus V direct | match | -22.96 ± 5.19 | [-25.15, -20.77] | 8.22e-17 | 2.05e-14 |
| Connect Four | 24 | V direct, added visits minus V direct | match | 4.34 ± 2.58 | [3.25, 5.43] | 2.6e-08 | 5.39e-06 |
| Connect Four | 24 | V direct, added visits minus V direct | win | 7.13 ± 9.97 | [2.92, 11.34] | 0.00192 | 0.27 |
| Connect Four | 24 | V direct, added visits minus V direct | draw | -0.10 ± 0.87 | [-0.47, 0.27] | 0.588 | 1 |
| Connect Four | 24 | V direct, added visits minus V direct | loss | -7.03 ± 10.01 | [-11.26, -2.81] | 0.00222 | 0.306 |
| Connect Four | 24 | V direct, added visits minus V direct | points | 0.14 ± 0.20 | [0.06, 0.23] | 0.00205 | 0.285 |
| Connect Four | 24 | Exact original frequency, direct (privileged) minus V direct | match | 23.11 ± 4.43 | [21.24, 24.98] | 2.16e-18 | 5.45e-16 |
| Connect Four | 24 | Mean exact incoming frequency, direct V (privileged) minus V direct | match | 23.11 ± 4.43 | [21.24, 24.98] | 2.16e-18 | 5.45e-16 |
| Connect Four | 24 | V final-move VNM, exact frequencies and original continuations minus Exact original frequency, direct (privileged) | match | -78.71 ± 13.06 | [-84.23, -73.20] | 8.66e-20 | 2.22e-17 |
| Connect Four | 24 | V final-move VNM, exact frequencies and fitted continuations minus V final-move VNM, fitted continuations | match | -2.58 ± 12.31 | [-7.78, 2.62] | 0.315 | 1 |
| Connect Four | 24 | V final-move VNM, fitted scores and original continuations minus V final-move VNM, fitted continuations | match | 2.08 ± 3.23 | [0.71, 3.44] | 0.00452 | 0.584 |
| Connect Four | 24 | V final-move VNM, exact frequencies and original continuations minus V final-move VNM, exact frequencies and fitted continuations | match | 0.07 ± 2.77 | [-1.11, 1.24] | 0.909 | 1 |
| Connect Four | 24 | V final-move VNM, exact frequencies and original continuations minus V final-move VNM, fitted scores and original continuations | match | -4.60 ± 13.46 | [-10.28, 1.09] | 0.108 | 1 |
| Connect Four | 24 | V signed local shortfalls VNM, fitted continuations minus V final-move VNM, fitted continuations | match | 53.08 ± 5.11 | [50.92, 55.24] | 3.89e-25 | 1.1e-22 |
| Connect Four | 24 | V signed local shortfalls VNM, fitted continuations minus V final-move VNM, fitted continuations | win | -9.73 ± 8.63 | [-13.38, -6.09] | 1.28e-05 | 0.00236 |
| Connect Four | 24 | V signed local shortfalls VNM, fitted continuations minus V final-move VNM, fitted continuations | draw | 0.05 ± 0.74 | [-0.26, 0.36] | 0.75 | 1 |
| Connect Four | 24 | V signed local shortfalls VNM, fitted continuations minus V final-move VNM, fitted continuations | loss | 9.68 ± 8.79 | [5.97, 13.40] | 1.75e-05 | 0.00304 |
| Connect Four | 24 | V signed local shortfalls VNM, fitted continuations minus V final-move VNM, fitted continuations | points | -0.19 ± 0.17 | [-0.27, -0.12] | 1.48e-05 | 0.00264 |
| Connect Four | 24 | V final frequency with last mover's sign removed minus V final-move VNM, fitted continuations | match | -4.81 ± 4.93 | [-6.89, -2.73] | 8.09e-05 | 0.0131 |
| Connect Four | 24 | V final-move VNM, direct breaks ties minus V final-move VNM, fitted continuations | match | 2.75 ± 2.40 | [1.74, 3.77] | 9.91e-06 | 0.00183 |
| Connect Four | 24 | V exact-frequency final-move VNM, exact direct breaks ties (privileged) minus V final-move VNM, exact frequencies and original continuations | match | 44.44 ± 14.20 | [38.44, 50.44] | 1.44e-13 | 3.38e-11 |
| Connect Four | 24 | Q direct, before bounding scores minus Q direct | match | 0.07 ± 0.22 | [-0.02, 0.16] | 0.126 | 1 |
| Connect Four | 24 | Q final-move VNM, fitted continuations minus Q direct | match | -68.12 ± 6.95 | [-71.06, -65.19] | 1.42e-24 | 3.89e-22 |
| Connect Four | 24 | Q final-move VNM, fitted continuations minus Q direct | win | -8.09 ± 9.55 | [-12.12, -4.05] | 0.00039 | 0.0573 |
| Connect Four | 24 | Q final-move VNM, fitted continuations minus Q direct | draw | -0.63 ± 2.04 | [-1.50, 0.23] | 0.141 | 1 |
| Connect Four | 24 | Q final-move VNM, fitted continuations minus Q direct | loss | 8.72 ± 9.27 | [4.81, 12.64] | 0.000122 | 0.0194 |
| Connect Four | 24 | Q final-move VNM, fitted continuations minus Q direct | points | -0.17 ± 0.19 | [-0.25, -0.09] | 0.000207 | 0.0324 |
| Connect Four | 24 | Q final-move VNM, fitted continuations minus Q direct | multi_match | -68.18 ± 6.97 | [-71.12, -65.24] | 1.48e-24 | 4.03e-22 |
| Connect Four | 24 | Q signed local shortfalls VNM, fitted continuations minus Q direct | match | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Connect Four | 24 | Q signed local shortfalls VNM, fitted continuations minus Q direct | win | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Connect Four | 24 | Q signed local shortfalls VNM, fitted continuations minus Q direct | draw | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Connect Four | 24 | Q signed local shortfalls VNM, fitted continuations minus Q direct | loss | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Connect Four | 24 | Q signed local shortfalls VNM, fitted continuations minus Q direct | points | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| Connect Four | 24 | Q signed local shortfalls VNM, original continuations minus Q direct | match | -8.54 ± 3.35 | [-9.96, -7.13] | 1.01e-11 | 2.3e-09 |
| Connect Four | 24 | Q direct, added visits minus Q direct | match | 2.53 ± 2.25 | [1.58, 3.48] | 1.34e-05 | 0.00243 |
| Connect Four | 24 | Q direct, added visits minus Q direct | win | -0.62 ± 3.51 | [-2.10, 0.87] | 0.398 | 1 |
| Connect Four | 24 | Q direct, added visits minus Q direct | draw | -0.52 ± 1.96 | [-1.35, 0.31] | 0.207 | 1 |
| Connect Four | 24 | Q direct, added visits minus Q direct | loss | 1.14 ± 3.11 | [-0.17, 2.45] | 0.0854 | 1 |
| Connect Four | 24 | Q direct, added visits minus Q direct | points | -0.02 ± 0.06 | [-0.04, 0.01] | 0.187 | 1 |
| Connect Four | 24 | Exact original frequency, direct (privileged) minus Q direct | match | 7.02 ± 6.24 | [4.39, 9.65] | 1.32e-05 | 0.00239 |
| Connect Four | 24 | Q final-move VNM, exact frequencies and original continuations minus Exact original frequency, direct (privileged) | match | -78.71 ± 13.06 | [-84.23, -73.20] | 8.66e-20 | 2.22e-17 |
| Connect Four | 24 | Q final-move VNM, exact frequencies and fitted continuations minus Q final-move VNM, fitted continuations | match | -2.54 ± 9.88 | [-6.71, 1.63] | 0.221 | 1 |
| Connect Four | 24 | Q final-move VNM, fitted scores and original continuations minus Q final-move VNM, fitted continuations | match | 1.13 ± 2.84 | [-0.07, 2.32] | 0.0642 | 1 |
| Connect Four | 24 | Q final-move VNM, exact frequencies and original continuations minus Q final-move VNM, exact frequencies and fitted continuations | match | -1.03 ± 2.77 | [-2.20, 0.14] | 0.0817 | 1 |
| Connect Four | 24 | Q final-move VNM, exact frequencies and original continuations minus Q final-move VNM, fitted scores and original continuations | match | -4.69 ± 12.83 | [-10.11, 0.72] | 0.0861 | 1 |
| Connect Four | 24 | Q signed local shortfalls VNM, fitted continuations minus Q final-move VNM, fitted continuations | match | 68.12 ± 6.95 | [65.19, 71.06] | 1.42e-24 | 3.89e-22 |
| Connect Four | 24 | Q signed local shortfalls VNM, fitted continuations minus Q final-move VNM, fitted continuations | win | 8.09 ± 9.55 | [4.05, 12.12] | 0.00039 | 0.0573 |
| Connect Four | 24 | Q signed local shortfalls VNM, fitted continuations minus Q final-move VNM, fitted continuations | draw | 0.63 ± 2.04 | [-0.23, 1.50] | 0.141 | 1 |
| Connect Four | 24 | Q signed local shortfalls VNM, fitted continuations minus Q final-move VNM, fitted continuations | loss | -8.72 ± 9.27 | [-12.64, -4.81] | 0.000122 | 0.0194 |
| Connect Four | 24 | Q signed local shortfalls VNM, fitted continuations minus Q final-move VNM, fitted continuations | points | 0.17 ± 0.19 | [0.09, 0.25] | 0.000207 | 0.0324 |
| Connect Four | 24 | Q final frequency with last mover's sign removed minus Q final-move VNM, fitted continuations | match | -5.89 ± 5.56 | [-8.24, -3.54] | 2.92e-05 | 0.00497 |
| Connect Four | 24 | Q final-move VNM, direct breaks ties minus Q final-move VNM, fitted continuations | match | 5.10 ± 5.27 | [2.87, 7.32] | 8.99e-05 | 0.0145 |
| Connect Four | 24 | Q exact-frequency final-move VNM, exact direct breaks ties (privileged) minus Q final-move VNM, exact frequencies and original continuations | match | 44.44 ± 14.20 | [38.44, 50.44] | 1.44e-13 | 3.38e-11 |
| Connect Four | 24 | Q normalized fit, original visits minus Q direct | match | 0.60 ± 2.70 | [-0.54, 1.74] | 0.289 | 1 |
| Connect Four | 24 | Q normalized fit, original visits minus Q direct | win | -1.38 ± 5.58 | [-3.74, 0.97] | 0.237 | 1 |
| Connect Four | 24 | Q normalized fit, original visits minus Q direct | draw | 0.02 ± 1.27 | [-0.52, 0.55] | 0.951 | 1 |
| Connect Four | 24 | Q normalized fit, original visits minus Q direct | loss | 1.37 ± 5.47 | [-0.94, 3.68] | 0.233 | 1 |
| Connect Four | 24 | Q normalized fit, original visits minus Q direct | points | -0.03 ± 0.11 | [-0.07, 0.02] | 0.232 | 1 |
| Connect Four | 24 | Q normalized fit, added visits minus Q direct, added visits | match | 2.71 ± 3.20 | [1.36, 4.07] | 0.000387 | 0.0572 |
| Connect Four | 24 | Q normalized fit, added visits minus Q direct, added visits | win | 0.88 ± 2.98 | [-0.38, 2.14] | 0.161 | 1 |
| Connect Four | 24 | Q normalized fit, added visits minus Q direct, added visits | draw | 0.13 ± 1.64 | [-0.56, 0.82] | 0.7 | 1 |
| Connect Four | 24 | Q normalized fit, added visits minus Q direct, added visits | loss | -1.01 ± 2.72 | [-2.16, 0.14] | 0.082 | 1 |
| Connect Four | 24 | Q normalized fit, added visits minus Q direct, added visits | points | 0.02 ± 0.05 | [-0.00, 0.04] | 0.104 | 1 |
| Connect Four | 24 | Q normalized fit, added visits minus Q normalized fit, original visits | match | 4.65 ± 3.23 | [3.28, 6.01] | 3.48e-07 | 6.83e-05 |
| Connect Four | 24 | Q normalized fit, added visits minus Q normalized fit, original visits | win | 1.64 ± 5.34 | [-0.61, 3.90] | 0.145 | 1 |
| Connect Four | 24 | Q normalized fit, added visits minus Q normalized fit, original visits | draw | -0.41 ± 1.23 | [-0.93, 0.11] | 0.119 | 1 |
| Connect Four | 24 | Q normalized fit, added visits minus Q normalized fit, original visits | loss | -1.24 ± 5.87 | [-3.71, 1.24] | 0.312 | 1 |
| Connect Four | 24 | Q normalized fit, added visits minus Q normalized fit, original visits | points | 0.03 ± 0.11 | [-0.02, 0.08] | 0.218 | 1 |
| Connect Four | 24 | Q normalized fit, added visits minus Q direct | match | 5.25 ± 5.12 | [3.09, 7.41] | 4.39e-05 | 0.00733 |
| Connect Four | 24 | Q normalized fit, added visits minus Q direct | win | 0.26 ± 3.33 | [-1.14, 1.66] | 0.705 | 1 |
| Connect Four | 24 | Q normalized fit, added visits minus Q direct | draw | -0.39 ± 1.08 | [-0.85, 0.07] | 0.0898 | 1 |
| Connect Four | 24 | Q normalized fit, added visits minus Q direct | loss | 0.13 ± 3.33 | [-1.27, 1.53] | 0.85 | 1 |
| Connect Four | 24 | Q normalized fit, added visits minus Q direct | points | 0.00 ± 0.07 | [-0.03, 0.03] | 0.923 | 1 |
| 4×4 Reversi | 24 | Action preferences plus game consequences minus Action preferences alone | match | 7.78 ± 1.01 | [7.35, 8.20] | 3.37e-22 | 9.02e-20 |
| 4×4 Reversi | 24 | Action preferences plus game consequences minus Action preferences alone | win | 20.61 ± 3.19 | [19.26, 21.95] | 1.88e-20 | 4.9e-18 |
| 4×4 Reversi | 24 | Action preferences plus game consequences minus Action preferences alone | draw | -4.20 ± 2.12 | [-5.09, -3.30] | 1.33e-09 | 2.83e-07 |
| 4×4 Reversi | 24 | Action preferences plus game consequences minus Action preferences alone | loss | -16.41 ± 2.82 | [-17.60, -15.21] | 1.93e-19 | 4.89e-17 |
| 4×4 Reversi | 24 | Action preferences plus game consequences minus Action preferences alone | points | 0.37 ± 0.06 | [0.35, 0.39] | 1.29e-20 | 3.39e-18 |
| 4×4 Reversi | 24 | Action preferences plus game consequences minus Q direct | match | 13.98 ± 1.33 | [13.42, 14.54] | 2.75e-25 | 7.78e-23 |
| 4×4 Reversi | 24 | Action preferences plus game consequences minus Q direct | win | 17.42 ± 4.67 | [15.44, 19.39] | 3.47e-15 | 8.47e-13 |
| 4×4 Reversi | 24 | Action preferences plus game consequences minus Q direct | draw | -4.10 ± 2.57 | [-5.19, -3.02] | 6.48e-08 | 1.31e-05 |
| 4×4 Reversi | 24 | Action preferences plus game consequences minus Q direct | loss | -13.31 ± 3.59 | [-14.83, -11.80] | 3.82e-15 | 9.29e-13 |
| 4×4 Reversi | 24 | Action preferences plus game consequences minus Q direct | points | 0.31 ± 0.08 | [0.27, 0.34] | 1.47e-15 | 3.61e-13 |
| 4×4 Reversi | 24 | Action preferences plus game consequences minus Q direct | multi_match | 35.19 ± 3.28 | [33.80, 36.57] | 1.84e-25 | 5.22e-23 |
| 4×4 Reversi | 24 | Game consequences, signed local shortfalls VNM minus Action preferences plus game consequences | match | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| 4×4 Reversi | 24 | Game consequences, signed local shortfalls VNM minus Action preferences plus game consequences | win | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| 4×4 Reversi | 24 | Game consequences, signed local shortfalls VNM minus Action preferences plus game consequences | draw | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| 4×4 Reversi | 24 | Game consequences, signed local shortfalls VNM minus Action preferences plus game consequences | loss | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| 4×4 Reversi | 24 | Game consequences, signed local shortfalls VNM minus Action preferences plus game consequences | points | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| 4×4 Reversi | 24 | Game outcome plus elicited ordering, low-index fallback minus Game outcome plus elicited ordering, high-index fallback | match | 0.05 ± 0.11 | [0.00, 0.09] | 0.0498 | 1 |
| 4×4 Reversi | 24 | Game outcome plus elicited ordering, low-index fallback minus Game outcome plus elicited ordering, high-index fallback | win | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| 4×4 Reversi | 24 | Game outcome plus elicited ordering, low-index fallback minus Game outcome plus elicited ordering, high-index fallback | draw | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| 4×4 Reversi | 24 | Game outcome plus elicited ordering, low-index fallback minus Game outcome plus elicited ordering, high-index fallback | loss | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| 4×4 Reversi | 24 | Game outcome plus elicited ordering, low-index fallback minus Game outcome plus elicited ordering, high-index fallback | points | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| 4×4 Reversi | 24 | Game outcome plus elicited ordering, low-index fallback minus Action preferences plus game consequences | match | 0.16 ± 0.20 | [0.08, 0.24] | 0.000513 | 0.0744 |
| 4×4 Reversi | 24 | Game outcome plus elicited ordering, low-index fallback minus Action preferences plus game consequences | win | 0.07 ± 0.15 | [0.00, 0.13] | 0.0428 | 1 |
| 4×4 Reversi | 24 | Game outcome plus elicited ordering, low-index fallback minus Action preferences plus game consequences | draw | -0.02 ± 0.18 | [-0.09, 0.06] | 0.664 | 1 |
| 4×4 Reversi | 24 | Game outcome plus elicited ordering, low-index fallback minus Action preferences plus game consequences | loss | -0.05 ± 0.13 | [-0.10, 0.01] | 0.083 | 1 |
| 4×4 Reversi | 24 | Game outcome plus elicited ordering, low-index fallback minus Action preferences plus game consequences | points | 0.00 ± 0.00 | [0.00, 0.00] | 0.0161 | 1 |
| 4×4 Reversi | 24 | V direct, before bounding scores minus V direct | match | -0.02 ± 0.07 | [-0.05, 0.01] | 0.129 | 1 |
| 4×4 Reversi | 24 | V final-move VNM, fitted continuations minus V direct | match | -1.20 ± 2.14 | [-2.11, -0.30] | 0.0115 | 1 |
| 4×4 Reversi | 24 | V final-move VNM, fitted continuations minus V direct | win | -11.49 ± 6.88 | [-14.40, -8.59] | 2.89e-08 | 5.95e-06 |
| 4×4 Reversi | 24 | V final-move VNM, fitted continuations minus V direct | draw | 4.98 ± 3.26 | [3.60, 6.36] | 1.34e-07 | 2.67e-05 |
| 4×4 Reversi | 24 | V final-move VNM, fitted continuations minus V direct | loss | 6.51 ± 6.45 | [3.79, 9.23] | 5.34e-05 | 0.00886 |
| 4×4 Reversi | 24 | V final-move VNM, fitted continuations minus V direct | points | -0.18 ± 0.13 | [-0.23, -0.13] | 5.89e-07 | 0.000114 |
| 4×4 Reversi | 24 | V signed local shortfalls VNM, fitted continuations minus V direct | match | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| 4×4 Reversi | 24 | V signed local shortfalls VNM, fitted continuations minus V direct | win | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| 4×4 Reversi | 24 | V signed local shortfalls VNM, fitted continuations minus V direct | draw | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| 4×4 Reversi | 24 | V signed local shortfalls VNM, fitted continuations minus V direct | loss | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| 4×4 Reversi | 24 | V signed local shortfalls VNM, fitted continuations minus V direct | points | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| 4×4 Reversi | 24 | V signed local shortfalls VNM, original continuations minus V direct | match | -1.82 ± 0.79 | [-2.15, -1.48] | 8.55e-11 | 1.89e-08 |
| 4×4 Reversi | 24 | V direct, added visits minus V direct | match | 4.75 ± 1.51 | [4.12, 5.39] | 1.22e-13 | 2.87e-11 |
| 4×4 Reversi | 24 | V direct, added visits minus V direct | win | 1.64 ± 5.16 | [-0.53, 3.82] | 0.132 | 1 |
| 4×4 Reversi | 24 | V direct, added visits minus V direct | draw | -0.80 ± 2.61 | [-1.90, 0.31] | 0.148 | 1 |
| 4×4 Reversi | 24 | V direct, added visits minus V direct | loss | -0.85 ± 3.44 | [-2.30, 0.61] | 0.241 | 1 |
| 4×4 Reversi | 24 | V direct, added visits minus V direct | points | 0.02 ± 0.08 | [-0.01, 0.06] | 0.159 | 1 |
| 4×4 Reversi | 24 | Exact original frequency, direct (privileged) minus V direct | match | 16.09 ± 1.49 | [15.46, 16.72] | 1.56e-25 | 4.44e-23 |
| 4×4 Reversi | 24 | Mean exact incoming frequency, direct V (privileged) minus V direct | match | 15.78 ± 1.56 | [15.12, 16.44] | 7.26e-25 | 2e-22 |
| 4×4 Reversi | 24 | V final-move VNM, exact frequencies and original continuations minus Exact original frequency, direct (privileged) | match | -7.71 ± 1.05 | [-8.16, -7.27] | 9.27e-22 | 2.48e-19 |
| 4×4 Reversi | 24 | V final-move VNM, exact frequencies and fitted continuations minus V final-move VNM, fitted continuations | match | 7.92 ± 1.35 | [7.35, 8.49] | 1.64e-19 | 4.17e-17 |
| 4×4 Reversi | 24 | V final-move VNM, fitted scores and original continuations minus V final-move VNM, fitted continuations | match | 0.75 ± 1.04 | [0.30, 1.19] | 0.00192 | 0.27 |
| 4×4 Reversi | 24 | V final-move VNM, exact frequencies and original continuations minus V final-move VNM, exact frequencies and fitted continuations | match | 1.66 ± 0.74 | [1.35, 1.97] | 1.14e-10 | 2.5e-08 |
| 4×4 Reversi | 24 | V final-move VNM, exact frequencies and original continuations minus V final-move VNM, fitted scores and original continuations | match | 8.84 ± 1.14 | [8.35, 9.32] | 3.04e-22 | 8.17e-20 |
| 4×4 Reversi | 24 | V signed local shortfalls VNM, fitted continuations minus V final-move VNM, fitted continuations | match | 1.20 ± 2.14 | [0.30, 2.11] | 0.0115 | 1 |
| 4×4 Reversi | 24 | V signed local shortfalls VNM, fitted continuations minus V final-move VNM, fitted continuations | win | 11.49 ± 6.88 | [8.59, 14.40] | 2.89e-08 | 5.95e-06 |
| 4×4 Reversi | 24 | V signed local shortfalls VNM, fitted continuations minus V final-move VNM, fitted continuations | draw | -4.98 ± 3.26 | [-6.36, -3.60] | 1.34e-07 | 2.67e-05 |
| 4×4 Reversi | 24 | V signed local shortfalls VNM, fitted continuations minus V final-move VNM, fitted continuations | loss | -6.51 ± 6.45 | [-9.23, -3.79] | 5.34e-05 | 0.00886 |
| 4×4 Reversi | 24 | V signed local shortfalls VNM, fitted continuations minus V final-move VNM, fitted continuations | points | 0.18 ± 0.13 | [0.13, 0.23] | 5.89e-07 | 0.000114 |
| 4×4 Reversi | 24 | V final frequency with last mover's sign removed minus V final-move VNM, fitted continuations | match | -3.01 ± 1.67 | [-3.72, -2.31] | 7.04e-09 | 1.47e-06 |
| 4×4 Reversi | 24 | V final-move VNM, direct breaks ties minus V final-move VNM, fitted continuations | match | -1.86 ± 0.55 | [-2.09, -1.63] | 2.58e-14 | 6.16e-12 |
| 4×4 Reversi | 24 | V exact-frequency final-move VNM, exact direct breaks ties (privileged) minus V final-move VNM, exact frequencies and original continuations | match | 2.91 ± 0.48 | [2.71, 3.11] | 7.74e-20 | 2e-17 |
| 4×4 Reversi | 24 | Q direct, before bounding scores minus Q direct | match | -0.01 ± 0.02 | [-0.02, 0.00] | 0.162 | 1 |
| 4×4 Reversi | 24 | Q final-move VNM, fitted continuations minus Q direct | match | -5.97 ± 1.92 | [-6.78, -5.16] | 1.61e-13 | 3.73e-11 |
| 4×4 Reversi | 24 | Q final-move VNM, fitted continuations minus Q direct | win | -13.00 ± 6.43 | [-15.72, -10.29] | 9.16e-10 | 1.97e-07 |
| 4×4 Reversi | 24 | Q final-move VNM, fitted continuations minus Q direct | draw | 4.22 ± 4.75 | [2.21, 6.22] | 0.000236 | 0.036 |
| 4×4 Reversi | 24 | Q final-move VNM, fitted continuations minus Q direct | loss | 8.79 ± 4.58 | [6.85, 10.72] | 2.46e-09 | 5.2e-07 |
| 4×4 Reversi | 24 | Q final-move VNM, fitted continuations minus Q direct | points | -0.22 ± 0.10 | [-0.26, -0.18] | 2.71e-10 | 5.89e-08 |
| 4×4 Reversi | 24 | Q final-move VNM, fitted continuations minus Q direct | multi_match | -15.02 ± 4.72 | [-17.01, -13.03] | 1e-13 | 2.36e-11 |
| 4×4 Reversi | 24 | Q signed local shortfalls VNM, fitted continuations minus Q direct | match | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| 4×4 Reversi | 24 | Q signed local shortfalls VNM, fitted continuations minus Q direct | win | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| 4×4 Reversi | 24 | Q signed local shortfalls VNM, fitted continuations minus Q direct | draw | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| 4×4 Reversi | 24 | Q signed local shortfalls VNM, fitted continuations minus Q direct | loss | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| 4×4 Reversi | 24 | Q signed local shortfalls VNM, fitted continuations minus Q direct | points | 0.00 ± 0.00 | [0.00, 0.00] | — | — |
| 4×4 Reversi | 24 | Q signed local shortfalls VNM, original continuations minus Q direct | match | -1.61 ± 0.61 | [-1.87, -1.35] | 5.17e-12 | 1.18e-09 |
| 4×4 Reversi | 24 | Q direct, added visits minus Q direct | match | 4.80 ± 1.09 | [4.34, 5.26] | 1.01e-16 | 2.5e-14 |
| 4×4 Reversi | 24 | Q direct, added visits minus Q direct | win | -2.69 ± 4.10 | [-4.42, -0.95] | 0.00389 | 0.505 |
| 4×4 Reversi | 24 | Q direct, added visits minus Q direct | draw | 0.36 ± 2.58 | [-0.73, 1.45] | 0.503 | 1 |
| 4×4 Reversi | 24 | Q direct, added visits minus Q direct | loss | 2.33 ± 3.47 | [0.86, 3.79] | 0.00321 | 0.427 |
| 4×4 Reversi | 24 | Q direct, added visits minus Q direct | points | -0.05 ± 0.07 | [-0.08, -0.02] | 0.00224 | 0.306 |
| 4×4 Reversi | 24 | Exact original frequency, direct (privileged) minus Q direct | match | 14.14 ± 1.26 | [13.61, 14.67] | 6.35e-26 | 1.82e-23 |
| 4×4 Reversi | 24 | Q final-move VNM, exact frequencies and original continuations minus Exact original frequency, direct (privileged) | match | -7.71 ± 1.05 | [-8.16, -7.27] | 9.27e-22 | 2.48e-19 |
| 4×4 Reversi | 24 | Q final-move VNM, exact frequencies and fitted continuations minus Q final-move VNM, fitted continuations | match | 10.70 ± 1.65 | [10.00, 11.40] | 1.66e-20 | 4.33e-18 |
| 4×4 Reversi | 24 | Q final-move VNM, fitted scores and original continuations minus Q final-move VNM, fitted continuations | match | 0.89 ± 0.69 | [0.60, 1.18] | 1.87e-06 | 0.000355 |
| 4×4 Reversi | 24 | Q final-move VNM, exact frequencies and original continuations minus Q final-move VNM, exact frequencies and fitted continuations | match | 1.69 ± 0.71 | [1.39, 1.99] | 4.09e-11 | 9.08e-09 |
| 4×4 Reversi | 24 | Q final-move VNM, exact frequencies and original continuations minus Q final-move VNM, fitted scores and original continuations | match | 11.51 ± 1.43 | [10.90, 12.11] | 1.35e-22 | 3.64e-20 |
| 4×4 Reversi | 24 | Q signed local shortfalls VNM, fitted continuations minus Q final-move VNM, fitted continuations | match | 5.97 ± 1.92 | [5.16, 6.78] | 1.61e-13 | 3.73e-11 |
| 4×4 Reversi | 24 | Q signed local shortfalls VNM, fitted continuations minus Q final-move VNM, fitted continuations | win | 13.00 ± 6.43 | [10.29, 15.72] | 9.16e-10 | 1.97e-07 |
| 4×4 Reversi | 24 | Q signed local shortfalls VNM, fitted continuations minus Q final-move VNM, fitted continuations | draw | -4.22 ± 4.75 | [-6.22, -2.21] | 0.000236 | 0.036 |
| 4×4 Reversi | 24 | Q signed local shortfalls VNM, fitted continuations minus Q final-move VNM, fitted continuations | loss | -8.79 ± 4.58 | [-10.72, -6.85] | 2.46e-09 | 5.2e-07 |
| 4×4 Reversi | 24 | Q signed local shortfalls VNM, fitted continuations minus Q final-move VNM, fitted continuations | points | 0.22 ± 0.10 | [0.18, 0.26] | 2.71e-10 | 5.89e-08 |
| 4×4 Reversi | 24 | Q final frequency with last mover's sign removed minus Q final-move VNM, fitted continuations | match | -3.21 ± 1.33 | [-3.77, -2.65] | 3.13e-11 | 6.99e-09 |
| 4×4 Reversi | 24 | Q final-move VNM, direct breaks ties minus Q final-move VNM, fitted continuations | match | -0.31 ± 0.34 | [-0.45, -0.16] | 0.000206 | 0.0324 |
| 4×4 Reversi | 24 | Q exact-frequency final-move VNM, exact direct breaks ties (privileged) minus Q final-move VNM, exact frequencies and original continuations | match | 2.91 ± 0.48 | [2.71, 3.11] | 7.74e-20 | 2e-17 |
| 4×4 Reversi | 24 | Q normalized fit, original visits minus Q direct | match | 2.03 ± 1.29 | [1.48, 2.58] | 8.45e-08 | 1.71e-05 |
| 4×4 Reversi | 24 | Q normalized fit, original visits minus Q direct | win | 4.88 ± 3.44 | [3.43, 6.34] | 4.39e-07 | 8.57e-05 |
| 4×4 Reversi | 24 | Q normalized fit, original visits minus Q direct | draw | -1.61 ± 2.55 | [-2.69, -0.53] | 0.00515 | 0.659 |
| 4×4 Reversi | 24 | Q normalized fit, original visits minus Q direct | loss | -3.27 ± 2.66 | [-4.39, -2.15] | 3.75e-06 | 0.000704 |
| 4×4 Reversi | 24 | Q normalized fit, original visits minus Q direct | points | 0.08 ± 0.06 | [0.06, 0.11] | 2.84e-07 | 5.6e-05 |
| 4×4 Reversi | 24 | Q normalized fit, added visits minus Q direct, added visits | match | 0.30 ± 0.79 | [-0.03, 0.63] | 0.0764 | 1 |
| 4×4 Reversi | 24 | Q normalized fit, added visits minus Q direct, added visits | win | 9.29 ± 3.85 | [7.67, 10.92] | 2.9e-11 | 6.53e-09 |
| 4×4 Reversi | 24 | Q normalized fit, added visits minus Q direct, added visits | draw | -2.36 ± 1.33 | [-2.92, -1.80] | 9.47e-09 | 1.97e-06 |
| 4×4 Reversi | 24 | Q normalized fit, added visits minus Q direct, added visits | loss | -6.93 ± 3.61 | [-8.46, -5.41] | 2.43e-09 | 5.15e-07 |
| 4×4 Reversi | 24 | Q normalized fit, added visits minus Q direct, added visits | points | 0.16 ± 0.07 | [0.13, 0.19] | 1.68e-10 | 3.69e-08 |
| 4×4 Reversi | 24 | Q normalized fit, added visits minus Q normalized fit, original visits | match | 3.07 ± 1.25 | [2.54, 3.60] | 2.06e-11 | 4.65e-09 |
| 4×4 Reversi | 24 | Q normalized fit, added visits minus Q normalized fit, original visits | win | 1.73 ± 3.84 | [0.10, 3.35] | 0.0381 | 1 |
| 4×4 Reversi | 24 | Q normalized fit, added visits minus Q normalized fit, original visits | draw | -0.39 ± 2.04 | [-1.25, 0.47] | 0.357 | 1 |
| 4×4 Reversi | 24 | Q normalized fit, added visits minus Q normalized fit, original visits | loss | -1.33 ± 3.34 | [-2.74, 0.08] | 0.0624 | 1 |
| 4×4 Reversi | 24 | Q normalized fit, added visits minus Q normalized fit, original visits | points | 0.03 ± 0.07 | [0.00, 0.06] | 0.0404 | 1 |
| 4×4 Reversi | 24 | Q normalized fit, added visits minus Q direct | match | 5.10 ± 1.45 | [4.49, 5.71] | 1.28e-14 | 3.1e-12 |
| 4×4 Reversi | 24 | Q normalized fit, added visits minus Q direct | win | 6.61 ± 4.67 | [4.64, 8.58] | 4.58e-07 | 8.88e-05 |
| 4×4 Reversi | 24 | Q normalized fit, added visits minus Q direct | draw | -2.00 ± 2.63 | [-3.11, -0.89] | 0.0011 | 0.156 |
| 4×4 Reversi | 24 | Q normalized fit, added visits minus Q direct | loss | -4.61 ± 3.88 | [-6.25, -2.97] | 6.41e-06 | 0.0012 |
| 4×4 Reversi | 24 | Q normalized fit, added visits minus Q direct | points | 0.11 ± 0.08 | [0.08, 0.15] | 7.48e-07 | 0.000143 |
