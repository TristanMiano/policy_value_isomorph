# Tic-tac-toe reconstruction pilot

Completed November 17, 2026. With perfect return estimates and the registered tie rule, the recovered player copied every minimax move, 75.9% of heuristic moves, 56.6% of moves by the frozen neural policy, and 37.7% of moves by the deliberately fallible policy. The learned successor-state value network missed the registered fidelity threshold for all five sources under both evaluation distributions. [Frozen protocol](experiments/tictactoe_pilot/manifest.json) · [Fidelity by method and population](experiments/tictactoe_pilot/results/fidelity.csv)

## Checkpoint decisions

The exact comparison uses all 4,520 reachable decision boards. The learned comparison uses 780 held-out decision boards. “Uniform” gives every board equal weight. “Visited” weights boards by their probability of being visited during source self-play and normalizes within the evaluated population. For deterministic sources the endpoint is move agreement, with 80% required. For the stochastic source it is mean total variation distance (TV), with at most 0.20 required. Bold values meet their registered threshold.

| Frozen source | Exact return, uniform | Exact return, visited | Learned V, uniform test | Learned V, visited test | Test boards with positive visitation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Minimax | **100.00%** | **100.00%** | 56.92% | 50.00% | 2 |
| Deliberately misses wins | 37.70% | 33.33% | 33.85% | 66.67% | 3 |
| Heuristic | 75.93% | 77.78% | 62.69% | 66.67% | 3 |
| Trained neural policy | 56.62% | 55.56% | 50.51% | 33.33% | 3 |
| Stochastic heuristic (TV) | 0.2764 | 0.5381 | 0.4376 | 0.4872 | 780 |

Each deterministic source visited nine boards in its game from the empty board. The visited-test percentages therefore describe just two or three decisions per deterministic source. The stochastic source visited all 4,520 decision boards with positive probability and averaged 7.613 decisions per game. Its test boards contributed 2.594 expected decisions per game. The exact behavioral decoder reproduced each source distribution on every board.

## Why the moves differed

For each source and legal action, we computed exact expected terminal payoff under the source's continuation. One calculation averaged successor-state values from X's perspective. A second propagated action values from each mover's perspective. The two agreed as rational numbers on all 16,167 legal actions for every source. Equal action values can support multiple moves, and a source can also choose a move with strictly lower expected payoff.

| Deterministic source | Same move as exact fixed-tie choice | Different equally good move | Strictly lower return |
| --- | ---: | ---: | ---: |
| Minimax | 4,520 | 0 | 0 |
| Deliberately misses wins | 1,704 | 1,016 | 1,800 |
| Heuristic | 3,432 | 853 | 235 |
| Trained neural policy | 2,559 | 1,094 | 867 |

For the stochastic source, these three categories accounted for 72.36%, 5.63%, and 22.02% of source action probability on the uniformly weighted boards. Even a stochastic decoder free to mix among equally good moves must discard the 22.02% outside the best set. That alone exceeds the 0.20 TV limit. The fixed lowest-index tie rule raises the observed distance to 0.2764. [Per-board choices, values, and probability masses](experiments/tictactoe_pilot/results/boards.csv.gz)

Three held-out positions illustrate the attribution. Cells are numbered 0–8 in row-major order, and the three board rows are separated by slashes.

- **Tie.** On `.../.../.OX`, X’s heuristic takes the center (4). Moves 2, 4, and 5 all yield an X win under heuristic continuation. The exact tie rule picks 2. Learned V picks 4, restoring the source move.
- **Deliberate mistake.** On `.../OXX/OXO`, X can take 0, 1, or 2. Their exact returns under the fallible source are 0, +1, and 0. The source chooses 0 and draws. Exact recovery and both learned return models choose 1 and win immediately.
- **Fitting error.** On `..O/O.X/X.X`, O must play 7 to draw against minimax. Each other legal move loses. Learned V predicts X payoffs of 0.954, 1.547, 1.058, and 1.517 for moves 0, 1, 4, and 7. O minimizes those predictions and chooses 0. Learned Q chooses 7. The existing regressors have unbounded linear outputs.

## Held-out fits

The imitation network learned from the source's moves. The V and Q networks learned terminal returns from the same frozen source. For stochastic imitation we evaluated the entire softmax distribution. The two return networks performed greedy move selection. The MSE columns compare the candidate move scores against exact X-perspective returns on all 2,747 legal moves across the 780 test boards.

| Source | Direct imitation | Successor V | Direct Q | Successor V MSE | Direct Q MSE | V changes from exact move (tied / strictly worse) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Minimax | 76.54% | 56.92% | 49.36% | 0.583 | 0.520 | 203 / 133 |
| Deliberately misses wins | 70.77% | 33.85% | 33.85% | 0.500 | 0.483 | 250 / 215 |
| Heuristic | 85.38% | 62.69% | 48.21% | 0.478 | 0.575 | 219 / 126 |
| Trained neural policy | 90.90% | 50.51% | 53.08% | 0.625 | 0.624 | 201 / 153 |
| Stochastic heuristic (TV) | 0.1847 | 0.4376 | 0.6145 | 0.267 | 0.379 | 108 / 153 |

For the frozen neural source, replacing exact values with learned V restored the source move on 95 test boards and lost it on 145. Direct imitation passed for this source, the heuristic, and the stochastic source under uniform test weighting. The stochastic imitation result changed to TV 0.3014 under source visitation. These are single-fit measurements. The effect of varying training error independently is reserved for Task 67. [All fidelity results](experiments/tictactoe_pilot/results/fidelity.csv) · [Calibration means, errors, and bin counts](experiments/tictactoe_pilot/results/calibration.csv)

## Playing strength

Each player played full games from the empty board in both seats. Each opponent family received one quarter of the panel weight. The five frozen random networks shared their family's quarter equally. The other families were uniform legal play, the heuristic, and minimax. The game score is win probability minus loss probability.

| Source | Original panel W / D / L (%) | Original win rate vs random nets | Original score | Exact return score | Imitation score | Learned V score | Learned Q score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Minimax | 42.51 / 57.49 / 0.00 | 80.00% | +0.425 | +0.425 | +0.050 | +0.185 | −0.214 |
| Deliberately misses wins | 18.68 / 68.25 / 13.07 | 40.00% | +0.056 | −0.305 | −0.213 | −0.370 | −0.516 |
| Heuristic | 45.21 / 54.64 / 0.16 | 90.00% | +0.450 | +0.377 | +0.465 | +0.135 | −0.287 |
| Trained neural policy | 38.89 / 43.83 / 17.28 | 80.00% | +0.216 | −0.151 | +0.087 | −0.188 | −0.001 |
| Stochastic heuristic | 41.67 / 37.39 / 20.94 | 83.00% | +0.207 | +0.433 | +0.120 | +0.081 | −0.170 |

Greedy exact return raised the stochastic source’s panel win probability from 41.67% to 56.05%. For the frozen neural source, it raised wins from 38.89% to 40.31% while losses rose from 17.28% to 55.45%. A move valued under source self-play can have a different consequence when the recovered player takes every subsequent turn against an external opponent. All 900 matchup rows integrate the known move distributions across reachable outcomes. They report per-seat and balanced probabilities, paired differences, and separate head-to-head results. These probabilities are conditional on the five specified random networks and the other frozen opponents. [Matchup table](experiments/tictactoe_pilot/results/matchups.csv)

## Replay and parity fixtures

The protocol and source weights were frozen in commit `eaf79b3` before measuring reconstruction. The source neural network learned heuristic moves on 528 distinct training boards. Recovery used the original orientations of 2,980 training boards, with another 760 boards assigned to validation. Test and validation symmetry classes were withheld. Policy imitation used the training decisions and Q used their 10,864 legal moves. V used 1,215 eligible positions after also excluding every successor queried by a validation or test decision. The differing supervision counts matter when comparing V and Q. For the stochastic source we used eight action draws per training decision and sixteen rollouts per return label. The five pilot sources each received one prespecified fit. [Manifest](experiments/tictactoe_pilot/manifest.json) · [Training counts and file hashes](experiments/tictactoe_pilot/results/run.json)

Run these commands from the repository root.

```sh
python scripts/run_tictactoe_pilot.py --output-dir artifacts/tictactoe_pilot_replay
python -m pytest -q
python scripts/demo_tictactoe.py
```

The recorded run used CPython 3.13.14 on Windows with the existing pure-Python trainers. A same-environment replay can be checked against the SHA-256 hashes in `run.json`. For backend parity, compare the exact columns in the 22,600-row compressed CSV and replay the saved model predictions with numerical tolerances. Each array in a board row follows its `legal_moves` order. Exact probabilities, visit weights, and returns are stored as integer or fraction strings. The matchup table uses double-precision enumeration. The held-out source and opponent seeds in the manifest are reserved for Task 62. [Per-board fixtures](experiments/tictactoe_pilot/results/boards.csv.gz) · [Recovery checkpoints](experiments/tictactoe_pilot/results/recovery_weights.json)
