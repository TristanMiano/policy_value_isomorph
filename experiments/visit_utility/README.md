# How often can a player win using only learned board scores?

**The strongest common rule in this search scored the final move.** It takes the square root of that move’s fitted frequency, gives it a positive sign if the player choosing now made it and a negative sign if the opponent did, and discounts it by 0.9 for each preceding turn. Using only the board after a move as input, it won **60.74 ± 3.34%** with equal weight for tic-tac-toe, Connect Four and 4×4 Reversi. Replacing every learned frequency with the *same* number raised that to **65.56 ± 1.32%**. The learned frequencies predicted something about the eventual winner, especially in Connect Four. The ending turn and who made it provided much of the playing strength. [Final matched games and numerical uncertainty](cells/refinement/confirmation.md)

## What the player can use

The **original** is a trained neural player frozen before its moves were recorded. The tic-tac-toe originals always preserve their best result against perfect replies, with a personal order among equally good moves. The Connect Four originals learned either immediate wins and blocks with a personal tie order, or a personal column order alone. Their actual network choices define the observations. In 4×4 Reversi every original chooses the first move in row order preserving its best guaranteed result, including a pass when required. [Architectures, original training, and checks on every reachable decision in the two smaller games](../../PHASE_V.md#what-the-players-learned)

The **board network (V)** estimates the fraction of visits taking a move from the board *after* that move. The **board-and-move network (Q)** additionally sees the previous board and candidate move. **Direct selection** chooses the largest fitted frequency immediately. **Von Neumann–Morgenstern (VNM) selection** chooses the move with the highest expected numerical score of a completed game. Here the only scores it can read are the fitted network's frequencies. We refitted the two networks with just cell occupancy and side to move, plus the action for Q. This removes the earlier networks’ hand-written line, threat, mobility and disc-count features. The environment provides legal moves, transitions, turn order and the end of play. [Input audit and refitting design](CELL_INPUTS.md)

For each candidate, both seats pick uniformly among legal moves until the game ends. Tic-tac-toe sums the possible continuations exactly. Connect Four and Reversi each use 64 sampled continuations per candidate. If a complete play takes **L** turns including the candidate, and the fitted frequency of the final move is **f**, its utility to the current player is **+0.9^(L−1) × √f** when that player moved last and **−0.9^(L−1) × √f** otherwise. Reversi passes count as turns. The square root compresses the differences between high and low frequencies and still gives zero to a zero prediction. Average these utilities, choose the largest, and plan again after the actual reply. Numerical ties at 12 decimal places go to the lowest legal move number. No simulated player consults the original, the actual opponent, a tactical rule, or the winning side. Outcomes are read separately after external games finish. [Finite planner](../../src/policy_value_isomorph/visit_selection.py) · [Larger-game planner and separate evaluator](../../src/policy_value_isomorph/visit_vnm.hpp)

## Which formulation won most often?

We first compared 88 network, continuation and whole-game-score combinations using the archived inputs, then repeated all 88 with board occupancy alone. The best occupancy-only combination in that first search used Q, uniform play by both seats and the discounted signed final frequency. Its constant-score control won more often. We therefore tested a repair that changed simulation precision, time discount and how strongly frequency differences affected the score. The refinement compared 60 configurations, using fresh external network opponents, openings and random choices. The combination named above led discovery with Q at **60.62 ± 3.69%**, followed by the identical V rule at **60.29 ± 3.60%** (eight independent originals per game). We froze the choice before evaluating it on the remaining originals. Those originals had contributed to the earlier experiments, so this refinement is conditional on known source families. [First search with archived inputs](discovery.md) · [First search with occupancy alone](cells/discovery.md) · [Refinement design](REFINEMENT.md) · [Final discovery ranking](cells/refinement/discovery.md)

**External win rate** counts games won from either seat. Four equally weighted opponent groups play uniformly, take an untrained network's favorite move, follow a fixed positional or tactical order, or use the game's tactical or optimal rule. Tic-tac-toe uses five untrained networks, a win/block/center/corner player and a player preserving the best guaranteed result. Connect Four uses two untrained networks, a central-column player and a win/block player with centrality as a tie-breaker. Reversi uses two untrained networks, a corners-then-edges player and an optimal player breaking ties in row order. Both seats receive equal weight. Tic-tac-toe enumerates game probabilities from the empty board. Each larger game supplies 4,096 external games per method, half from the ordinary start and half from legal prefixes of up to six turns. **Agreement** counts choosing the original’s move on withheld boards. Tic-tac-toe weights its withheld decision boards equally, Connect Four weights 160 sampled boards per partition equally, and Reversi weights 184 sampled boards to represent the reachable fractions with several placements, one placement, or a forced pass. [Evaluation populations](../../CHECKPOINT_F.md#what-was-compared)

Percentages below are means ± sample standard deviations across independently trained originals, after averaging their two independently drawn collections and withheld board sets and two fitting initializations per collection. In the equal-game row each game counts for one third, and the spread describes drawing one independent original from each. The **constant-score control** replaces every fitted frequency by 0.5, retaining the same sign, discount, random continuations and decision rule. Since both continuing players act uniformly, this removes the network from the choice entirely. Any common positive constant gives the same ordering in exact arithmetic.

| Game (confirmation originals) | Original wins | Direct V wins | VNM V wins | VNM Q wins | Constant-score VNM wins | Agreement, direct V → VNM V |
|---|---:|---:|---:|---:|---:|---:|
| Tic-tac-toe (24) | 44.67 ± 5.80 | 28.94 ± 3.75 | 46.75 ± 1.47 | 46.58 ± 1.41 | 46.67 ± 1.65 | 53.45 ± 3.52 → 65.50 ± 3.07 |
| Connect Four (16) | 59.30 ± 11.40 | 37.62 ± 12.19 | 84.38 ± 8.85 | 82.62 ± 9.09 | 94.36 ± 1.99 | 71.97 ± 4.49 → 37.65 ± 6.54 |
| 4×4 Reversi (16) | 67.80 ± 2.75 | 44.48 ± 6.62 | 51.10 ± 4.49 | 52.64 ± 4.79 | 55.64 ± 3.02 | 84.28 ± 1.42 → 82.39 ± 1.45 |
| Equal games (24/16/16) | 57.26 ± 4.36 | 37.01 ± 4.79 | **60.74 ± 3.34** | **60.61 ± 3.46** | **65.56 ± 1.32** | 69.90 ± 1.96 → 61.85 ± 2.46 |

A **paired difference** compares the same fits on the same evaluation cases. A 95% confidence interval estimates uncertainty in the average difference across originals. A two-sided **p-value** measures how extreme the observed difference would be under a zero-average-difference reference. Holm adjustment accounts for examining multiple comparisons. The next table uses the 48 matched architecture comparisons. The complete report also gives two-sided raw p-values, per-original differences and intervals obtained by resampling whole originals for the main formulation comparisons. [Architecture comparisons](cells/refinement/architecture_contrasts.json.gz) · [All rule comparisons](cells/refinement/confirmation.md)

| Game | VNM V minus direct V in win percentage points, spread [95% interval], adjusted p | VNM V minus constant in win percentage points, spread [95% interval], adjusted p |
|---|---:|---:|
| Tic-tac-toe (24) | +17.81 ± 3.84 [16.19, 19.43], 1.17 × 10⁻¹⁵ | +0.08 ± 1.47 [−0.54, 0.70], 1.00 |
| Connect Four (16) | +46.75 ± 8.12 [42.43, 51.08], 1.42 × 10⁻¹¹ | −9.99 ± 8.72 [−14.63, −5.34], 0.00541 |
| 4×4 Reversi (16) | +6.62 ± 9.47 [1.57, 11.66], 0.104 | −4.54 ± 3.02 [−6.15, −2.93], 0.000443 |
| Equal games (24/16/16) | +23.73 ± 4.35 [21.55, 25.91], 1.01 × 10⁻¹⁹ | −4.82 ± 3.11 [−6.44, −3.19], 0.000116 |

The Q version led the selection data and the board-only V version had 0.13 points more wins on confirmation. Q minus V was **−0.13 ± 1.96 points [−1.14, 0.88]**, adjusted p = 1.00. V is the default when assigning a score to a state. In the frozen Q comparison, raising the simulation count from 8 to 64 with the square root fixed gained **6.05 ± 3.10 points [4.46, 7.65]**, adjusted p = 2.40 × 10⁻⁶. Taking the square root with 64 paths fixed gained **2.50 ± 1.82 points [1.58, 3.42]**, adjusted p = 0.000287. The earlier VNM formula that explicitly supplies game outcomes and tactical continuations won **73.03 ± 1.89%** on these cases. The frozen Q visit-only rule changed wins by **−12.42 ± 3.50 points [−14.15, −10.68]**, adjusted p = 2.48 × 10⁻¹³. Each cross-game comparison uses 24/16/16 originals. [Matched changes of one component](cells/refinement/confirmation.md)

**Exact-copy self-play** has both seats deploy and replan with the identical network and rule. For V, its seat-balanced win and draw rates respectively were **50.00 ± 0.00% and 0.00 ± 0.00%** in tic-tac-toe, **48.75 ± 1.92% and 2.49 ± 3.84%** in Connect Four, and **46.48 ± 3.26% and 7.03 ± 6.52%** in Reversi (24/16/16 originals). Against external opponents the corresponding draw rates were 25.94 ± 0.73%, 0.34 ± 0.53% and 7.50 ± 2.87%. The archive gives all external wins, draws and losses and the Q self-play results. [Outcomes by rule](cells/refinement/confirmation.md)

## What scores did the board network give wins and losses?

For the V player, the **raw terminal score** is the original fitted frequency for the final board, before the square root, the change of sign, or discounting. On the actual played paths, that raw score was higher on wins by **0.344** in Connect Four. In tic-tac-toe the difference was **0.023**, and in Reversi it was **−0.004**. Raw frequencies are on a scale from zero to one. A win or loss is always defined from the evaluated seat. Tic-tac-toe samples 64 separate paths per fit for this diagnostic. The larger games use the actual 64 external games per fit. [Board-network path measurements, fit counts and adjusted uncertainty](cells/refinement/state_scores_path_diagnostics.json.gz)

| Game (originals) | Diagnostic wins / draws / losses | Raw V on wins | Raw V on losses | Paired win minus loss, spread [95% interval], adjusted p | Final signed, discounted VNM utility on wins / losses |
|---|---:|---:|---:|---:|---:|
| Tic-tac-toe (24) | 2,860 / 1,586 / 1,698 | 0.399 ± 0.040 | 0.376 ± 0.061 | +0.023 ± 0.063 [−0.003, 0.050], 0.505 | +0.373 ± 0.023 / −0.322 ± 0.027 |
| Connect Four (16) | 3,456 / 14 / 626 | 0.452 ± 0.062 | 0.108 ± 0.106 | +0.344 ± 0.093 [0.294, 0.394], 6.11 × 10⁻⁹ | +0.270 ± 0.027 / −0.045 ± 0.050 |
| 4×4 Reversi (16) | 2,093 / 307 / 1,696 | 0.745 ± 0.027 | 0.749 ± 0.034 | −0.004 ± 0.033 [−0.022, 0.014], 1.00 | +0.323 ± 0.010 / +0.023 ± 0.052 |

The corresponding **raw Q** differences on the Q player's own games were **+0.163 ± 0.076 [0.131, 0.195]**, **+0.491 ± 0.129 [0.422, 0.560]**, and **+0.031 ± 0.028 [0.016, 0.046]**. Adjusted p-values were 7.76 × 10⁻⁹, 4.33 × 10⁻⁹ and 0.00494. In tic-tac-toe and Connect Four the last mover won every decisive game by the rules. In Reversi the Q player moved last in an average of 99.0% of its wins and 48.5% of its losses. That explains how the signed utility could distinguish winners even when the raw fitted board score carried little information about them. A played game's outcome also depends on the replies actually made. Ranking played endings is a different prediction task from choosing the best first move against unknown replies. [Per-game and per-original diagnostic data](cells/refinement/path_diagnostics.json.gz)

## Does summing discounted scores favor longer games?

On an unchanged prefix, adding another positive frequency increases an unsigned discounted sum. On actual games, we measured the **slope**, the change in that sum associated with one additional turn, using a straight line fitted separately within each network fit. The unsigned sum adds every fitted frequency at weights 1, 0.9, 0.81 and so on. For V it increased with duration in all three games. Yet the average win ended earlier than the average loss in each sample. In Connect Four the V player's wins ended **5.89 turns earlier** on average. Half the larger-game starts were legal prefixes, so durations count turns remaining from the test opening. These associations describe games played under the measured rule. Estimating the causal effect of delaying a move would require assigning different moves at the same board. [Within-fit slopes, original-level spread and game counts](cells/refinement/state_scores_path_diagnostics.json.gz)

| Game (originals) | Change in unsigned sum per additional turn, spread [95% interval], adjusted p | Turns on wins minus turns on losses, spread [95% interval], adjusted p |
|---|---:|---:|
| Tic-tac-toe (24) | +0.223 ± 0.048 [0.203, 0.244], 9.43 × 10⁻¹⁶ | −1.077 ± 0.197 [−1.160, −0.994], 2.55 × 10⁻¹⁷ |
| Connect Four (16) | +0.063 ± 0.021 [0.052, 0.074], 1.04 × 10⁻⁷ | −5.886 ± 4.376 [−8.217, −3.554], 0.00122 |
| 4×4 Reversi (16) | +0.138 ± 0.024 [0.126, 0.151], 1.27 × 10⁻¹¹ | −0.293 ± 0.427 [−0.520, −0.065], 0.136 |

We then deployed the unsigned sum on the **same V fits**, using the same continuations, simulation budget and external games. It won **15.75 ± 1.90%** in tic-tac-toe, **16.26 ± 4.72%** in Connect Four and **48.63 ± 5.16%** in Reversi. The equal-game rate was **26.88 ± 2.41%**. Choosing by the final-board formula instead gained **33.86 ± 3.27 percentage points [32.21, 35.51]**, adjusted p = 3.96 × 10⁻²⁸ (24/16/16 originals). The corresponding per-game gains were **31.00 ± 2.31 [30.02, 31.98]**, **68.12 ± 6.28 [64.77, 71.46]** and **2.47 ± 7.19 [−1.36, 6.29]** points. Adjusted p-values were 1.72 × 10⁻²⁶, 4.15 × 10⁻¹⁶ and 0.570. This comparison directly measures the consequences of rewarding longer, high-frequency paths. [Both board-scoring sums and all paired changes](cells/refinement/state_scores_contrasts.json.gz) · [Absolute outcome rates](cells/refinement/state_scores_estimates.json.gz)

## Fitting and scope

Each legal training candidate receives equal weight. The label is its observed count divided by all visits to the parent board. Tic-tac-toe collected complete plays starting throughout the finite decision board population. The larger games used 480 complete games per fixed original and collection, 96 from the ordinary opening and 384 from randomly sampled legal starts, with copies of the frozen original continuing both seats. In the confirmation collections, the number of distinct observed decision boards averaged **4,520 exactly**, **4,187 ± 798** and **1,188 ± 24**, respectively. Tic-tac-toe minimized summed squared frequency errors plus a penalty on squared non-intercept output weights, using 64 or 192 random rectified features and choosing the penalty and width on separate validation boards. Each larger-game network added fitted weights for 12 patterns of four board cells. It optimized half the sum of squared frequency errors and selected among 4, 16 and 48 fitting steps using validation positions. Rotations or reflections stayed in the same partition. V also excluded reserved resulting-board inputs from training. [Collection, fitting populations and loss settings](../../PHASE_V.md#positions-opponents-and-uncertainty) · [Occupancy-only architecture](CELL_INPUTS.md)

At the selected fits, the **training mean squared error**, in squared frequency units, was **0.1676 ± 0.0048 / 0.1607 ± 0.0055** for V/Q in tic-tac-toe, **0.1155 ± 0.0051 / 0.0569 ± 0.0133** in Connect Four, and **0.1279 ± 0.0090 / 0.0605 ± 0.0137** in Reversi (24/16/16 originals). The larger failures to win cannot be assigned to fitting error alone. The constant control retains meaningful information about endings, and the training objective estimates how often the original moved to a board. These results characterize three fixed games, the named players, these fitted networks and the tested budgets. They leave open what other observation targets or rollout rules could reveal. [Training candidate counts, coverage, and all measured comparisons](cells/refinement/confirmation.md)

## Reproduction

Use the Phase V numerical dependencies and C++17 compiler in the repository's Linux/WSL environment. The following commands choose the formulation and recreate its evaluations from the archived originals and visit counts. Set `PVI_VISIT_UTILITY_OUTPUT` to an empty directory to regenerate the results separately. The optional compiler build is cached outside the repository. Each completed original is stored atomically and can be resumed. [Environment and installation](../../REPRODUCIBILITY.md)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for study in archived cells refinement; do
  unset PVI_VISIT_SEARCH
  if [ "$study" = refinement ]; then
    export PVI_VISIT_INPUTS=cells PVI_VISIT_SEARCH=refinement
  else
    export PVI_VISIT_INPUTS="$study"
  fi
  python scripts/compare_visit_utilities.py discovery
  python scripts/report_visit_utilities.py discovery
  python scripts/compare_visit_utilities.py confirmation
  python scripts/compare_visit_utilities.py matched
  python scripts/report_visit_utilities.py confirmation
  python scripts/render_visit_results.py discovery
  python scripts/render_visit_results.py confirmation
done
python scripts/compare_visit_utilities.py state_scores
python scripts/compare_visit_utilities.py board_sums
python scripts/report_visit_utilities.py state_scores
```

For a fitted state-frequency table in tic-tac-toe, `FiniteVisitPlanner(board, frequencies).law()` deploys this rule on V. For the larger games, `visit_game_player(engine, seed)` uses installed V weights. Pass `standard_visit_vnm("Q")` to deploy the identical formula on Q. Each compressed record includes the original, partition, initialization, external cases, agreement, and seat-level outcomes. The path files retain the actual sequence of predicted frequencies and the first mover’s identity relative to the evaluated seat, allowing independent calculation of the utility and duration. [Frozen refinement choice](cells/refinement/selection.json) · [Statistical procedure](../../scripts/report_visit_utilities.py) · [Independent path checks](../../tests/test_visit_selection.py)
