# Recovering a frozen tic-tac-toe player from observed moves

A **policy** is a rule that chooses a move from a board. We first trained a neural policy, froze its adjustable weights, and checked its playing strength. We then sampled games from the frozen policy, counted transitions between boards, and trained separate networks to predict their frequencies. We compared two ways of using each fitted network. The **direct decoder** takes the highest immediate score. The **von Neumann–Morgenstern (VNM) decoder** takes the highest average utility of complete sampled plays following a candidate move. Its utility comes from the learned network.

## Which neural player was trained?

**Minimax** chooses the move with the best guaranteed outcome against an opponent making its best reply. Cells are numbered 0–8 across the rows. The teacher resolves ties by lowest cell number. We solved every reachable decision board by working backward from wins, draws and losses, then used only the teacher's chosen moves as the network's training labels. The network takes ten inputs (nine cells encoded as O = −1, empty = 0, X = +1, followed by the side to move), has two hidden layers of 128 units each, and outputs nine move scores. Each hidden unit uses a **rectified linear activation**, which keeps a positive input and returns zero for a negative input. The policy chooses its highest-scoring legal cell. Its outputs are move preferences. It receives no value target and performs no search when choosing a move.

Training used all 4,520 reachable decision boards, shuffled in batches of 256. The loss was the negative log probability assigned to the teacher's move after normalizing the legal scores. **Adam**, an optimizer that scales each weight update using running estimates of past gradients, used a step size of 0.003 and its library defaults for other settings. Seed 7101 fixes the initial weights and random ordering. The runner checks choices every ten passes, with up to 300 passes and step-size reductions by a factor of 0.35 after passes 100 and 200. At pass 80 the network chose the teacher's exact move on **4,520/4,520** boards. The largest guaranteed payoff given up anywhere was **zero**. Exhaustive teacher data establish what this frozen source can do throughout the game. The later test positions are withheld from the recovery fits.

The older neural player was taught a priority rule on 528 distinct boards. It used the same ten inputs, one hidden layer of 16 units with the hyperbolic tangent activation (a smooth output between −1 and +1), and nine move scores. It used 30 shuffled passes and one gradient update per example with step size 0.025. On the 780 test boards it gave up a better guaranteed outcome on 157 boards. These are two different training objectives. The demonstrated recipe achieves optimal play on this finite game. Finding the smallest architecture or the least training time or data would require additional comparisons. [Original training procedure](src/policy_value_isomorph/pilot_protocol.py) · [New training procedure](src/policy_value_isomorph/neural_game_control.py)

## From games to training targets

For a board `s` and move `a`, let `C(s,a)` count occasions when the frozen player actually took that move. Divide by the total observations of moves from that board. The resulting target estimates the probability of moving from `s` to its immediate successor. An observed board can have legal moves with zero selections. A board with no visits has an undefined distribution.

We first played 256 complete games from the empty board with the source playing both sides. Then we played 32 games starting at each reachable decision board. The latter gives poorly visited boards direct observations. Within each of the training, validation and test groups we counted transitions from all visits to parents in that group in the games started in that group. This yields independent batches for estimating each group's frequencies. Training has 2,980 decision boards, validation has 760 for choosing training duration, and the test comparison uses 780. Rotations and reflections are kept together. The raw counts from empty-board games and the counts used for fitting are saved separately. [Counts, denominators and normalized targets](experiments/neural_tictactoe_control/visits.csv.gz)

A subsequent experiment repeated the collection with one complete game per starting board. For the frozen optimal neural player this gave identical normalized frequencies on all 16,167 legal moves. It also compared empty starts, random opponents and restarts on rarely observed boards at fixed numbers of observed source moves. [Coverage, agreement on common and unusual boards, and game outcomes](TICTACTOE_SAMPLING.md)

All new recovery networks use two hidden layers of 128 rectified linear units and a single output converted to the interval from zero to one by the logistic function. The fitted output is the learned move score. We measure its squared difference from the observed frequency on each legal candidate move. Candidates receive equal weight during fitting. The same weights and score scale are used for both decoders.

| Network name | Inputs to the learned score |
| --- | --- |
| Board value (`value_mlp`, V) | Ten numbers describing the resulting board and its side to move. Different parents can lead to the same input with different frequency targets. |
| Transition value (`transition_mlp`) | Twenty numbers describing the current board and the resulting board. |
| Action value (`q_mlp`, Q) | Nineteen numbers describing the current board, its side to move and nine indicators identifying the candidate move. |

The transition and action networks see all 10,864 legal training moves. For board value we also remove any training candidate whose resulting board occurs as a validation or test input, and exclude test inputs from validation. This leaves 4,980 training candidates for board value. We minimize average squared frequency error using Adam, batches of 256, an initial step size of 0.003 and the same reduction schedule as the source. We evaluate after the first and every tenth pass up to 120 and keep the weights with the lowest validation error. For the optimal neural source we repeat the fits with seeds 7201, 7202 and 7203. A separate fit on all 16,167 candidate moves uses seed 7201 and selects by training error through 300 passes. That fit measures attainable performance with full observation. Each of the other sources has one held-out fit with seed 7201. [Counts, training curves and selected passes](experiments/neural_tictactoe_control/training.json)

## What the VNM decoder averages

Write `h` for the moves so far, `a` for the candidate move and `τ` for a complete play. We estimate the move score `S_P(h,a) = E[U(τ) | h, a, game rules, continuation policy P]`. For each candidate, take the move, then sample **both sides from copies of the original frozen policy until the game ends**. For a randomized source we use 64 independently drawn continuations per candidate (seed 8201). For a fixed-move source there is one possible continuation, which we play to the end. Every method for a given source uses the same sampled paths.

For each move along a path the network gives a learned frequency score for the player making that move. Write `u_t` for this score from X's perspective, positive for an X move and negative for an O move. We evaluated the following whole-game utilities.

- **Final move** uses only the learned score `u_last` of the final transition.
- **Sum of moves** uses `U(τ) = Σ u_t` over the played moves.
- **Time-discounted sum** uses `U(τ) = Σ 0.9^t u_t`. The candidate move has weight one, the next move 0.9 and the following move 0.81. Taking the time origin at the current decision removes a common prefix and a common positive scale from the comparison.

X selects the highest average X utility, O the lowest. All ties go to the lowest-numbered legal cell. Using opposite signs for the players rewards the current player's preferred moves and penalizes allowing the opponent its preferred moves. These utilities can assign different scores to plays with the same winner. The observed counts estimate move frequencies. Interpreting those frequencies as contributions to a player's utility is an explicit assumption. Identifying the player's own preferences over uncertain outcomes would require separate choice evidence. We also ran each utility with two copies of the **fitted network's direct move rule** finishing the game. Rows ending in `_copies` use this second rule, which can plan without querying the original policy. For playing strength the recovered player uses its selected rule on **every** turn against each external opponent. [Sampling and utility calculation](src/policy_value_isomorph/trajectory_utility.py)

## Players and measurements

The original players are the certified neural player above, its minimax teacher, a **priority-rule player** that takes an immediate win, blocks one, then prefers the center, a corner or any legal cell, a **missed-win player** that takes the first nonwinning move whenever an immediate win and an alternative coexist and otherwise uses the priority rule, the older neural imitator, and a **randomized priority player** that independently uses the priority rule with probability ¾ and otherwise draws uniformly among legal cells. The fixed rules break ties by lowest cell number.

**Agreement** is the fraction of equally weighted test boards on which the recovered move equals the original move. For the randomized source, **total variation distance** is half the sum of the absolute differences in legal-move probabilities, averaged across boards. Lower distance indicates closer imitation. An **optimal move** preserves the best payoff the mover can guarantee, including alternative equally good moves. Whole-game percentages give equal weight to X and O. Four opponent groups each receive one quarter of the weight. They are uniformly random legal play, five untrained networks with fixed random weights that each take their highest-scoring legal cell, the priority-rule player and the minimax teacher. We enumerate outcomes of these fixed players from the empty board. [Move comparisons](experiments/neural_tictactoe_control/move_comparisons.csv) · [Game results by opponent and seat](experiments/neural_tictactoe_control/games.csv)

## Outcomes for the optimal neural source

The 256 games from the empty board produced 2,304 move observations on just **nine distinct boards**. Starting games at each decision board instead produced 243,488 counted moves and observations on all 4,520 boards. Choosing each board's most frequent move directly from its count table reproduced the source on 4,520/4,520 boards. This reference has observations for every evaluation board.

The following means give equal weight to the three held-out fits. Agreement and optimal moves use the same 780 test boards. The final three columns are whole-game rates against the four opponent groups. Both sides follow the original neural source during simulated continuations in this table.

| Score input | Move selection | Agreement | Optimal moves | Win | Draw | Loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original neural source | Highest legal policy score | 100.00% | 100.00% | 42.51% | 57.49% | 0.00% |
| Board | Direct | 71.11% | 83.50% | 39.77% | 41.28% | 18.94% |
| Board | Final move | 79.02% | 99.32% | 44.28% | 55.49% | 0.23% |
| Board | Sum of moves | 73.63% | 93.12% | 38.60% | 53.02% | 8.38% |
| Board | Time-discounted sum | 74.79% | 92.99% | 40.34% | 59.58% | 0.08% |
| Transition | Direct | 82.78% | 90.68% | 41.54% | 40.94% | 17.52% |
| Transition | Final move | 74.87% | 92.78% | 43.27% | 49.98% | 6.75% |
| Transition | Sum of moves | 84.06% | 95.56% | 38.70% | 44.12% | 17.18% |
| Transition | Time-discounted sum | 84.74% | 95.56% | 38.28% | 43.99% | 17.73% |
| Action | Direct | 83.72% | 91.54% | 43.75% | 31.53% | 24.72% |
| Action | Final move | 77.61% | 95.90% | 42.98% | 53.40% | 3.62% |
| Action | Sum of moves | 84.40% | 96.32% | 42.95% | 37.58% | 19.47% |
| Action | Time-discounted sum | 84.96% | 96.37% | 41.57% | 43.16% | 15.27% |

For action inputs, time-discounting raised agreement by 1.24 percentage points and lowered losses by 9.45 points relative to direct selection. Direct loss rates ranged from 16.32% to 41.49% across the three fits. With time-discounting the range was 13.49% to 16.36%. These ranges describe variation across fits of the same source. The files also give each fit's results with equal board weights and with boards weighted by the probability of visiting them in games played by the source. [Individual fit results](experiments/neural_tictactoe_control/move_comparisons.csv) · [Individual game results](experiments/neural_tictactoe_control/games.csv)

**Example from the empty board.** With the original optimal player continuing for both sides, all nine openings end in draws. For the action network with seed 7201, the summed utilities of opening in the upper-left corner, top-middle cell and center were +1.262, −1.740 and +0.645. The learned final-move scores for those same plays were +0.0011, +0.0011 and +0.9834. Direct selection and the sum chose the upper-left corner. Final-move utility chose the center. Thus even when every possible outcome has the same game payoff, the learned trajectory utility can rank the moves differently.

**Whose play is simulated matters.** Holding the fitted scores and the undiscounted sum fixed, we changed both players after the candidate move. The entries below average the same three fits. “Network copies” means each copy chooses its highest-scoring legal move directly at each subsequent turn.

| Score input | Both original, agreement / win / loss | Two network copies, agreement / win / loss |
| --- | ---: | ---: |
| Board | 73.63% / 38.60% / 8.38% | 71.54% / 43.76% / 13.46% |
| Transition | 84.06% / 38.70% / 17.18% | 83.68% / 45.66% / 16.10% |
| Action | 84.40% / 42.95% / 19.47% | 84.06% / 44.73% / 19.76% |

## How much difference did training make?

We fitted the action network on **all** 16,167 legal candidates from the optimal source's observed visit table. With squared frequency error as the training loss it reproduced the source on 97.39% of all 4,520 boards. In a follow-up check we changed only the fitting loss to **binary cross-entropy**, the average negative log probability assigned to observing or not observing each candidate according to its empirical frequency. We kept the architecture, initial weights, observations, batches, 300 training passes, and selection by squared training error fixed. Direct selection now reproduced **4,520/4,520 moves**, won **42.51%** of games and lost **0.00%**. Both the summed and time-discounted trajectory utilities also reproduced 4,520/4,520 moves with either continuation rule. The final-move utility reproduced 77.61% with the source continuing. This full-observation result demonstrates exact neural reconstruction under specified conditions. The 780-board held-out comparison measures how accurately a fit predicts source moves on omitted positions.

The board-only input has a separate information limit. Of the 5,477 distinct resulting positions, 2,026 have incoming moves with different observed frequency targets for the optimal source. Even assigning the best possible single score (the mean of its incoming targets) to each resulting position leaves an average squared error of 0.08574 over the 16,167 moves. The fitted full-observation board network reached 0.09446. The current-board-and-action input distinguishes those incoming moves.

For independent confirmation, repeat the fitting-loss comparison with settings selected on validation boards, vary coverage, and report variation across new source networks. The present result demonstrates one sufficient architecture and training procedure for an optimal neural player. It also identifies conditions under which a value network can reproduce that player exactly. [Training measurements](experiments/neural_tictactoe_control/training.json) · [All-board and held-out comparisons](experiments/neural_tictactoe_control/move_comparisons.csv)

## Comparison with the earlier payoff-trained weights

We also took the **original saved V and Q weights** from the earlier pilot. Direct selection uses their predicted X game payoff. For the paired VNM comparison we used the same V on the terminal board, or the same Q on the last board and move, as the utility of the completed trajectory. We simulated the original player on both sides. This uses a single network output per trajectory and avoids repeatedly counting an already predicted remaining payoff.

Each cell below gives agreement on the 780 test boards followed by whole-game win rate. In the randomized row the first number is total variation distance on its 0–1 scale.

| Original source | V direct | V final move | Q direct | Q final move |
| --- | ---: | ---: | ---: | ---: |
| Minimax | 56.92%, 45.07% | 46.67%, 27.42% | 49.36%, 38.89% | 37.95%, 29.33% |
| Missed win | 33.85%, 29.74% | 36.67%, 23.54% | 33.85%, 18.31% | 30.77%, 29.59% |
| Priority rule | 62.69%, 42.74% | 43.08%, 43.50% | 48.21%, 33.68% | 32.18%, 22.15% |
| Older neural imitator | 50.51%, 36.68% | 51.15%, 21.65% | 53.08%, 33.73% | 43.97%, 28.69% |
| Randomized priority | 0.4376, 39.24% | 0.4915, 39.31% | 0.6145, 28.14% | 0.6251, 32.48% |

For comparison, on the randomized source the **visit-trained** action network had distance 0.1982 and won 45.07% under direct selection. The time-discounted sum had distance 0.2020 and won 45.23%. Its typical sampling error in the estimated candidate utility was 0.0501 utility units, calculated as the square root of the average estimated variance of the 64-rollout means on the 2,747 legal test moves. The published game rates condition on that realized set of choices. [Full numerical comparisons](experiments/neural_tictactoe_control/move_comparisons.csv) · [Full game outcomes](experiments/neural_tictactoe_control/games.csv)

## Reproduction

Install the optional training libraries with `python -m pip install -e ".[neural-control]"`. From the repository root run `python scripts/run_neural_tictactoe_control.py --output-dir artifacts/visit_replay`. The published run used a single CPU thread. The saved weights contain the frozen source and every new recovery fit. [Executable experiment](scripts/run_neural_tictactoe_control.py) · [Trained network weights](experiments/neural_tictactoe_control/weights.pt)
