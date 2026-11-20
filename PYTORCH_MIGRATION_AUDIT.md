# PyTorch migration audit (Task 21)

This audit defines the migration boundary for replacing manual optimization code paths with a PyTorch-first training stack while keeping Task 1 tic-tac-toe rollout/value-recovery workflows stable.

## Current manual gradient/update code paths

### Policy training
- File: `src/policy_value_isomorph/policy_mlp.py`
- Entry point: `train_policy_mlp(...)`
- Manual optimization currently includes:
  - Forward pass via `_forward(...)` with Python lists.
  - Cross-entropy gradient built from masked softmax (`dlogits[target] -= 1.0`).
  - In-place SGD-style parameter updates for `w2`, `b2`, `w1`, `b1`.
  - RNG-driven per-epoch shuffling and persisted RNG state for resume determinism.

### Value training
- File: `src/policy_value_isomorph/value_mlp.py`
- Entry point: `train_value_mlp(...)`
- Manual optimization currently includes:
  - Forward pass via `_forward(...)`.
  - MSE derivative (`d_out = 2.0 * err`).
  - In-place parameter updates for all layers.
  - RNG-driven shuffling and RNG state persistence.

### Action-value (Q) training
- File: `src/policy_value_isomorph/q_mlp.py`
- Entry point: `train_q_mlp(...)`
- Manual optimization currently includes:
  - State-action feature assembly (`_encode_state_action(...)`).
  - Forward pass via `_forward(...)`.
  - MSE derivative and in-place parameter updates.
  - RNG-driven shuffling and RNG state persistence.

### Checkpoint/telemetry coupling relevant to migration
- Checkpoint operations are called directly inside training loops (`save_checkpoint`, `load_checkpoint`, `latest_checkpoint_path`).
- Telemetry rows are emitted once per epoch using `TelemetryRecord(step, loss, wall_time_seconds)` and packaged as `TrainingTelemetry`.
- Resume behavior depends on `step` and serialized RNG state in checkpoint metadata.

## Migration boundary (what changes vs what remains stable)

### Must be replaced by PyTorch in migration phase
- Internal parameter storage for MLPs (`list[list[float]]`, `list[float]`, `float`) in training execution.
- Manual forward/backward math and explicit gradient calculations.
- In-loop in-place SGD updates.

### Must remain stable during migration
- Public training entrypoints and signatures:
  - `train_policy_mlp(...)`
  - `train_value_mlp(...)`
  - `train_q_mlp(...)`
- Data contracts and dataset object types used by training/eval code:
  - `StateActionSample`, `StateValueTarget`, `StateActionValueTarget`.
- Prediction/action APIs used across evaluation and demos:
  - `policy_mlp_action(...)`, `value_mlp_predict(...)`, `q_mlp_predict(...)`, `recovered_action_from_q(...)`.
- Value-sign convention semantics documented in docstrings and used by eval code.
- CLI surface area and expected flags/workflows for existing scripts/subcommands.
- Telemetry schema (`TrainingTelemetry`, `TelemetryRecord`) and emitted fields.
- Output artifact conventions and directory layout for checkpoints, logs, and plots.

## Compatibility constraints for the first PyTorch pass

1. **Behavioral parity over architectural expansion**
   - Keep the same 1-hidden-layer MLP shapes and activation (`tanh`) for policy/value/Q until parity tests pass.

2. **Loss/objective parity**
   - Policy should keep masked legal-action behavior and cross-entropy objective equivalent to current implementation.
   - Value and Q should keep scalar regression targets and MSE objective.

3. **Reproducibility continuity**
   - Keep existing seed arguments and deterministic hooks.
   - Add deterministic PyTorch seeding/settings without removing current seed plumbing.

4. **Checkpoint and resume continuity**
   - Preserve resume-from-latest/resume-from-path behavior.
   - Maintain step semantics used by periodic evaluation/checkpoint intervals.

5. **Scope control**
   - Migration in this phase is limited to training internals.
   - Rollout generation, tic-tac-toe environment rules, and evaluation/reporting logic should not be redesigned.

## Out of scope for Task 21
- Introducing new games or new model classes.
- Changing artifact formats unless explicitly versioned later.
- Adding optimizer experimentation (e.g., Muon) before baseline AdamW parity.

## Replaying the frozen tic-tac-toe networks

Tasks 23 and 24 moved training and prediction for three networks to PyTorch. The **policy network** scores the nine cells given the board and side to move. It excludes occupied cells, then chooses the highest-scoring legal move or assigns probabilities proportional to the exponentials of the legal scores, as specified for each player. The **board-value network (V)** predicts the payoff from a board. The **board-and-move network (Q)** predicts it given a board and candidate move. In this archived experiment a win for X earns +1, a draw 0 and a loss −1. X chooses the largest predicted payoff and O the smallest, with ties resolved by lowest cell number. This replay concerns the pilot's game-payoff training. The later experiments trained on observed move frequencies.

The frozen comparison loads the original weights in both representations. Each of five original players has three fitted networks evaluated on all 4,520 reachable boards with a move, giving 22,600 decisions and 80,835 legal candidate scores per network type. The old board-level records contain every predicted payoff and move distribution. For the policy's raw scores, which precede conversion to probabilities, the replay also evaluates the original weights in the original order of arithmetic. The table gives absolute differences. [Original players, training settings and board split](experiments/tictactoe_pilot/manifest.json) · [Replay data, player rules and field definitions](experiments/tictactoe_pilot/torch_replay.json)

| Network | Decisions with a different top move | Boards with a different set of tied best moves | Largest change per legal score |
| --- | ---: | ---: | ---: |
| Policy | 0 / 22,600 | 0 / 22,600 | 5.33 × 10⁻¹⁵ in raw score |
| V | 0 / 22,600 | 0 / 22,600 | 1.11 × 10⁻¹⁵ game points |
| Q | 0 / 22,600 | 0 / 22,600 | 1.44 × 10⁻¹⁵ game points |

Across the 80,835 move scores for V and the same number for Q, zero scores changed sign for either model. The largest absolute change in legal move probability was 9.99 × 10⁻¹⁶. The original trained player and five untrained neural opponents also retained their moves on all 27,120 player–board cases. The frozen networks had zero exact ties for best legal move. A separate software test supplies tied scores and a larger score on an occupied cell to exercise the tie and legality rules.

Starting from the empty board, the replay enumerated the possible games for all 900 archived combinations of player, method, opponent and seat, including the published averages. The maximum change was 2.22 × 10⁻¹⁶ for each of win, draw and loss probability, and 2.22 × 10⁻¹⁶ points for expected score (win probability minus loss probability). These are full-population comparisons conditional on the saved weights. Variation from fitting new networks is a separate question.

For example, the **priority-rule player** takes an immediate win, blocks an immediate loss, takes the centre, takes a corner, or takes any legal cell, in that order, choosing the lowest cell number within a priority. Move agreement below counts the 780 equally weighted boards withheld from training on which a player chose that exact move. Game probabilities give equal weight to each seat and each of four opponent rules. These are the priority rule, equal probability per legal cell, best guaranteed outcome against perfect replies with lowest-cell ties, and the average of five untrained networks each choosing its highest-scoring legal cell with lowest-cell ties. Both implementations give the displayed results at this precision. [Archived move comparisons](experiments/tictactoe_pilot/results/fidelity.csv) · [Replayed game probabilities](experiments/tictactoe_pilot/torch_replay.json)

| Player | Move agreement | Win | Draw | Loss |
| --- | ---: | ---: | ---: | ---: |
| Original priority rule | 780 / 780 (100%) | 45.21% | 54.64% | 0.16% |
| Policy trained to copy the rule | 666 / 780 (85.38%) | 46.88% | 52.75% | 0.37% |
| V trained on game payoff | 489 / 780 (62.69%) | 42.74% | 27.98% | 29.29% |
| Q trained on game payoff | 376 / 780 (48.21%) | 33.68% | 3.95% | 62.37% |

### Checking training and continuation

The architecture has one hidden layer using the hyperbolic tangent, followed by a linear output. All three trainers retain their seeded starting weights, shuffled example order and one update per example using **stochastic gradient descent**, which subtracts the learning rate times that example's loss gradient from each weight. The policy loss is minus the natural logarithm of the observed legal move's probability, in natural-log units per example. The logged loss retains the original probability floor of 10⁻¹². The gradient uses the full log probability. V and Q minimize mean squared error in squared game points per example.

A small software fixture records the original Python trainers at revision `4ff4bc2` on four examples, three hidden units, three passes, learning rate 0.07 and random seed 19. At the third pass, PyTorch recorded a policy loss of 1.8132391 natural-log units per example, a V loss of 0.2678414 squared points per example and a Q loss of 0.3819410 squared points per example. Across the three epoch losses per trainer, the largest absolute differences were 4.44 × 10⁻¹⁶ for policy, 5.55 × 10⁻¹⁷ for V and zero for Q. Across all 170 fitted weights and biases the largest difference was 5.55 × 10⁻¹⁷. The software tests use an absolute tolerance of 10⁻¹² for these floating-point comparisons. New checkpoints include the shuffled example order so a split run reproduces the uninterrupted run. Older files load with their original continuation rule, which starts with a fresh index list and uses the saved random generator state. [Original inputs, losses and weights](tests/fixtures/manual_mlp_training.json) · [Training, continuation and replay tests](tests/test_torch_mlp_parity.py)

Reproduce the saved comparison from the repository root with `python scripts/replay_torch_pilot.py`. It used PyTorch 2.8.0 on CPU in 64-bit precision. To train fresh pilot networks, use `python scripts/run_tictactoe_pilot.py`. Its default destination is `artifacts/tictactoe_pilot_pytorch`. [Replay script](scripts/replay_torch_pilot.py)
