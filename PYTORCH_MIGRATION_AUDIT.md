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
