# TODO (Next steps after Task 1)

- [x] 1. Add state-space sampling utilities to generate on-policy and off-policy tic-tac-toe datasets.
- [x] 2. Implement a small MLP policy network and training loop for tic-tac-toe.
- [x] 3. Freeze the trained policy network and generate Monte Carlo value targets with configurable rollout budgets.
- [x] 4. Implement and train a value network `V_phi(s)` on rollout labels.
- [x] 5. Add evaluation metrics: action agreement, top-k agreement, win/draw rate, and value calibration plots.
- [x] 6. Add optional `Q_phi(s,a)` training and compare direct Q-recovery vs successor-state V-recovery.
- [x] 7. Add symmetry augmentation/reduction for tic-tac-toe to improve sample efficiency.
- [x] 8. Add CLI entrypoints for data generation, training, and evaluation.
- [x] 9. Port the environment/pipeline structure to Connect Four.
- [x] 10. Document experiment configurations and reproducibility settings (seeds, splits, versions).

## Monitoring, checkpoints, and resumability (next jobs)

- [x] 11. Define and standardize training telemetry schema across policy, value `V(s)`, and action-value `Q(s,a)` runs.
  - Track loss vs iteration/epoch.
  - Track loss vs wall-clock time.
  - Track run metadata (seed, game, model type, hyperparameters, checkpoint step).

- [x] 12. Implement lightweight logging backends for telemetry (e.g., CSV/JSONL) with minimal dependencies.
  - Ensure logs are append-safe for resumed runs.
  - Keep output directory layout consistent across tic-tac-toe and Connect Four.

- [x] 13. Add plotting utilities for training curves.
  - Generate loss-by-iteration/epoch plots.
  - Generate loss-by-wall-clock-time plots.
  - Support policy, `V(s)`, and `Q(s,a)` training runs.

- [x] 14. Add periodic gameplay evaluation during training.
  - Every `N` epochs/steps, run a fixed evaluation bundle of games.
  - Report aggregate score metrics over time (win/draw/loss or equivalent average score).
  - Store results in machine-readable form for later plotting.

- [x] 15. Add plotting/reporting for gameplay/score-over-time.
  - Visualize checkpointed evaluation score trends.
  - Keep format consistent between tic-tac-toe and Connect Four.

- [x] 16. Save human-readable gameplay samples at each checkpoint.
  - Persist at least 1–2 sampled games per checkpoint.
  - Include full move sequence and rendered board states after each move.
  - Provide game-format-appropriate rendering for tic-tac-toe and Connect Four.

- [x] 17. Add periodic checkpoint saving for policy, `V(s)`, and `Q(s,a)` training.
  - Save model weights and optimizer/training-state metadata.
  - Record enough state to resume deterministically where feasible.

- [x] 18. Implement checkpoint loading and resume/partial-run continuation.
  - Resume from latest or specified checkpoint.
  - Continue logging without corrupting prior metrics.
  - Verify resumed runs still emit plots/evaluations/samples at expected intervals.

- [x] 19. Add CLI support for monitoring/checkpoint workflows.
  - Flags for evaluation cadence, checkpoint cadence, output dirs, and resume behavior.
  - Subcommands/utilities to regenerate plots from saved logs.

- [x] 20. Add tests and smoke demos for monitoring and checkpointing.
  - Unit tests for telemetry writers/readers and checkpoint I/O.
  - Integration smoke tests for resume behavior and periodic evaluation hooks.
  - Demo scripts that produce example plots and gameplay sample artifacts.


## Neural-network training modernization (PyTorch-first)

- [ ] 21. Audit current manual training/optimizer code paths and define PyTorch migration boundaries.
  - Identify where gradients, parameter updates, and optimizer state are currently handled.
  - List components that must remain stable during migration (data formats, CLI flags, telemetry schema).

- [ ] 22. Record the PyTorch-first decision and migration constraints in project docs.
  - Treat PyTorch as the canonical backend for this phase.
  - Keep tic-tac-toe rollout/value-recovery APIs and artifact conventions stable unless explicitly versioned.

- [ ] 23. Introduce minimal internal interfaces for PyTorch-backed models/steps.
  - Keep pure-function-friendly boundaries for data prep/evaluation.
  - Avoid unnecessary abstraction layers while enabling policy/value/Q consistency.

- [ ] 24. Port policy/value/Q MLP models to PyTorch with parity tests.
  - Preserve current input/output contracts and value-sign conventions.
  - Add forward-pass shape and dtype assertions.

- [ ] 25. Replace custom optimizer updates with configurable `torch.optim.AdamW`.
  - Expose optimizer hyperparameters through existing CLI/training entrypoints.
  - Preserve reproducibility hooks (seed handling; deterministic settings where feasible).

- [ ] 26. Optionally evaluate Muon only after AdamW baseline parity is complete.
  - Gate optimizer choice behind runtime config (no code fork).
  - Adopt only if implementation quality/maintenance is adequate.

- [ ] 27. Migrate checkpointing/resume to include PyTorch model and optimizer state.
  - Support resume from latest or specified checkpoint.
  - Provide backward-compatible loading or a one-time conversion path if required.

- [ ] 28. Expand tests and smoke demos for the PyTorch training stack.
  - Unit tests for optimizer configuration, training-step execution, and checkpoint round-trips.
  - Short end-to-end smoke runs for policy training and rollout-label + value training (plus Q when enabled).

- [ ] 29. Update docs/examples for PyTorch-first architecture iteration workflows.
  - Document commands for end-to-end tic-tac-toe runs.
  - Clarify how to add/replace model architectures with minimal pipeline churn.
