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

- [ ] 15. Add plotting/reporting for gameplay/score-over-time.
  - Visualize checkpointed evaluation score trends.
  - Keep format consistent between tic-tac-toe and Connect Four.

- [ ] 16. Save human-readable gameplay samples at each checkpoint.
  - Persist at least 1–2 sampled games per checkpoint.
  - Include full move sequence and rendered board states after each move.
  - Provide game-format-appropriate rendering for tic-tac-toe and Connect Four.

- [ ] 17. Add periodic checkpoint saving for policy, `V(s)`, and `Q(s,a)` training.
  - Save model weights and optimizer/training-state metadata.
  - Record enough state to resume deterministically where feasible.

- [ ] 18. Implement checkpoint loading and resume/partial-run continuation.
  - Resume from latest or specified checkpoint.
  - Continue logging without corrupting prior metrics.
  - Verify resumed runs still emit plots/evaluations/samples at expected intervals.

- [ ] 19. Add CLI support for monitoring/checkpoint workflows.
  - Flags for evaluation cadence, checkpoint cadence, output dirs, and resume behavior.
  - Subcommands/utilities to regenerate plots from saved logs.

- [ ] 20. Add tests and smoke demos for monitoring and checkpointing.
  - Unit tests for telemetry writers/readers and checkpoint I/O.
  - Integration smoke tests for resume behavior and periodic evaluation hooks.
  - Demo scripts that produce example plots and gameplay sample artifacts.
