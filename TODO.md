# TODO (Next steps after Task 1)

- [x] 1. Add state-space sampling utilities to generate on-policy and off-policy tic-tac-toe datasets.
- [x] 2. Implement a small MLP policy network and training loop for tic-tac-toe.
- [x] 3. Freeze the trained policy network and generate Monte Carlo value targets with configurable rollout budgets.
- [x] 4. Implement and train a value network `V_phi(s)` on rollout labels.
- [x] 5. Add evaluation metrics: action agreement, top-k agreement, win/draw rate, and value calibration plots.
- [x] 6. Add optional `Q_phi(s,a)` training and compare direct Q-recovery vs successor-state V-recovery.
- [ ] 7. Add symmetry augmentation/reduction for tic-tac-toe to improve sample efficiency.
- [ ] 8. Add CLI entrypoints for data generation, training, and evaluation.
- [ ] 9. Port the environment/pipeline structure to Connect Four.
- [ ] 10. Document experiment configurations and reproducibility settings (seeds, splits, versions).
