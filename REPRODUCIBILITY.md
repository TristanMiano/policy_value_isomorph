# Reproducibility Guide

This document records experiment settings for the current minimal pipeline and provides a stable way to rerun comparable results.

## Version and environment

- Package version: `policy-value-isomorph==0.1.0`.
- Python requirement: `>=3.10`.
- Runtime dependencies: none beyond Python stdlib.
- Dev dependency for tests: `pytest>=8.0`.

Source of truth for these constraints is `pyproject.toml`.

## Determinism and seeds

The project uses explicit integer seeds in the stochastic pipeline pieces:

- Off-policy dataset generation: `generate_off_policy_dataset(..., seed=...)`.
- Policy MLP training: `train_policy_mlp(..., seed=...)`.
- Value MLP training: `train_value_mlp(..., seed=...)`.
- CLI commands expose `--seed` for data/train/eval.

Recommended default seed for baseline reproduction: `0`.

## Dataset splits

Current implementation intentionally keeps splitting minimal and inspectable:

- For quick experiments in `policy-value-cli train` and `policy-value-cli eval`, a single on-policy dataset is generated and reused directly.
- Tests validate behavior and shape/logic correctness, not benchmark-quality generalization.

Recommended reproducible split protocol for reporting numbers:

1. Generate one on-policy dataset with a fixed seed and episode count.
2. Create an index-based deterministic split, e.g. first 80% train, last 20% validation.
3. Reuse the same split indices for policy/value comparisons.
4. Report both split ratio and absolute sample counts.

This keeps reproducibility explicit without adding extra framework dependencies.

## Canonical command recipes

Run from repository root.

### 1) Install and verify

```bash
pip install -e .[dev]
pytest -q
python scripts/demo_tictactoe.py
```

### 2) Dataset summary runs

```bash
policy-value-cli data --mode on --episodes 120 --seed 0
policy-value-cli data --mode off --episodes 120 --seed 0
```

### 3) Baseline train run

```bash
policy-value-cli train \
  --episodes 120 \
  --policy-hidden 24 \
  --value-hidden 24 \
  --policy-epochs 60 \
  --value-epochs 60 \
  --learning-rate 0.03 \
  --rollouts 1 \
  --seed 0
```

### 4) Baseline eval run

```bash
policy-value-cli eval \
  --episodes 90 \
  --policy-hidden 24 \
  --value-hidden 24 \
  --epochs 40 \
  --games 30 \
  --rollouts 1 \
  --seed 0
```

## Reporting template

When logging an experiment, include at minimum:

- Commit hash.
- Python version.
- Exact command used.
- Seed.
- Episode count.
- Model widths/epochs/lr/rollout budget.
- Train/validation split rule (if used).
- Key outputs (losses, agreement, W/D/L rates).

## Notes

- Tic-tac-toe environment is deterministic; residual variation comes from seeded sampling/training choices.
- Keep root-player perspective and sign conventions unchanged when comparing across runs.
