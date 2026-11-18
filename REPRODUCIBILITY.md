# Reproducibility Guide

This document records experiment settings for the current minimal pipeline and provides a stable way to rerun comparable results.

## Version and environment

- Package version: `policy-value-isomorph==0.1.0`.
- Python requirement: `>=3.10`.
- Runtime dependencies: none beyond Python stdlib.
- Dev dependency for tests: `pytest>=8.0`.

Source of truth for these constraints is `pyproject.toml`.

## Determinism and seeds

A **seed** is an integer used to repeat the same sequence of random choices. Use `seed` in `generate_off_policy_dataset` to reproduce positions from games in which both players draw uniformly among legal moves at every turn. For networks trained to copy moves or predict payoffs, use it in `train_policy_mlp` or `train_value_mlp`. The command-line programs accept `--seed` for data generation, training and evaluation. The baseline examples use `0`.

## Dataset splits

The quick training and evaluation commands reuse positions from games played by the original player. Report their measurements as comparisons on training data and distinguish repeated visits from distinct boards.

For experiments divided by visit order, record the split, for example the first 80% of visits for training and the remaining 20% for validation (checking the model during development). Use the same indices for every model and count how many distinct boards occur in both groups.

For comparisons on unseen positions, record how examples were assigned to groups, their counts and weights, and which inputs each method saw. The tic-tac-toe study assigned distinct boards to training, validation, and test (positions reserved for the final comparison). It kept rotations and reflections of a board together and used the same test decisions for every method. [Study settings](experiments/tictactoe_pilot/manifest.json)

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

Use the two mandatory project protocols in experiments, reports, progress updates, and chat summaries. Define a new term before using it. Express each measurable question and answer numerically. [Project instructions](AGENTS.md#mandatory-research-protocols)

- Identify the code revision, Python version, command, and random seeds (numbers used to reproduce random choices).
- Describe each original player, each method used to copy it, and each opponent by its move rule, training history, use of randomness, and rule for tied scores.
- Give the number of games, training examples, training passes, adjustable model parameters, and simulated game continuations used to estimate each value. Describe how positions were assigned to training and evaluation.
- Define each measurement, its scale, and its direction before presenting it. For example, **move agreement** is the fraction of evaluated decisions at which the recovered player chooses the source player's move. Give the numerator and denominator or the population weights. For play, give separate win, draw, and loss rates by opponent and seat.
- Report each value, its difference from the named comparison, variation across runs, and uncertainty from sampling when applicable. Keep undefined measurements distinguishable from measured zero. Keep cutoffs and binary research verdicts out of tables and prose.

## Notes

- Tic-tac-toe environment is deterministic; residual variation comes from seeded sampling/training choices.
- Keep root-player perspective and sign conventions unchanged when comparing across runs.
