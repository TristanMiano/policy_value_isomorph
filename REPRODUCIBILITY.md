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

## Continuing a training run

A **checkpoint** stores a fit at the end of a completed pass through its training examples. New files use PyTorch format version 2 at `<directory>/<policy|value|q>/step_000002.pt` (here, two completed passes). A temporary file is replaced atomically after the complete snapshot has been written. The snapshot contains the model parameters, the optimizer's accumulated updates, precision and training settings, the order of examples, random-generator states, and elapsed training time. A digest of the ordered inputs, targets and legal-move masks identifies the training batch.

For example, these commands run two passes, then continue the same three fits to four total passes. The second command restores the original learning rate, optimizer, architecture and seed from each file. Each training log reports the newly completed steps and cumulative elapsed fit time.

```sh
policy-value-cli train --episodes 2 --optimizer adamw --learning-rate 0.001 --seed 13 --policy-epochs 2 --value-epochs 2 --q-epochs 2 --policy-checkpoint-dir artifacts/resume --value-checkpoint-dir artifacts/resume --q-checkpoint-dir artifacts/resume --telemetry-root-dir artifacts/logs
policy-value-cli train --episodes 2 --policy-epochs 4 --value-epochs 4 --q-epochs 4 --policy-checkpoint-dir artifacts/resume --value-checkpoint-dir artifacts/resume --q-checkpoint-dir artifacts/resume --policy-resume-latest --value-resume-latest --q-resume-latest --telemetry-root-dir artifacts/logs
```

Each trainer also accepts an explicit file through its `--policy-resume-checkpoint`, `--value-resume-checkpoint` or `--q-resume-checkpoint` option. Latest selects the highest numbered complete file for that model and prefers the native file when both formats represent the same step. Historical JSON files supply the saved model and shuffle state for continuing the original stochastic gradient descent rule. Older JSON files that omit the previous example order follow the original fresh-index-list convention. JSON files contain no AdamW update history. The frozen pilot weights remain readable by the prediction functions. [Checkpoint implementation](src/policy_value_isomorph/checkpointing.py)

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
python scripts/smoke_torch_training.py
```

The short training check fits and freezes a move-selecting network, plays games with copies of it to label the two payoff networks, resumes all three trainers after the first pass, and plays each finished model from both seats. It uses temporary storage by default. Add `--output-dir artifacts/smoke` to inspect a fresh run's checkpoint files and per-pass training logs. The default device is the CPU. Use `python -m pytest -m gpu -q` for the separately marked graphics-processor check.

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
