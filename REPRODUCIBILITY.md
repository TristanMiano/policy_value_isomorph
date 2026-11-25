# Reproducibility Guide

The short baseline fits three tic-tac-toe networks using PyTorch, the library that computes predictions and changes weights during training. Use Python 3.10 or newer. The package version is 0.1.0. For a new checkout, create a virtual environment with `python -m venv .venv`. Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in a POSIX shell, then run the install and verification commands below.

For the public training and evaluation calls, which player's payoff each score measures, and what is saved when freezing or resuming a model, see the [Checkpoint A interface reference](CHECKPOINT_A.md#what-later-experiments-can-rely-on).

## Dependencies and optional groups

An **extra** installs dependencies for an additional feature with `python -m pip install -e ".[name]"`. The base package requires PyTorch 2.0 or later. For a graphics-processor installation, select a compatible build of PyTorch for the installed driver. The source package runs on a central processor by default. [Package configuration](pyproject.toml)

| Group | Availability and purpose |
| --- | --- |
| `dev` | Installs pytest to run the test suite and NumPy for the array-based experiment tests. |
| `neural-control` | Installs NumPy for the existing tic-tac-toe neural training and visit-count experiments. |
| `muon` | Requests PyTorch 2.9 or later for the optional matrix-update benchmark. |
| `phase-iv` | Installs NumPy for numerical arrays, SciPy for game enumeration and statistical inference, and Matplotlib for figures. [Independent tic-tac-toe runs and result files](experiments/phase_iv/README.md) |
| `phase-v` | Installs NumPy, SciPy and Matplotlib for the repeated three-game study. Rerunning the new-game simulator also uses a C++17 compiler and Python development headers on Linux or Windows Subsystem for Linux. [Commands, operating conditions and files](experiments/phase_v/README.md) |
| Control | Planned for continuous-state environments and numerical integration once an experiment specifies them. |
| Sequence | Planned for datasets and tools for time-series or token-sequence models. |
| Large language models | Planned for tokenizers and pretrained text-model loading. |
| Visualization | Planned for optional interactive displays. Current training and gameplay plots write scalable vector graphics using the standard library. |

The first five rows are installable extras. Record the code revision, Python and PyTorch versions, device, numerical precision and seed alongside an experiment. The cross-platform workflow was exercised on Windows with Python 3.13 and PyTorch 2.8.0 (CPU), and on Ubuntu with Python 3.12 and PyTorch 2.11.0 (CPU and NVIDIA GPU). For deterministic NVIDIA training set the CUDA matrix-library workspace before starting Python, e.g. `$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"` in PowerShell or `export CUBLAS_WORKSPACE_CONFIG=:4096:8` in a POSIX shell.

## Determinism and seeds

A **seed** is an integer used to repeat the same sequence of random choices. Use `seed` in `generate_off_policy_dataset` to reproduce positions from games in which both players draw uniformly among legal moves at every turn. For networks trained to copy moves or predict payoffs, use it in `train_policy_mlp`, `train_value_mlp` or `train_q_mlp`. The command-line programs accept `--seed` for data generation, training and evaluation. The training and evaluation recipes use seed 13. The short end-to-end check uses seed 28.

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
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/demo_tictactoe.py
python scripts/smoke_torch_training.py
```

The short training check fits and freezes a move-selecting network, plays games with copies of it to label the two payoff networks, resumes all three trainers after the first pass, and plays each finished model from both seats. It uses temporary storage by default. Add `--output-dir artifacts/smoke` to inspect a fresh run's checkpoint files and per-pass training logs. The default device is the CPU. Use `python -m pytest -m gpu -q` for the separately marked graphics-processor check.

### 2) Dataset summary runs

```bash
policy-value-cli data --mode on --episodes 120 --seed 0
policy-value-cli data --mode off --episodes 120 --seed 0
```

### 3) Train and evaluate the payoff baseline

The reference player takes an immediate win, blocks an immediate loss, takes the center, takes a corner or takes any legal square, in that order with lowest-square ties. The training command fits a move network to its actions. It fits two payoff networks to games continued by that reference player, with +1 for X winning, 0 for a draw and −1 for X losing. Giving `--q-epochs` a positive value enables the network that scores a board and a move. `--episodes` counts games and `--rollouts` counts continuations per payoff label. The evaluation command fits a fresh move network and board network, compares moves with the reference player and plays both seat assignments.

```sh
policy-value-cli train --optimizer adamw --learning-rate 0.001 --seed 13 --device cpu --dtype float64 --deterministic --episodes 4 --policy-epochs 3 --value-epochs 3 --q-epochs 3 --rollouts 1
policy-value-cli eval --optimizer adamw --learning-rate 0.001 --seed 13 --device cpu --dtype float64 --deterministic --episodes 4 --epochs 3 --games 2 --rollouts 1
```

The independent frozen-weight replay uses `python scripts/replay_torch_pilot.py`. The primary visit-count experiment has separate commands, archived settings and reports. [Visit-count training](NEURAL_TICTACTOE_CONTROL.md) · [Rare-board sampling](TICTACTOE_SAMPLING.md) · [Withheld-board experiments](TICTACTOE_CHOICE_TRAINING.md)

## Replaying Checkpoint F

Use the Linux or Windows Subsystem for Linux environment and C++17 compiler described in the Phase V instructions. Install the numerical study dependencies with `python -m pip install -e ".[dev,phase-v]"`. From the repository root run `python scripts/review_checkpoint_f.py --workers 4 --output /tmp/pvi-checkpoint-f`. This replays the frozen original and both types of visit-trained network. It checks the four reconstructed players' individual external games (exact probabilities in tic-tac-toe), win/draw/loss rates and unrestricted agreement against the existing confirmation files, then computes predictions after removing the original's favorite move. [Phase V environment and data](experiments/phase_v/README.md) · [Checkpoint F review](CHECKPOINT_F.md)

The compressed observations contain one JSON record per original. Each partition retains the board identifiers, weights, legal move counts, original favorite and next choice. Each fit retains unrestricted and restricted predictions for every method and the original's external outcomes. Look up reconstructed external outcomes in the confirmation files by game, original, partition, fit and method. The summary gives original-level spread, paired differences, individual 95% intervals, and raw and multiplicity-adjusted p-values. [Observations](experiments/checkpoint_f/observations.jsonl.gz) · [Checkpoint summary](experiments/checkpoint_f/summary.json) · [Original confirmation files](experiments/win_selection/confirmation)

The follow-up asking how often fitted move frequencies can win on their own uses the same Phase V environment and archived observations. It refits the networks from board occupancy, holds out original players when choosing the rule, and records how its scores vary with outcomes and duration. [Reproduction commands and results](experiments/visit_utility/README.md#reproduction)

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
