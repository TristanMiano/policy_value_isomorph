# AGENTS.md

## Scope
These instructions apply to the entire repository.

## Mandatory research protocols

1. **Keep every quantitative comparison quantitative.** From the initial question through the saved data and the final summary, ask how much, how often, by how many, or how the result changes. Give the measured values, their differences, units, populations and denominators or weights. Report uncertainty when observations were sampled. Never turn a measurable comparison into a yes/no question, introduce a success cutoff, or attach pass/fail, sufficient/insufficient, or equivalent verdicts to research measurements. Individual moves can match or differ and individual games can end in wins, draws or losses. Report their counts and rates. Software tests may verify a calculation or a file exactly.
2. **Define every new term before using it.** Introduce each name, abbreviation, symbol, method, comparison player and measure with a plain-language definition before using it to explain a plan or interpret a result. Give the actual rule for each named policy (how the player chooses a move), its role, and the meaning and direction of each measurement. A glossary or source link can supplement the definition at first use.

Both rules are mandatory for the whole project, including study designs, task descriptions, code-generated result tables and their field descriptions, research reports, progress updates, and all chat replies and summaries. Each standalone communication must be understandable without a previous chat message.

For neural tic-tac-toe comparisons with minimax (the best outcome a player can guarantee against perfect replies), first train a neural source and check its moves on every reachable decision board. Report the number preserving that outcome, the payoff given up, and agreement with the specific rule for breaking ties. Use an optimal neural source for the optimal-play comparison. Describe weaker sources by their actual training target. Give the architecture, training data, optimization settings, and coverage for both source and reconstruction.

Compare a board-scoring network (V) and a board-and-move-scoring network (Q) under both direct move selection and the von Neumann–Morgenstern (VNM) expected-utility formulation. For the latter, define the learned utility of a complete trajectory, the environment, continuation policy, perspective, and rule for ties. For each move take the expectation of that same utility over possible continuations. Report move fidelity and game outcomes side by side, and disclose access to the source during selection. A fixed win/draw/loss payoff provides an additional reference.

Label expected-utility selection explicitly as **VNM** in reporting tables. Identify the utility and which player continues each rollout when those vary across rows.

In the primary visit-count experiment, train and freeze the source first. Sample complete games, retain transition counts and their denominators, and specify how games start. Fit the value networks to normalized observed transition counts. Compare direct selection with sampled expected-utility selection using the same fitted scores. Record the continuation rule for both players, rollout budget, whole-game utility and time decay. Show how many distinct boards were observed. Report results obtained from game-outcome training as a separate experiment.

For visit-trained networks, give the loss actually optimized, its units and weighting, the normalization of the target counts, and the measured training loss at the checkpoint reported. When diagnosing residual disagreements, show legal-move predictions and observed source counts for example positions, stating whether each was fitted or withheld. Compare predicted and observed counts at the same number of visits.

## Current project scope

Keep the implementation small and inspectable. The primary experiment freezes a trained tic-tac-toe player, measures its moves, and trains networks on those observations. The game-payoff baseline plays complete games under a fixed player and predicts the resulting points. PyTorch provides the three network trainers. The roadmap schedules comparisons in Connect Four and four-by-four Reversi. [Current task and scope](TODO.md#resume-here)

## Structure
- Python package code lives in `src/policy_value_isomorph/`.
- Tests live in `tests/`.
- Runnable scripts live in `scripts/`.
- Design and planning notes live in markdown files at repo root.

## Commands

From the repository root, use `python -m pip install -e ".[dev]"` for an editable test installation. Run `python -m pytest -q` for the suite, `python scripts/demo_tictactoe.py` for sample moves, and `python scripts/smoke_torch_training.py` to exercise training and continuation from saved weights. [Reproduction guide](REPRODUCIBILITY.md)

## Verification rules
Before finalizing changes:
1. Run `pytest -q` and confirm all tests pass.
2. Run `python scripts/demo_tictactoe.py` and confirm it prints sample states, recovered values, and chosen actions.
3. Keep dependencies minimal and documented.

## Coding guidelines
- Prefer explicit pure functions and small dataclasses.
- Keep value-sign conventions explicit in docstrings.
- Use type hints.
