# AGENTS.md

## Scope
These instructions apply to the entire repository.

## Project goals (Task 1)
- Keep the implementation minimal, correct, and easy to inspect.
- Focus only on tic-tac-toe and rollout-based value recovery from a fixed policy.
- Do not introduce neural-network training yet unless explicitly requested.

## Structure
- Python package code lives in `src/policy_value_isomorph/`.
- Tests live in `tests/`.
- Runnable scripts live in `scripts/`.
- Design and planning notes live in markdown files at repo root.

## Commands
- Install package (editable): `pip install -e .`
- Run tests: `pytest -q`
- Run demo: `python scripts/demo_tictactoe.py`

## Verification rules
Before finalizing changes:
1. Run `pytest -q` and confirm all tests pass.
2. Run `python scripts/demo_tictactoe.py` and confirm it prints sample states, recovered values, and chosen actions.
3. Keep dependencies minimal and documented.

## Coding guidelines
- Prefer explicit pure functions and small dataclasses.
- Keep value-sign conventions explicit in docstrings.
- Use type hints.
