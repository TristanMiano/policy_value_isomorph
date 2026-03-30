# REFERENCES

This Task 1 implementation uses only Python standard library features and local code (no external game/RL dependency).

## Conceptual references
- AlphaZero paper for policy/value framing and notation: https://arxiv.org/abs/1712.01815
- Project inspiration post cited by README: https://thothhermes.substack.com/p/condensed-response-to-hidden-complexity

## Why no external library yet?
For tic-tac-toe, a local environment is smaller and more inspectable than adding a third-party dependency. This reduces setup complexity and makes sign/perspective conventions explicit in code.
