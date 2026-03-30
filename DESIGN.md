# DESIGN (Task 1 Vertical Slice)

## Scope
This first slice intentionally implements only a **tabular/deterministic tic-tac-toe demo**:
- a fixed baseline policy,
- rollout-based recovery of `V^pi(s)`, and
- action choice from successor-state values.

No neural nets are included yet.

## State-value vs action-value
- **State-value**: `V^pi(s) = E[return | start at s, then follow pi]`.
- **Action-value**: `Q^pi(s,a) = E[return | take a in s, then follow pi]`.

`V` is attached to states; `Q` is attached to state-action pairs.

## When `Q(s,a)` and `V(T(s,a))` coincide
In deterministic turn-based games with no intermediate reward, if action `a` from state `s` leads to successor `s' = T(s,a)`, then:
- under a fixed perspective convention, `Q^pi(s,a)` is numerically equal to the appropriately signed value of `s'`.

In this repo we define value from a **fixed root-player perspective**. Therefore:
- if root player acts, use `argmax_a V(T(s,a))`.
- if opponent acts, use `argmin_a V(T(s,a))`.

This handles zero-sum sign behavior explicitly.

## When path/history matters
`V(s)` only works cleanly when `s` is Markov (contains all relevant information).
- Tic-tac-toe is Markov with `(board, side_to_move)`.
- In larger games, hidden/rule/history state can matter, so the representation may need to be enlarged (e.g., include repetition counters, castling rights, etc.).

## What is deferred to later tasks
1. Train a tic-tac-toe **policy network**.
2. Freeze policy net and generate larger Monte Carlo datasets.
3. Train a **value network** `V_phi` from rollout labels.
4. Compare recovered `argmax_a V_phi(T(s,a))` against original policy net over large state sets.
5. Extend the experiment to **Connect Four**.
