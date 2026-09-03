# DESIGN (Task 1 Vertical Slice)

## Scope

This first slice intentionally implements only a **tabular/deterministic tic-tac-toe demo**:

- a fixed baseline policy,
- rollout-based recovery of `V^pi(s)`, and
- action choice from rollout-trained successor-state values.

No neural nets are included yet.

## Value representation does not determine selection by itself

A real-valued `V` or `Q` can be consumed by different selection mechanisms. A direct decoder is

```text
pi_score(s) in argmax_a S(s,a)
```

or, in a deterministic environment,

```text
pi_score(s) in argmax_a S(T(s,a)).
```

This is a broad policy-representation mechanism. An arbitrary decoder score can reproduce behavior without necessarily being a cardinal utility over lotteries.

For a stronger utility-agent interpretation, actions should instead be evaluated through the distributions over continuations or complete trajectories that they induce:

```text
pi_EV(h) in argmax_a E[U(tau) | h, a, environment, continuation rule].
```

Here `U(tau)` is a declared cardinal utility over complete trajectories. This expected-trajectory selector is the preferred semantics for claims that an agent acts as a utility function and is the form most directly related to VNM expected utility.

See [`docs/selection_semantics.md`](docs/selection_semantics.md) for the full contract and caveats.

## Why the current successor-state comparison already approximates expected-trajectory selection

The current `V^pi` is estimated from average rollout returns:

```text
V^pi(s) = E[U(tau) | start at s, then follow pi].
```

Likewise, an action value is

```text
Q^pi(s,a) = E[U(tau) | take a in s, then follow pi].
```

These values already contain an expectation over policy continuations. Therefore, choosing an action by comparing `Q^pi(s,a)`, or an equivalent successor-state `V^pi`, is already an approximate argmax over expected trajectory utility. It does not require another outer rollout average unless transition or continuation uncertainty has not yet been integrated into the supplied value.

Do not sum a continuation value again at every step as though it were an instantaneous reward; that would usually count future consequences repeatedly.

## State-value vs action-value

- **State-value**: `V^pi(s) = E[return | start at s, then follow pi]`.
- **Action-value**: `Q^pi(s,a) = E[return | take a in s, then follow pi]`.

`V` is attached to states; `Q` is attached to state-action pairs.

## When `Q(s,a)` and `V(T(s,a))` coincide

In deterministic turn-based games with no intermediate reward, if action `a` from state `s` leads to successor `s' = T(s,a)`, then, under a fixed perspective convention, `Q^pi(s,a)` is numerically equal to the appropriately signed value of `s'`.

In this repo we define value from a **fixed root-player perspective**. Therefore:

- if the root player acts, use `argmax_a V(T(s,a))`;
- if the opponent acts, use `argmin_a V(T(s,a))`.

This handles zero-sum sign behavior explicitly. Operationally, the current direct successor-state comparison is an expected-trajectory selector because each successor value was itself recovered from rollout returns.

## Reconstruction versus greedy policy improvement

A decoder score may be constructed to reproduce a policy by design. A rollout return value induced by a policy has different semantics:

```text
pi(s) need not equal argmax_a Q^pi(s,a).
```

The right-hand side is the one-step greedy improvement of `pi` under the declared payoff, environment, and continuation rule. Matching the source policy is therefore an empirical result. Disagreement can reflect value-estimation error, insufficient state, stochasticity, or genuine failure of the source policy to greedily maximize its own induced expected return.

Evaluation should eventually distinguish:

1. exact or learned decoder-score reconstruction;
2. estimation error in `V^pi` or `Q^pi`; and
3. genuine greedy-improvement differences from the source policy.

## When path/history matters

`V(s)` only works cleanly when `s` is Markov and contains all information relevant to future trajectory utility.

- Tic-tac-toe is Markov with `(board, side_to_move)`.
- In larger games, hidden, rule, or history state can matter, so the representation may need to be enlarged—for example, by including repetition counters, castling rights, belief state, or the full trajectory prefix.

The consequence space used for expected utility may likewise be richer than terminal board labels: an agent can value complete trajectories, including path-dependent distinctions.

## What is deferred to later tasks

1. Train a tic-tac-toe **policy network**.
2. Freeze the policy net and generate larger Monte Carlo datasets.
3. Train a **value network** `V_phi` from expected rollout-return labels.
4. Compare greedy expected-return selection through `V_phi(T(s,a))` with the original policy over large state sets.
5. Separate reconstruction error from genuine one-step policy improvement.
6. Extend the experiment to **Connect Four** and later stochastic environments.
7. Test when recovered values support a stable cardinal extension to explicit lotteries, rather than only an argmax decoder.
