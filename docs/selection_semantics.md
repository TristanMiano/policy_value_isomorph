# Selection Semantics, Rollout Value, and Lotteries

The common symbols, score units and fields describing a recovered model are defined in the [notation and recovery contract](../formalism/00_notation.md).

A **policy** is a rule for choosing actions. A **score** assigns a number to an action or its consequences. A **decoder** converts the scores into a chosen action or probabilities for the available actions. When reporting a value, give both its meaning and the decoder.

This project distinguishes two closely related but importantly different uses of value.

## 1. Direct score decoding

Write `h` for the current history, `C(h)` for the available actions and `a` for one action. The score `S` can use the history and action, or the resulting position `T(h,a)`. The policy `pi_score` chooses a highest-scoring action, written `argmax`. The first form is

```text
pi_score(h) in argmax_{a in C(h)} S(h,a)
```

or, in a deterministic environment,

```text
pi_score(h) in argmax_{a in C(h)} S(T(h,a)).
```

This is a broad policy-representation mechanism. With a sufficiently expressive, context-sensitive score, many policies can be represented this way, including policies whose scores have no established interpretation as cardinal utility over uncertain outcomes.

Choosing the largest score gives an exact reference for copying a fixed-action policy. The **von Neumann–Morgenstern (VNM) expected-utility** formulation additionally gives each possible consequence a numerical utility and weights it by its probability when evaluating a move.

## 2. Expected-trajectory selection

For a stronger agent or utility interpretation, an action should be evaluated by the lottery over continuations or complete trajectories that it induces.

Let an action `a`, environment `M`, and declared continuation rule `kappa` induce a distribution over trajectories `tau`. Let `U(tau)` be a cardinal utility assigned to a complete trajectory. The expected-trajectory selector is

```text
pi_EV(h) in argmax_{a in C(h)} E[U(tau) | h, a, M, kappa].
```

A common special case is discounted additive trajectory utility:

```text
U(tau) = sum_{t >= 0} gamma^t u(h_t, a_t).
```

The utility may instead be terminal-only or depend nonadditively on the entire path. Treating actions as lotteries over complete trajectories and maximizing expected trajectory utility is the project’s preferred default when making a substantive rational-agent interpretation. It is also the form most directly related to VNM expected utility.

The expected-value selector is not, on its own, proof that an observed policy satisfies all VNM axioms. It specifies the behavioral model that would make the value artifact VNM-like: one cardinal utility over the relevant consequences, extended linearly to lotteries and maximized in expectation.

## When rollout-trained `V` or `Q` already contains the expectation

The distinction above can disappear operationally when the learned value is already a return value. For a source policy `P`, define

```text
V^P(h)   = E[U(tau) | h, then follow P]
Q^P(h,a) = E[U(tau) | h, take a, then follow P].
```

If a network is trained on mean Monte Carlo rollout returns, then `V_phi` or `Q_phi` is intended to approximate one of these expectations. In that case,

```text
argmax_a Q_phi(h,a)
```

is already an approximate argmax over expected trajectory utility. It does not require another outer rollout average merely to become an expected-value selector.

In a deterministic transition system with immediate utility or reward `r`,

```text
Q^P(h,a) = r(h,a) + gamma * V^P(T(h,a)),
```

up to the project’s perspective and sign conventions. Thus comparing rollout-trained successor-state values can itself be an implementation of expected-trajectory selection.

For stochastic transitions, the corresponding relationship is

```text
Q^P(h,a) = E_{h' | h,a}[r(h,a,h') + gamma * V^P(h')].
```

A selector must average over successor uncertainty unless the supplied `Q` estimate already performs that averaging.

## Do not count continuation value twice

If `V^P` or `Q^P` already denotes expected continuation return, it should not normally be summed again at every step of a sampled trajectory. For example,

```text
sum_t gamma^t Q^P(h_t,a_t)
```

usually counts later consequences repeatedly. Either:

1. define trajectory utility from instantaneous utilities/rewards and then take its expectation; or
2. use the root `V^P` or `Q^P`, which already summarizes that expected continuation.

## Policy reconstruction versus greedy improvement

An exact decoder score can be constructed to reproduce a source policy `P` by design. A return value induced by `P` has different semantics and need not reproduce `P` under greedy selection:

```text
P(h) need not equal argmax_a Q^P(h,a).
```

The right-hand side selects a move with the highest expected payoff and then assumes the source plays the remaining turns. Report how often its selected move matches the source, the frequency or probability of a different equally good move, and the probability of a strictly lower-payoff move. For the size of the difference, subtract each chosen move's expected payoff from the best available payoff for the acting player and average over moves and positions. Give the population and its weights. For a randomized source also give the difference between the two move distributions.

The project should report direct decoder reconstruction separately from expected-return greedy selection whenever both are available.

## Required metadata for a value artifact

A value artifact intended for action selection should declare:

- whether its numbers are decoder scores, instantaneous utilities, terminal utilities, or expected returns;
- the consequence space: states, state-action pairs, trajectory prefixes, or complete trajectories;
- the environment or transition model;
- the continuation policy or planning rule;
- horizon, discounting, terminal payoff, and perspective conventions;
- whether transition and rollout expectations are already folded into the value;
- the decoder or selector used to produce behavior.

The project’s default hierarchy is therefore:

```text
pointwise argmax score decoder
    -> rollout-induced expected return value
    -> expected-trajectory maximization
    -> tested preferences over lotteries / VNM-style interpretation.
```

Pointwise argmax remains the broad representation baseline. Expected-trajectory maximization is the preferred semantics for stronger claims about an agent acting as a utility function.
