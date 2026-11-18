# Policy-to-Value Reconstruction Demo

A small research/demo project exploring the following idea:

> A **policy** gives a move or probabilities for the legal moves in each position. A **value representation** assigns numerical scores to moves or their consequences. How closely can choices made from those scores reproduce the original policy, and how does playing strength change?

This repo is meant to be a concrete, economical implementation of a claim inspired by Thoth-Hermes's post "Condensed response to 'hidden complexity of wishes'", especially the suggestion that there is an approximate correspondence between a policy-like representation and a utility/value-like representation over states, actions, continuations, or trajectories.

The first experiment uses tic-tac-toe. A **rollout** plays a game forward under a specified policy to obtain its outcome. A **decoder** is the rule that uses scores to choose a move. The pilot report defines all five source players and compares their recovered moves and game outcomes. [Tic-tac-toe results](TICTACTOE_PILOT.md)

Two protocols apply throughout this project. Keep quantitative questions and answers numerical, including magnitudes and differences. Never turn a measurable comparison into a yes/no verdict. Define every new term before using it in a design, report, progress update, or chat summary. [Mandatory research protocols](AGENTS.md#mandatory-research-protocols)

---

## Core idea: two nested claims

The project now separates two questions that should not be conflated.

### 1. Behavioral reconstruction

Write `P` for the original policy, `S_P` for scores constructed to represent its choices, and `D` for the decoder that turns those scores into choices. The aim is to recover scores such that

```text
P approximately equals D(S_P).
```

A **deterministic** policy chooses one fixed move per position. Choosing the highest-scoring legal move (`argmax`) with a fixed rule for ties can reproduce it exactly. A **stochastic** policy specifies move probabilities. Taking their logarithms as scores and converting the scores back to probabilities gives an exact representation of the move distribution.

### 2. Expected-utility factorization

Measure how closely an action score that reproduces the policy can be expressed as the average utility of the outcomes of that action. Here `h` is the history, `a` a move, and `U(tau)` the numerical utility assigned to a complete course of play `tau`. The proposed form is

```text
S_P(h,a) approximately equals E[U(tau) | h, a, environment, continuation rule],
```

followed by an `argmax` over actions. The possible courses of play and their probabilities form a **lottery**. Maximizing the probability-weighted utility of its outcomes is the formulation associated with von Neumann–Morgenstern (VNM) expected utility.

The **state value** `V^P` is the expected payoff from a position when play follows `P`. The **action value** `Q^P` is the expected payoff after a specified first move and then following `P`. Either can be estimated from rollouts. Choosing the move with the highest `Q^P` may change the original player's move, even with perfectly estimated payoffs. The experiment measures how often this changes the choice and by how many payoff points.

The resulting conceptual spine is

```text
P  ->  behaviorally faithful score S_P  ->  possible trajectory utility U.
```

The first arrow is the broad policy-value representation claim. The second is a stronger utility-factorization hypothesis to test rather than assume.

See [`docs/policy_reconstruction_vs_utility_factorization.md`](docs/policy_reconstruction_vs_utility_factorization.md) for the full argument and [`docs/selection_semantics.md`](docs/selection_semantics.md) for the operational contract governing scores, rollout values, lotteries, and double counting.

---

## Important terminology: state-value vs action-value

This distinction matters.

### State-value

A **state-value** answers:

> "How good is this state, assuming we continue from here according to some policy?"

In reinforcement-learning notation:

`V^pi(s) = E[return | s, then follow pi]`

This is a scalar attached to the state.

### Action-value

An **action-value** answers:

> "How good is it to take this particular action in this state, and then continue according to some policy?"

In notation:

`Q^pi(s, a) = E[return | take action a in s, then follow pi]`

This is a scalar attached to a state-action pair.

### Their relationship

In a standard Markov setting:

`V^pi(s) = E_{a ~ pi(.|s)}[Q^pi(s, a)]`

So the state-value is the policy-weighted average of the action-values.

---

## When do `V` and `Q` coincide?

They do **not** usually coincide as objects.

They coincide only in special cases, for example:

1. **There is only one legal action.** Then there is no real distinction between the value of the state and the value of the action available in that state.
2. **You shift perspective to successor states in a deterministic environment.** If the game is deterministic and action `a` always leads from `s` to `s' = T(s, a)`, then after accounting for reward/perspective bookkeeping, `Q(s, a)` is equivalent to the value of the successor state.
3. **You redefine "state" to already include the action choice.** Then what was formerly an action-value can be represented as a state-value on an expanded state space.

In the simplest deterministic, turn-based game with no intermediate rewards, one often has something like:

`Q^pi(s, a) = V^pi(T(s, a))`

or, in a zero-sum alternating-turn game,

`Q^pi(s, a) = -V^pi(T(s, a))`

depending on whether `V` is defined from the viewpoint of the player to move, or from the viewpoint of a fixed root player.

That sign convention is crucial.

---

## When do they **not** coincide?

They differ whenever the action matters in a way not already folded into the current state value.

Typical reasons:

1. **Multiple legal actions.** Different moves can lead to very different futures.
2. **Stochastic transitions.** The same action can lead to different successor states.
3. **Averaging over policy choices.** `V^pi(s)` averages across the actions selected by the policy, while `Q^pi(s, a)` conditions on one specific action.
4. **History matters but is not fully encoded in the visible board.** In that case the true state is not just the board position.

That last point matters a lot outside tiny toy games. Tic-tac-toe is effectively Markov if the board and side-to-move are included. Chess is more subtle because repetition status, castling rights, en passant rights, and move counters matter, so the true state is richer than "piece placement alone." AlphaZero-style systems therefore encode additional rule-state information rather than just the visible board.

---

## A useful way to think about the "entire trajectory" question

If `V(s)` and `Q(s, a)` do not cleanly line up on a coarse state representation, one remedy is to **lift the state space**.

Instead of defining the state as only the current board, define it as something richer, such as:

- the full game history up to time `t`
- the full legal/rule state
- a belief state over hidden information
- or a complete trajectory prefix `h_t = (s_0, a_0, s_1, a_1, ..., s_t)`

Then a state-value over this enlarged object is often perfectly well-defined:

`V^pi(h_t) = E[return | history h_t, then follow pi]`

and the corresponding action-value is:

`Q^pi(h_t, a) = E[return | history h_t, take a, then follow pi]`

In deterministic settings, these relate cleanly via the next history:

`Q^pi(h_t, a) = r(h_t, a) + gamma * V^pi(h_t appended with a and the next state)`

So there is a real sense in which a "state-value over the whole trajectory-so-far" can absorb distinctions that would otherwise require an action-value.

Put differently:

- On a **too-small** state representation, `V` may blur distinctions that `Q` preserves.
- On a **rich enough** state representation, those distinctions can often be moved back into `V`.

For tic-tac-toe, the simple board state is already rich enough. For larger games, it may not be.

---

## Project thesis for this repo

The broad thesis is:

> Given a sufficiently well-defined policy `P`, a value-like action or continuation score plus an appropriate declared decoder can reproduce its behavior exactly or approximately.

The stronger empirical thesis is:

> Some behaviorally faithful scores can also be recovered or approximated as expected utility over trajectories under a compact, stable return contract.

The game experiment estimates `V^pi` or `Q^pi` from policy rollouts, trains a value model, and measures how closely the selected moves match the source. A model trained directly on the source's moves supplies a comparison. For a source move with lower expected payoff, measure both how often it occurs and how many payoff points it gives up.

---

## PyTorch-first migration decision (Task 22)

For the current modernization phase, this project treats **PyTorch as the canonical training backend** for policy, state-value `V(s)`, and action-value `Q(s,a)` models.

Migration constraints:

1. Keep tic-tac-toe rollout/value-recovery APIs and value-sign semantics stable.
2. Keep existing CLI workflows and artifact conventions (checkpoints, telemetry, plots) stable unless explicitly versioned.
3. Limit changes to model/training internals first (forward/backward/optimizer/checkpoint state), not environment/evaluation redesign.
4. Preserve resume semantics and reproducibility hooks (seed plumbing and deterministic settings where feasible).

Detailed migration boundaries and compatibility constraints are documented in `PYTORCH_MIGRATION_AUDIT.md`.

---

## Why start with tic-tac-toe?

Because it is small enough that we can make the experiment crisp instead of hand-wavy.

Advantages:

- tiny state space
- deterministic transitions
- easy legal-move masking
- exact or near-exact enumeration is feasible
- easy to compare against minimax ground truth
- very fast training and rollout generation

Connect Four provides a larger board on which to compare the same measurements.

---

## Recommended staged plan

### Phase 0: exact/tabular sanity check

Before using neural nets, do the simplest possible version.

1. Implement the tic-tac-toe environment.
2. Create a policy `pi`:
   - either minimax (choose the best guaranteed payoff against the opponent's best replies),
   - or a small trained policy,
   - or even a scripted suboptimal policy.
3. Enumerate or sample many reachable states.
4. Build a direct decoder-score baseline that reproduces deterministic actions by `argmax` and stochastic policies by normalized sampling.
5. For each state, run many rollouts under `pi` and estimate:
   - `V^pi(s)` from terminal outcomes, and optionally
   - `Q^pi(s, a)` for each legal action.
6. Construct the return-greedy policy using:
   - `argmax_a V^pi(T(s,a))`, or
   - `argmax_a Q^pi(s,a)`,
   with the appropriate perspective/sign convention.
7. Compare both recovered policies with the source and distinguish decoder approximation, return-estimation error, and genuine one-step policy improvement.

This phase separates the broad representation claim from the stronger expected-return factorization claim.

### Phase 1: policy network

Train or import a small policy net for tic-tac-toe.

Candidate forms:

- small MLP on flattened board state
- tiny CNN with board planes
- small AlphaZero-style network with policy head

The network should output legal-action logits or probabilities.

### Phase 2: rollout-generated value labels

Freeze the policy network.

For many sampled states `s`, generate Monte Carlo rollouts under the frozen policy and estimate:

`target_value(s) = average terminal return from s under pi`

For deterministic two-player zero-sum tic-tac-toe, use terminal labels in `{-1, 0, +1}` from a fixed player perspective.

These labels are Monte Carlo estimates of expected trajectory utility under the frozen continuation policy, not merely arbitrary pointwise scores.

### Phase 3: train value network

Train a separate network:

`V_phi(s) ~ target_value(s)`

using mean squared error (MSE), the average squared difference between predicted and target values, or another stated loss.

Optionally also train:

`Q_phi(s, a)`

for direct action-value prediction.

### Phase 4: recover policy from value

Construct a recovered value-guided policy:

`pi_V(s) = argmax_a V_phi(T(s, a))`

or

`pi_Q(s) = argmax_a Q_phi(s, a)`.

Because `V_phi` and `Q_phi` are trained toward expected rollout returns, these pointwise comparisons are intended to approximate expected-trajectory maximization. Then evaluate how closely they match the original policy `pi`, while recognizing that greedy maximization of `Q^pi` can improve rather than exactly reproduce an arbitrary `pi`.

### Phase 5: expand to Connect Four

Repeat the measurements on a larger game and report how the results change.

Why Connect Four:

- still discrete and deterministic
- still manageable on consumer hardware
- more interesting than tic-tac-toe
- more plausible place for a small CNN or AlphaZero-style setup

An RTX 5070 should be more than enough for small-scale versions of this experiment.

---

## Metrics to report

1. **Direct decoder fidelity**

   This measures how closely choices made directly from the recovered scores copy the original player. Report the fraction of matching moves for a fixed-move player and the differences between legal-move probabilities for a randomized player.

2. **Action agreement**

   Percentage of states where the return-value-guided policy chooses the same move as the original policy.

3. **Top-k agreement**

   The fraction of positions where the reference move appears among the evaluated model's `k` highest-scoring legal moves. Give `k`, the count, and the denominator.

4. **Move-probability differences**

   Total variation distance is half the sum of absolute differences between the players' legal-move probabilities. Cross-entropy is the mean negative log probability the recovered model assigns to a move drawn from the source. Kullback–Leibler (KL) divergence subtracts the source's cross-entropy with itself to give the excess on that scale.

5. **Win, draw and loss rates**

   Give each player's probabilities of winning, drawing and losing whole games. Evaluate the original and every recovered player against the same opponents as both X and O. Give each rate and its difference from the original player's rate. Also report games against the original player separately.

6. **Calibration of value predictions**

   For positions with similar predicted payoffs, compare the average prediction with the average payoff observed or calculated exactly. Give the number of observations and the difference in payoff points.

7. **Comparison with perfect play**

   The **minimax value** is the payoff a player can guarantee against an opponent that always makes its best reply. In tic-tac-toe this can be calculated exactly. Give the difference between each estimated payoff and this reference when perfect play is the intended target.

8. **Payoff forgone by a move**

   Give the best expected payoff available minus the chosen move's expected payoff, from the acting player's perspective. Specify who plays the remaining turns. Report differences caused by estimation error and differences already present with exact payoffs.

9. **Utility-factorization gap**

   Measure how much behavioral fidelity is lost when a direct policy score is constrained to arise from a declared expected-trajectory model.

---

## Key implementation cautions

### 1. Perspective and sign convention

Be explicit about whose value is being predicted.

Two common choices:

- **root-player perspective:** value is always from the viewpoint of the player whose move we are evaluating at the root
- **side-to-move perspective:** value is from the viewpoint of whoever is about to move in the current state

These lead to different formulas when comparing successor states. If this is not handled carefully, the recovered policy may systematically prefer losing moves.

### 2. Markov state definition

Make sure the state representation includes everything needed for future returns to depend only on the current state.

For tic-tac-toe, board + side-to-move is enough.

### 3. Legal move masking

Never let the policy or recovered argmax choose illegal actions.

### 4. Distribution shift

If value labels are collected only from states visited by the original policy, the value net may behave unpredictably on off-policy states. Keep evaluation focused on the same state distribution at first.

### 5. Symmetry handling

For tic-tac-toe and Connect Four, symmetry reduction or augmentation may improve sample efficiency a lot.

### 6. Value semantics and double counting

Record whether a learned value is a decoder score, instantaneous utility, terminal utility, or expected continuation return. Do not average or sum continuation values as though they were instantaneous rewards when the rollout expectation is already included.

### 7. Reconstruction versus optimization

Do not infer exact policy recovery merely because `Q^pi` is accurately estimated. Greedy expected-return selection can differ from the policy whose rollouts generated the labels.

### 8. Decoder matching

Do not judge reconstruction of a stochastic policy only by the `argmax` action. The decoder must reproduce the policy distribution when distributional behavior is part of the target.

---

## Minimal repo structure

Suggested initial files:

- `README.md` — project overview and plan
- `env.py` — tic-tac-toe environment wrapper and helpers
- `policy.py` — policy network and/or scripted policy
- `generate_rollouts.py` — create value targets from frozen policy
- `train_value.py` — train the value network
- `evaluate.py` — compare original and recovered policies
- `docs/selection_semantics.md` — value meanings and selection mechanisms
- `docs/policy_reconstruction_vs_utility_factorization.md` — full theoretical distinction between imitation and consequential factorization
- `notebooks/` or `plots/` — visualizations and diagnostics

---

## Suggested first milestone

A good first milestone is:

- build tic-tac-toe state encoding
- implement a decent policy
- construct an exact direct decoder-score baseline
- estimate `V^pi(s)` for every reachable state or a very large fraction of them
- train a small value net
- show how often greedy expected-return selection via `V^pi(T(s,a))` matches the original policy
- separate direct reconstruction error, value-estimation error, and genuine greedy-improvement differences

The resulting percentages and payoff differences quantify both forms of recovery on the same boards.

---

## Possible extensions

1. **Compare decoder-score, `V`, and direct `Q` reconstruction.**  
   This separates behavioral imitation from successor-state and action-return recovery.

2. **Use intentionally imperfect policies.**  
   This can show when greedy use of recovered return value reproduces a policy and when it improves or departs from it.

3. **Measure how reconstruction quality changes with rollout budget.**  
   This connects directly to the "sample many trajectories" idea.

4. **Use learned embeddings.**  
   Instead of valuing only raw board states, value a latent representation induced by the policy net.

5. **Move beyond perfect-information games.**  
   This would force the repo to confront the trajectory/history and lottery issues more directly.

6. **Test lottery preferences explicitly.**  
   Compare choices among induced trajectory distributions with the expectations predicted by a recovered cardinal value artifact.

7. **Measure utility-factorization complexity.**  
   Ask how simple, stable, and transferable a trajectory utility can remain while preserving the direct decoder's behavioral fidelity.

---

## What the evidence measures

The mathematical representation question calls for an explicit construction and proof under stated assumptions. For learned scores, report move agreement or move-probability distance, differences in action rankings, and variation across independent recoveries.

For values learned from game outcomes, report prediction error, the frequency and payoff cost of deviations, and the change in win, draw, and loss probabilities. To compare an explanation based on consequences with direct imitation, give its size, prediction error on unseen choices, and the variation across fits. Record the population, weights, and units for each comparison.

---

## Background / inspiration

- Thoth-Hermes, *Condensed response to "hidden complexity of wishes"*  
  https://thothhermes.substack.com/p/condensed-response-to-hidden-complexity

- Silver et al., *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* (AlphaZero)  
  https://arxiv.org/abs/1712.01815

- McIlroy-Young et al., *Amortized Planning with Large-Scale Transformers: A Case Study on Chess*  
  https://arxiv.org/abs/2402.04494

---

## Notes for Codex

If Codex is asked to continue from this README, it should begin with the smallest robust version of the project:

1. implement tic-tac-toe
2. define a clean state encoding and transition function
3. choose a policy representation
4. implement a direct decoder-score baseline
5. generate rollout-based expected-return labels from a frozen policy
6. train a value net
7. construct a value-guided policy by greedily maximizing the learned expected return
8. evaluate decoder fidelity, agreement, greedy-improvement differences, calibration, and mistakes

Prefer a simple, correct version over a flashy one. The first target is a reproducible tic-tac-toe demo, not a giant training stack.

---

## Reproducibility

For experiment settings (seeds, split protocol, baseline CLI commands, and reporting checklist), see `REPRODUCIBILITY.md`.

## Task 1 status (implemented)

The repository now contains a runnable minimal vertical slice for tic-tac-toe:

- canonical tic-tac-toe state (`board` + `to_move`)
- legal move generation, transition, terminal and winner detection
- deterministic baseline policy (win/block/center/corners)
- rollout-based `V^pi(s)` estimation from a fixed root-player perspective
- recovered policy via successor-state expected-value comparison (max for root turn, min for opponent turn)
- pytest coverage for game logic, value sign behavior, and recovered action behavior
- runnable demo script printing sample states, values, and chosen actions

### Quickstart

```bash
pip install -e .[dev]
pytest -q
python scripts/demo_tictactoe.py
```
