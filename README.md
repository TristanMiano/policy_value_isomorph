# Policy-to-Value Reconstruction Demo

A small research/demo project exploring the following idea:

> If we start from an agent's **policy**—a mapping from states to moves or next states—can we reconstruct a useful **value representation** from that policy alone, and then use that recovered value representation to choose moves by argmax?

This repo is meant to be a concrete, economical implementation of a claim inspired by Thoth-Hermes's post "Condensed response to 'hidden complexity of wishes'", especially the suggestion that there is an approximate correspondence between a policy-like representation and a utility/value-like representation over states.

The immediate practical goal is **not** to settle the philosophical claim in full generality. The goal is narrower and empirical:

1. Start with a known policy for a small discrete game.
2. Freeze that policy.
3. Use rollouts of the policy to estimate a value function induced by that policy.
4. Train a value network on those estimates.
5. Recover move choice by taking an argmax over successor-state values.
6. Measure how closely the recovered value-guided agent matches the original policy.

---

## Core idea

We begin with a policy-like object:

- `P(s) -> a`, a deterministic move chooser, or
- `P(s) -> pi(.|s)`, a distribution over legal actions.

In a deterministic board game, the chosen action `a` induces a next state `T(s, a)`.

What we want to recover is a value-like object:

- `V(s) -> real`, a scalar estimate of how good state `s` is, or
- `Q(s, a) -> real`, a scalar estimate of how good it is to take action `a` in state `s`.

Once we have a value function, we can choose a move by comparing the values of successor states:

`a* = argmax_a V(T(s, a))`

or, equivalently,

`a* = argmax_a Q(s, a)`

when a `Q` function is available.

The main experiment in this repo is to see how much of the original policy can be reconstructed this way.

---

## Important terminology: state-value vs action-value

This distinction matters.

### State-value

A **state-value** answers:

> "How good is this state, assuming we continue from here according to some policy?"

In reinforcement-learning notation:

`V^pi(s) = E[ return | s, then follow pi ]`

This is a scalar attached to the state.

### Action-value

An **action-value** answers:

> "How good is it to take this particular action in this state, and then continue according to some policy?"

In notation:

`Q^pi(s, a) = E[ return | take action a in s, then follow pi ]`

This is a scalar attached to a state-action pair.

### Their relationship

In a standard Markov setting:

`V^pi(s) = E_{a ~ pi(.|s)} [ Q^pi(s, a) ]`

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

Yes: if `V(s)` and `Q(s, a)` do not cleanly line up on a coarse state representation, one remedy is to **lift the state space**.

Instead of defining the state as only the current board, define it as something richer, such as:

- the full game history up to time `t`
- the full legal/rule state
- a belief state over hidden information
- or a complete trajectory prefix `h_t = (s_0, a_0, s_1, a_1, ..., s_t)`

Then a state-value over this enlarged object is often perfectly well-defined:

`V^pi(h_t) = E[ return | history h_t, then follow pi ]`

and the corresponding action-value is:

`Q^pi(h_t, a) = E[ return | history h_t, take a, then follow pi ]`

In deterministic settings, these relate cleanly via the next history:

`Q^pi(h_t, a) = r(h_t, a) + gamma * V^pi(h_t appended with a and the next state)`

So there is a real sense in which a "state-value over the whole trajectory-so-far" can absorb distinctions that would otherwise require an action-value.

Put differently:

- On a **too-small** state representation, `V` may blur distinctions that `Q` preserves.
- On a **rich enough** state representation, those distinctions can often be moved back into `V`.

For tic-tac-toe, the simple board state is already rich enough. For larger games, it may not be.

---

## Project thesis for this repo

We will test the following concrete claim:

> Given a frozen policy `pi`, a value function induced by that policy can be estimated from rollouts, trained into a value network, and used to recover much of the original move behavior by argmax over successor-state values.

This is intentionally narrower than "all policies and all utilities are equivalent." It is a tractable, empirical version of the broader idea.

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

If the idea works there, we can move to Connect Four as the next step.

---

## Recommended staged plan

### Phase 0: exact/tabular sanity check

Before using neural nets, do the simplest possible version.

1. Implement tic-tac-toe environment.
2. Create a policy `pi`:
   - either minimax,
   - or a small trained policy,
   - or even a scripted suboptimal policy.
3. Enumerate or sample many reachable states.
4. For each state, run many rollouts under `pi` and estimate:
   - `V^pi(s)` from terminal outcomes, and optionally
   - `Q^pi(s, a)` for each legal action.
5. Reconstruct action choice using:
   - `argmax_a V(T(s,a))`, or
   - `argmax_a Q(s,a)`.
6. Compare reconstructed decisions to the original policy.

This phase tells us whether the basic idea works at all.

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

### Phase 3: train value network

Train a separate network:

`V_phi(s) ~ target_value(s)`

using MSE or another simple regression loss.

Optionally also train:

`Q_phi(s, a)`

for direct action-value prediction.

### Phase 4: recover policy from value

Construct a recovered policy:

`pi_V(s) = argmax_a V_phi(T(s, a))`

or

`pi_Q(s) = argmax_a Q_phi(s, a)`

Then evaluate how closely these match the original policy `pi`.

### Phase 5: expand to Connect Four

If the tic-tac-toe demo works, repeat the same experiment on a slightly larger game.

Why Connect Four:

- still discrete and deterministic
- still manageable on consumer hardware
- more interesting than tic-tac-toe
- more plausible place for a small CNN or AlphaZero-style setup

An RTX 5070 should be more than enough for small-scale versions of this experiment.

---

## Metrics to report

The repo should not just produce a cool picture; it should measure the reconstruction.

Recommended metrics:

1. **Action agreement**  
   Percentage of states where the recovered value-guided policy chooses the same move as the original policy.

2. **Top-k agreement**  
   Whether the recovered move lies in the original policy's top-k actions.

3. **KL divergence / cross-entropy**  
   If the original policy is stochastic.

4. **Win rate / draw rate**  
   Play the recovered policy against the original policy.

5. **Calibration of value predictions**  
   Compare `V_phi(s)` against empirical rollout returns.

6. **Comparison to ground-truth minimax value**  
   Especially useful for tic-tac-toe.

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

---

## Minimal repo structure

Suggested initial files:

- `README.md` — project overview and plan
- `env.py` — tic-tac-toe environment wrapper and helpers
- `policy.py` — policy network and/or scripted policy
- `generate_rollouts.py` — create value targets from frozen policy
- `train_value.py` — train the value network
- `evaluate.py` — compare original and recovered policies
- `notebooks/` or `plots/` — visualizations and diagnostics

---

## Suggested first milestone

A good first milestone is:

- build tic-tac-toe state encoding
- implement a decent policy
- estimate `V^pi(s)` for every reachable state or a very large fraction of them
- train a small value net
- show that `argmax_a V(T(s,a))` matches the original policy on a large share of states

If that works, the repo already demonstrates the core idea in a real, inspectable way.

---

## Possible extensions

1. **Compare `V` reconstruction to direct `Q` reconstruction.**  
   This directly tests whether action-values are easier to recover than successor-state values.

2. **Use intentionally imperfect policies.**  
   This can show that the method works for arbitrary policies, not only optimal ones.

3. **Measure how reconstruction quality changes with rollout budget.**  
   This connects directly to the "sample many trajectories" idea.

4. **Use learned embeddings.**  
   Instead of valuing only raw board states, value a latent representation induced by the policy net.

5. **Move beyond perfect-information games.**  
   This would force the repo to confront the trajectory/history issue more directly.

---

## What success would look like

This project is successful if it shows something like the following:

- a policy can be learned or imported for a small discrete game
- a value function induced by that policy can be estimated from rollouts
- a value network can learn that induced value function reasonably well
- a policy recovered from the value network reproduces a substantial fraction of the original policy's choices

That would not prove a universal philosophical isomorphism. But it would provide a concrete working demonstration of a nontrivial bridge from policy-like behavior to value-like representation.

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
4. generate rollout-based value labels from a frozen policy
5. train a value net
6. reconstruct the policy by argmax over successor-state values
7. evaluate agreement and visualize mistakes

Prefer a simple, correct version over a flashy one. The first target is a reproducible tic-tac-toe demo, not a giant training stack.
