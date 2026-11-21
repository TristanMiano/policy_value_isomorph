# Policy Reconstruction versus Expected-Utility Factorization

For common definitions of histories, scores and whole-play utility, see the [notation and recovery contract](../formalism/00_notation.md).

## The question

A **policy** `P` is the original rule for choosing an action or assigning probabilities to the available actions. A **score** is a number attached to an action or its consequences. A **decoder** turns scores into actions or action probabilities. The project asks how closely a score and decoder together can reproduce a policy's choices.

> Given an arbitrary policy `P`, recover a value-like object together with a declared selection mechanism that approximately reproduces `P`.

This raises an important design question: which selection mechanism should be expected to reproduce an arbitrary `P` most faithfully?

The answer depends on the target claim.

- **Behavioral reconstruction** measures how closely the recovered choices match the original. For a policy that always makes one fixed choice, assign that choice the largest score and select it (`argmax`). For a policy that draws actions at random, recover the original move probabilities and sample from them.
- **Von Neumann–Morgenstern (VNM) expected utility** assigns each possible complete outcome a numerical utility. An action's **lottery** is the set of possible outcomes and their probabilities. Choose an action with the greatest probability-weighted average utility.
- A **rollout** is a simulated continuation under a specified policy. The average payoff from a position (its state value, `V`) or after a specified first action (its action value, `Q`) can be estimated from rollouts. The action with the greatest expected payoff can differ from the original choice.

The project should therefore treat universal policy reconstruction and expected-utility factorization as two related but distinct research problems.

---

## 1. The universal reconstruction form

Let

```text
h in H                     history or sufficiently rich state
C(h) subset A              feasible candidate actions
P(. | h, C)                source policy
S_P(h,C,a) in extended R   fitted or constructed move score, with larger values preferred
D(S_P, h, C)               declared selection mechanism
```

The broadest representation thesis is

```text
P(. | h, C) approximately equals D(S_P, h, C).
```

The value-like object and its decoder form a pair. A bare scalar function does not determine behavior until the project specifies whether it is consumed by `argmax`, softmax, sampling from an energy model, thresholding, or another rule.

Choose a decoder that can express the target, a move or a distribution over moves. For a fitted model, measure fidelity on histories withheld from fitting and record access to the source.

---

## 2. Deterministic policies: direct `argmax` can be exact

Suppose `P` is deterministic. On any finite feasible action set, define

```text
S_P(h,a) = 1  if a = P(h)
           0  otherwise.
```

Then, with a fixed tie-breaking convention,

```text
P(h) = argmax_{a in C(h)} S_P(h,a).
```

This representation is exact. It places no rationality, optimality, temporal consistency, or consequentialist assumptions on `P`. The policy can be path-dependent, locally inconsistent under a coarser state description, deliberately suboptimal under an external reward, or otherwise arbitrary.

Many other exact scores are possible. Giving the chosen action a strictly greater score is sufficient. A tie can reproduce the same choice when the specified tie rule selects it. This nonuniqueness is why the base artifact is better understood as a decoder-relative value or score equivalence class rather than a uniquely identified utility.

For a deterministic policy on a metric action space, an analogous construction is

```text
S_P(h,a) = -d(a, P(h)),
```

provided the metric separates actions and the decoder can identify the unique maximizer.

### Why action value is the broad primitive

A scalar state value `V(h)` alone is not universally sufficient. Two actions may lead to the same represented successor state, transitions may be stochastic, or the action itself may matter independently of its successor. The broad primitive should therefore usually be an action or continuation score

```text
S_P(h,a)  or  Q_P(h,a).
```

Recording the history and forced action together retains which choice was made. When only the resulting state is available, histories can merge. The score comparisons imposed by all parents must be compatible with one assignment and the declared tie rule.

---

## 3. Stochastic policies: reproduce the distribution, not only its mode

If `P` is stochastic, an `argmax` decoder generally reproduces only its modal action. To reconstruct the entire policy distribution, the decoder must itself be stochastic.

For a discrete policy with positive probabilities and equal base weights, define

```text
S_P(h,a) = T * log P(a | h) + c(h),
```

where `T > 0` is the temperature controlling the spread of sampled choices and `c(h)` is any state-dependent additive constant. Then

```text
softmax(S_P(h,.) / T) = P(. | h).
```

Zero-probability actions can be represented by `-infinity` or approximated with sufficiently negative finite scores.

Thus the exact stochastic counterpart of deterministic score-plus-`argmax` is

```text
log score + exponential sampling at the declared temperature.
```

For a continuous stochastic policy with density `p(a | h)` relative to a declared base measure, the score on positive density is `T log p(a | h) + c(h)`. Energy is its negative. Sampling weights that measure by the exponential of the score divided by temperature.

The behavioral reconstruction objective should therefore match the policy type:

```text
deterministic P:  action agreement or classification loss, then argmax
stochastic P:     distributional loss such as KL/cross-entropy, then sampling
```

---

## 4. Expected-trajectory maximization is a stronger factorization

A VNM-style selector treats each action as inducing a lottery over consequences. In a sequential environment, the consequences may be complete trajectories.

Let an action `a`, environment `M`, and continuation rule `kappa` induce a distribution over trajectories `tau`. Let `U(tau)` be a cardinal utility over complete trajectories. Define

```text
Q_U^kappa(h,a) = E[U(tau) | h, a, M, kappa].
```

The expected-utility policy is

```text
P_U(h) in argmax_{a in C(h)} Q_U^kappa(h,a).
```

This is still an `argmax`; the stronger claim lies in the internal structure of the score. Instead of an arbitrary decoder score, the action value must factor through an expectation over induced consequences:

```text
action
  -> distribution over trajectories
  -> cardinal trajectory utility
  -> expectation
  -> argmax.
```

A discounted additive return is one special case:

```text
U(tau) = sum_{t >= 0} gamma^t u(h_t,a_t).
```

Discounting is not required by VNM theory itself. VNM expected utility requires linear expectation over lotteries of consequences. The utility of a complete trajectory may be terminal-only, path-dependent, nonadditive across time, or otherwise structured. Exponential discounting adds further assumptions such as temporal separability and a particular form of time preference.

---

## 5. Why expected return need not reconstruct an arbitrary policy

Suppose a declared trajectory utility `U` is fixed independently of the source policy. Evaluate the consequences of taking each action once and then returning to `P`:

```text
Q_U^P(h,a) = E[U(tau) | h, take a, then follow P].
```

Greedy selection gives

```text
G_U(P)(h) in argmax_a Q_U^P(h,a).
```

This is generally a one-step policy improvement operator, not a policy-reconstruction operator.

A one-step counterexample is enough. Let `P(h) = A`, and let both actions terminate with

```text
U(A) = 0
U(B) = 1.
```

A direct decoder value can reproduce `P` exactly:

```text
S_P(h,A) = 1
S_P(h,B) = 0.
```

But return evaluation gives

```text
Q_U^P(h,A) = 0
Q_U^P(h,B) = 1,
```

so greedy expected-return selection chooses `B` rather than `A`.

Nothing has failed in the return-value estimate. It has correctly identified that the source policy chooses the lower-utility action under the declared payoff. The mismatch is semantic: policy evaluation plus greedification asks which one-step deviation has the best consequence, whereas direct policy reconstruction asks which action `P` actually chooses.

The exact agreement condition is

```text
P(h) in argmax_a Q_U^P(h,a)
```

for every relevant history. This says that `P` is already greedy, or tied for greedy, with respect to the declared trajectory utility and continuation contract.

For an arbitrary `P`, there is no reason for this condition to hold. For a policy trained successfully against the same return contract, it may hold approximately.

---

## 6. The rollout-trained-value subtlety

The repository often recovers value by running the source policy and averaging rollout returns. Such a value is not merely a pointwise decoder score. Its intended semantics are already

```text
V_U^P(h)   = E[U(tau) | h, then follow P]
Q_U^P(h,a) = E[U(tau) | h, take a, then follow P].
```

Therefore,

```text
argmax_a Q_U^P(h,a)
```

is already an `argmax` over expected trajectory utility. It does not need another outer average over rollouts merely to become an expected-value selector.

For an additive utility with immediate payoff `r`, discount `gamma` and deterministic successor `M(h,a)`, the recurrence is

```text
Q_U^P(h,a) = r(h,a) + gamma * V_U^P(M(h,a)),
```

up to the project’s perspective and sign conventions. If there is no intermediate reward, comparing successor-state rollout values can be equivalent to comparing action values.

For the same additive utility with random transitions,

```text
Q_U^P(h,a)
  = E_{h' | h,a}[r(h,a,h') + gamma * V_U^P(h')].
```

A whole-play utility that depends on the path can be evaluated directly on complete trajectories. A recursion using only the current state needs that state to retain all information required for the chosen utility. Record whether an action score already includes the expectation over transitions.

### Do not count continuation value twice

If `V_U^P` or `Q_U^P` already denotes expected continuation return, it should not normally be accumulated again along the same sampled trajectory as though it were instantaneous utility:

```text
sum_t gamma^t Q_U^P(h_t,a_t)
```

usually counts later consequences repeatedly. Either:

1. define `U(tau)` from instantaneous or terminal utilities and take its expectation; or
2. use the root continuation value that already summarizes that expectation.

---

## 7. Can every policy be rationalized by some trajectory utility?

With enough freedom in the utility and consequence representation, almost any deterministic policy can be made optimal.

For example, define a policy-following reward

```text
r_P(h,a) = 1[a = P(h)]
```

and a discounted trajectory utility

```text
U_P(tau) = sum_{t >= 0} gamma^t r_P(h_t,a_t)
```

with `gamma < 1/2`. Following `P` earns an immediate advantage of `1`. No possible difference in all later discounted rewards can outweigh that immediate advantage because

```text
gamma / (1 - gamma) < 1.
```

Thus `P` is uniquely optimal at every history.

This proves an existence result, but it is not automatically explanatory. The policy has simply been encoded into the reward. A similarly unconstrained utility can rationalize behavior while providing little compression, transfer, prediction, or insight.

For stochastic policies, classical expected-utility maximization ordinarily concentrates choice on maximizing actions. Exact nondegenerate action probabilities require additional structure, such as:

- ties plus an external mixing rule;
- latent random utility or latent context;
- entropy-regularized or maximum-entropy choice;
- a softmax/quantal-response decoder;
- or treating the policy’s random seed as part of an enriched state.

Accordingly, `softmax(log P)` is the direct exact reconstruction of a stochastic policy, while a VNM-style account of its precise randomization is an additional modeling claim.

---

## 8. Which comparison answers each question?

The answer is objective-relative.

| Objective | Preferred representation and selection mechanism |
|---|---|
| Exact deterministic behavioral reproduction | action score plus fixed-tie-break `argmax` |
| Exact stochastic behavioral reproduction | log score plus exponential sampling |
| Reproduce only the modal action of a stochastic policy | `argmax` over action scores |
| Measure how often `P` chooses a highest-return move and how much payoff its choices forgo | Compare the source move probabilities with the estimated return of each legal move under `Q_U^P` |
| Give a VNM-style rational-agent interpretation | `argmax` over expected cardinal utility of induced trajectory lotteries |
| Give a stochastic utility-like interpretation | random-utility, entropy-regularized, or soft choice with an explicit contract |

Exact encoding with access to the source gives a reference for fidelity. When training a model from finite observations, compare direct move selection and expected-utility selection on the same withheld boards. Report separately how simulation error, the payoff, the continuation rule and access to the source change the result.

Expected-return structure can nevertheless be preferable for other reasons. It may be smoother, more compressible, more transferable across action sets, more predictive of interventions, and more interpretable in terms of consequences. Those are stronger properties to test, not guaranteed consequences of behavioral reconstruction.

---

## 9. The recommended two-stage research program

The clean theoretical spine is

```text
P  ->  S_P  ->  U.
```

### Stage A: universal behavioral value form

Recover a move score `S_P` such that

```text
P approximately equals D(S_P).
```

For supported finite policies this can be an exact theorem. The decoder family should remain small and policy-independent:

```text
deterministic policy  -> argmax
stochastic policy     -> normalized sampling.
```

This stage establishes the broad policy-value representation claim.

### Stage B: expected-utility factorization

Measure how closely a shared utility over outcomes approximates the scores that reproduce the original choices. The proposed form is

```text
S_P(h,a) approximately equals
E[U(tau) | h, a, environment, continuation rule]
```

up to the invariances relevant to the decoder.

Compare this utility with a table of scores for individual positions and actions. Report the error in predicting new choices, the number of adjustable values in each explanation, and how much each varies across independent fits.

Desirable evidence includes:

- low behavioral loss after the factorization;
- stability across independent recoveries;
- a compact or regular trajectory utility;
- transfer to unseen states, action sets, or environmental changes;
- correct predictions for action removal and counterfactual transitions;
- calibrated preferences over explicit lotteries;
- and resistance to arbitrary state enrichment that merely memorizes each decision.

The expected-utility factorization should be compared with the direct decoder score, not substituted for it by definition.

---

## 10. Practical evaluation consequences

Experiments should separate at least four sources of disagreement:

1. **Decoder approximation error**  
   The learned score fails to reproduce the policy even under its intended decoder.

2. **Return-estimation error**  
   Measure how far `V_phi` or `Q_phi` is from the declared expected return, separating variation from finitely many rollouts from error in fitting those targets.

3. **Greedy-improvement difference**  
   With exact expected returns, count different moves tied at the maximum separately from source moves with strictly lower return. Only the latter give up payoff for the forced move.

4. **Semantic-factorization error**  
   Measure the residual mismatch under a specified, restricted complete-play utility. A trained model supplies an achieved error. A claim about the entire utility family needs an optimum or justified bound.

Useful paired baselines are therefore:

```text
Behavioral score baseline:
    train S_phi directly on P’s actions, probabilities, or logits
    decode with argmax or normalized sampling

Return-value baseline:
    train Q_phi on rollout returns under a declared payoff contract
    greedily maximize Q_phi
```

For fixed-move policies, report the fraction of matching moves, the difference between the best and next-best scores, and the average source score forgone. For randomized policies, compare their move probabilities directly. Total variation is half the sum of the absolute probability differences. Cross-entropy is the average negative logarithm of the recovered probability of a source move. Subtracting the same quantity calculated with the source's own probabilities gives the Kullback–Leibler (KL) divergence. Report the logarithm base and any missing probability on a move the source makes. For return models also report prediction error and the average payoff forgone by the chosen moves.

---

## 11. Recommended terminology

To prevent overloaded uses of “value,” use the following distinctions:

- **Decoder value / behavioral score:** a score whose declared decoder reproduces `P`.
- **Revealed value:** an ordering or cardinalized score inferred from counterfactual choice behavior.
- **Return value:** expected trajectory utility under a declared environment, payoff, horizon, and continuation policy.
- **Utility factorization:** a representation of action scores as expectations of one utility over induced consequences.
- **Mechanistic value:** an internal representation that causally mediates the source policy.

The main universal claim concerns decoder value. Return value and utility factorization add semantics and explanatory content.

---

## 12. Bottom line

With direct access to the source, these are references for behavioral reconstruction.

```text
deterministic P:
    P(h) = argmax_a S_P(h,a)

stochastic P:
    P(. | h) = softmax(S_P(h,.) / T) for the log scores and equal base weights above.
```

A deterministic expected-utility maximizer reproduces the source move when the selected maximizer, including the tie rule, is the source move. For a random source, compare the entire induced distribution. The explanatory question is how much a shared utility with restricted inputs predicts about fresh conditions. [Checkpoint B decision](../CHECKPOINT_B.md)

The project should therefore defend two nested theses:

> **Behavioral representation thesis:** broad policy classes admit exact or arbitrarily accurate value-plus-decoder representations.

> **Utility-factorization thesis:** some behaviorally faithful value representations also admit simple, stable, useful factorizations as expected utility over trajectories.

The first thesis gives the universal policy-value bridge. The second determines when that bridge supports a substantive rational-agent or VNM-style interpretation.

See also:

- [`selection_semantics.md`](selection_semantics.md) for the operational contract governing scores, rollout values, lotteries, and double counting;
- [`../README.md`](../README.md) for the project overview;
- [`../TODO.md`](../TODO.md) for the planned representation, quotient, stability, and interpretation experiments.
