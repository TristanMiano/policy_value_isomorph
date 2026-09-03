# Policy Reconstruction versus Expected-Utility Factorization

## The question

The broad project claim is:

> Given an arbitrary policy `P`, recover a value-like object together with a declared selection mechanism that approximately reproduces `P`.

This raises an important design question: which selection mechanism should be expected to reproduce an arbitrary `P` most faithfully?

The answer depends on the target claim.

- For **behavioral reconstruction**, the theoretically closest mechanism is a decoder matched directly to the type of policy: `argmax` for deterministic policies and normalized-score sampling for stochastic policies.
- For a **VNM-style or consequential interpretation**, the stronger mechanism is maximization of expected utility over the trajectory lottery induced by each action.
- These mechanisms can coincide, especially when `V` or `Q` already approximates expected rollout return, but they do not coincide for an arbitrary policy.

The project should therefore treat universal policy reconstruction and expected-utility factorization as two related but distinct research problems.

---

## 1. The universal reconstruction form

Let

```text
h in H                     history or sufficiently rich state
C(h) subset A              feasible candidate actions
P(. | h, C)                source policy
V_P(h,a) in extended R     recovered score, energy, or value
D(V_P, h, C)               declared selection mechanism
```

The broadest representation thesis is

```text
P(. | h, C) approximately equals D(V_P, h, C).
```

The value-like object and its decoder form a pair. A bare scalar function does not determine behavior until the project specifies whether it is consumed by `argmax`, softmax, sampling from an energy model, thresholding, or another rule.

For arbitrary behavior, the most faithful decoder is the one whose output space matches the policy itself.

---

## 2. Deterministic policies: direct `argmax` can be exact

Suppose `P` is deterministic. On any finite feasible action set, define

```text
V_P(h,a) = 1  if a = P(h)
           0  otherwise.
```

Then, with a fixed tie-breaking convention,

```text
P(h) = argmax_{a in C(h)} V_P(h,a).
```

This representation is exact. It places no rationality, optimality, temporal consistency, or consequentialist assumptions on `P`. The policy can be path-dependent, locally inconsistent under a coarser state description, deliberately suboptimal under an external reward, or otherwise arbitrary.

Many other exact scores are possible. The chosen action only needs to receive a strictly greater score than every unchosen action. This nonuniqueness is why the base artifact is better understood as a decoder-relative value or score equivalence class rather than a uniquely identified utility.

For a deterministic policy on a metric action space, an analogous construction is

```text
V_P(h,a) = -d(a, P(h)),
```

provided the metric separates actions and the decoder can identify the unique maximizer.

### Why action value is the broad primitive

A scalar state value `V(h)` alone is not universally sufficient. Two actions may lead to the same represented successor state, transitions may be stochastic, or the action itself may matter independently of its successor. The broad primitive should therefore usually be an action or continuation score

```text
V_P(h,a)  or  Q_P(h,a).
```

A state-value representation can recover the same generality by lifting the state space to include the candidate action, successor, or trajectory prefix.

---

## 3. Stochastic policies: reproduce the distribution, not only its mode

If `P` is stochastic, an `argmax` decoder generally reproduces only its modal action. To reconstruct the entire policy distribution, the decoder must itself be stochastic.

For a discrete policy with positive probabilities, define

```text
V_P(h,a) = tau * log P(a | h) + c(h),
```

where `tau > 0` is a temperature and `c(h)` is any state-dependent additive constant. Then

```text
softmax(V_P(h,.) / tau) = P(. | h).
```

Zero-probability actions can be represented by `-infinity` or approximated with sufficiently negative finite scores.

Thus the exact stochastic counterpart of deterministic score-plus-`argmax` is

```text
log score or energy + normalized sampling.
```

For a continuous stochastic policy with density `p(a | h)` relative to a declared base measure, the analogous energy is proportional to `log p(a | h)`. The base measure is part of the semantics and cannot be omitted.

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
V_P(h,A) = 1
V_P(h,B) = 0.
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

In a deterministic transition system,

```text
Q_U^P(h,a) = r(h,a) + gamma * V_U^P(T(h,a)),
```

up to the project’s perspective and sign conventions. If there is no intermediate reward, comparing successor-state rollout values can be equivalent to comparing action values.

In a stochastic transition system,

```text
Q_U^P(h,a)
  = E_{h' | h,a}[r(h,a,h') + gamma * V_U^P(h')].
```

The transition expectation must be performed explicitly unless the supplied `Q` estimate already contains it.

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

## 8. Which mechanism is theoretically closest to `P`?

The answer is objective-relative.

| Objective | Preferred representation and selection mechanism |
|---|---|
| Exact deterministic behavioral reproduction | action score plus fixed-tie-break `argmax` |
| Exact stochastic behavioral reproduction | log score/energy plus normalized sampling |
| Reproduce only the modal action of a stochastic policy | `argmax` over action scores |
| Test whether `P` greedily maximizes a declared return | `argmax` over rollout-estimated `Q_U^P` |
| Give a VNM-style rational-agent interpretation | `argmax` over expected cardinal utility of induced trajectory lotteries |
| Give a stochastic utility-like interpretation | random-utility, entropy-regularized, or soft choice with an explicit contract |

For the sole loss function “match `P`,” direct policy-score distillation is theoretically dominant: it can be exact and trains against the behavioral target directly. Return-value recovery introduces environment dependence, payoff assumptions, rollout variance, continuation-policy dependence, and possible policy-improvement disagreement.

Expected-return structure can nevertheless be preferable for other reasons. It may be smoother, more compressible, more transferable across action sets, more predictive of interventions, and more interpretable in terms of consequences. Those are stronger properties to test, not guaranteed consequences of behavioral reconstruction.

---

## 9. The recommended two-stage research program

The clean theoretical spine is

```text
P  ->  S_P  ->  U.
```

### Stage A: universal behavioral value form

Recover a decoder value or energy `S_P` such that

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

Ask whether the behaviorally faithful score admits a simpler consequential structure:

```text
S_P(h,a) approximately equals
E[U(tau) | h, a, environment, continuation rule]
```

up to the invariances relevant to the decoder.

This is the stronger scientific question. It asks whether the policy’s apparent preferences can be explained by a stable utility over consequences rather than by a context-indexed action lookup table.

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
   `V_phi` or `Q_phi` inaccurately estimates the declared rollout return.

3. **Greedy-improvement difference**  
   The return estimate is accurate, but `P` does not greedily maximize its own induced `Q_U^P`.

4. **Semantic-factorization error**  
   A behaviorally faithful score does not admit the proposed simple expected-trajectory representation.

Useful paired baselines are therefore:

```text
Behavioral score baseline:
    train S_phi directly on P’s actions, probabilities, or logits
    decode with argmax or normalized sampling

Return-value baseline:
    train Q_phi on rollout returns under a declared payoff contract
    greedily maximize Q_phi
```

For deterministic policies, report action agreement, action margins, and regret under the source score. For stochastic policies, report KL divergence, cross-entropy, calibration, and sampling fidelity. For return values, separately report return calibration and the exact or estimated greedy-improvement gap.

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

For arbitrary `P`, the closest theoretical reconstruction is:

```text
deterministic P:
    P(h) = argmax_a V_P(h,a)

stochastic P:
    P(. | h) = sample from normalized V_P(h,.).
```

Maximizing expected trajectory utility is not generally a better imitation mechanism. It is a stronger attempted explanation of the policy. It reproduces `P` when `P` is already greedy with respect to the declared expected-utility contract, or when the utility has been tailored closely enough to encode `P`.

The project should therefore defend two nested theses:

> **Behavioral representation thesis:** broad policy classes admit exact or arbitrarily accurate value-plus-decoder representations.

> **Utility-factorization thesis:** some behaviorally faithful value representations also admit simple, stable, useful factorizations as expected utility over trajectories.

The first thesis gives the universal policy-value bridge. The second determines when that bridge supports a substantive rational-agent or VNM-style interpretation.

See also:

- [`selection_semantics.md`](selection_semantics.md) for the operational contract governing scores, rollout values, lotteries, and double counting;
- [`../README.md`](../README.md) for the project overview;
- [`../TODO.md`](../TODO.md) for the planned representation, quotient, stability, and interpretation experiments.
