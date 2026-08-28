# Policy–Value Isomorph Research Program — TODO

Last updated: 2026-08-28

## Resume here

**Next task: Task 23 — introduce the minimal PyTorch-backed model/training interfaces.**

Complete exactly one numbered task or one checkpoint per focused work session unless Tristan explicitly requests a larger batch. Preserve completed history, record negative results, and treat checkpoint revisions as part of the research rather than as administrative cleanup.

## Main philosophical stance

This project adopts a **value-first semantics of policy**:

> A policy is a discrete or probabilistic behavioral readout from a richer value-like landscape over actions, reachable states, continuations, or trajectories. Values are the primary semantic object; policy choices are obtained from them by `argmax`, softmax/sampling, thresholding, or another declared decoder.

The motivating semantic ladder is:

```text
Boolean truth {0,1}
    -> uncertain truth / expectation [0,1]
    -> general signed and unbounded value R
    -> policy or decision through argmax / sampling
```

Every nondegenerate bounded real interval is affinely equivalent to `[0,1]`; an unbounded carrier supports unrestricted signed differences, additive composition, margins, advantages, costs, and losses. Neural training reinforces this perspective because minimizing a loss `L` is exactly maximizing the value `U = -L`. The historical progression from hard thresholds to sigmoid/tanh and then ReLU-like activations is a motivating hypothesis to audit, not a premise to assume.

The long-range conjecture is not that every activation literally is utility. It is that neural networks may owe part of their generality and interpretability potential to computing in a rich real-valued value carrier and postponing discrete truth, class, or action readouts until late in the computation.

The project will not define success as recovering one metaphysically unique reward hidden inside every system. It will proceed through increasingly strong meanings of “value form”:

1. **Decoder value:** a score or energy whose declared decoder exactly reproduces the policy.
2. **Revealed value:** a ranking or cardinalized score inferred from behavior under counterfactual action sets or perturbations.
3. **Return value:** expected outcome or cumulative return under a declared environment, horizon, perspective, and payoff contract.
4. **Mechanistic value:** an internal representation that causally mediates the source policy’s computation.

The first level should exist very broadly. The later levels are stronger empirical and formal claims. Failure to identify a unique reward does **not** refute policy-to-value representation; it identifies the relevant equivalence class, missing query information, or extra assumptions needed for a stronger interpretation.

## Core research objects

Use the following general form unless a task deliberately narrows it:

```text
h in H                     state, context, or full history
C(h) subset A              feasible candidate actions/outputs
P(. | h, C)                deterministic or stochastic policy
V_P(h, a) in extended R    recovered value/score/energy form
D(V_P, h, C)               declared decoder producing behavior
R(P, contract, seed)       recovery procedure returning a ValueArtifact
```

A `policy` is therefore any history-to-action transducer: a tabular controller, MLP, time-series forecaster, robot controller, autoregressive language model, tool-using agent, or another bounded decision procedure. For large or continuous output spaces, `C(h)` may be a generated candidate set and the primary universal object will usually be action/continuation value `Q(h,a)` or energy rather than a scalar state value alone.

A `ValueArtifact` must eventually contain not only a trained network, but also its decoder, normalization, recovery contract, supported domain, provenance, uncertainty/repeated-run information, and evaluation results.

## High-level theses

### PV-H1 — Universal value-form representation

For broad policy classes, an exact value-like representation exists after fixing a decoder and minimal structure on the action space.

- Finite deterministic policies admit exact action-score representations decoded by tie-broken `argmax`.
- Discrete stochastic policies admit log-score/energy representations decoded by softmax, using extended values for zero-probability actions when needed.
- Deterministic policies on metric action spaces admit distance-based energies such as `-d(a,P(h))`.
- State-value form can absorb action value by lifting the state/history space to include the candidate action or successor.

**Research route:** prove exact scoped theorems, implement executable witnesses, then determine which constructions are useful rather than merely lossless.

### PV-H2 — The recoverable object is a value equivalence class plus a normalization

Behavior usually determines value only up to transformations invisible to the declared decoder or query family. The project should characterize that quotient and then choose transparent canonical representatives rather than treating nonuniqueness as failure.

**Research route:** prove decoder- and query-relative equivalence results; compare anchor, affine, rank, quantile, and temperature-aware normalizations.

### PV-H3 — Independent recoveries have a stable core

Repeated runs of a fixed recovery contract should converge, after appropriate alignment, on at least some combination of:

- induced policy behavior;
- within-state action ordering;
- normalized action gaps;
- local finite-difference geometry;
- counterfactual predictions; and
- return predictions where a return contract exists.

**Research route:** separate variation from data, rollout noise, initialization, architecture, optimization, candidate generation, and normalization. If a single representative does not converge, characterize clusters or an ensemble-valued “value posterior”; stable structure across runs is the candidate interpretable core.

### PV-H4 — Value form provides interpretive leverage beyond the original policy interface

A recovered value artifact should make questions answerable that a bare action choice does not directly answer: runner-up actions, degrees of preference, nearby reachable-state comparisons, indifference surfaces, sensitivity to constraints, and predicted response to policy interventions.

**Research route:** define prospective interpretability tasks and compare value-based explanations with direct policy/logit/saliency baselines.

### PV-H5 — Value supplies a controlled stochastic completion of deterministic behavior

A normalized value function can induce a family of stochastic policies, for example

```text
pi_(tau,mu)(a|h) proportional to mu(a|h) * exp(V(h,a)/tau),
```

where `mu` is a base measure and `tau` a temperature. The zero-temperature limit should recover the deterministic argmax when it is unique.

**Research route:** prove invariances and scale/temperature relations, then test whether stochastic completions are stable across recovered values and useful for exploration, diversity, or uncertainty-aware behavior.

### PV-H6 — One recovery contract can span heterogeneous policy classes

The same conceptual interface—history, candidates, policy access, counterfactual query family, decoder, normalization, and evaluation semantics—should apply across discrete games, continuous controllers, forecasting models, autoregressive sequence models, and agents, even when the concrete recovery algorithm differs.

**Research route:** build adapters in increasing difficulty and publish a capability matrix specifying which semantics are supported under black-box, probability/logit, rollout, and white-box access.

### PV-H7 — External value recovery can become a neural interpretability program

For neural policies, externally recovered value structure may be predictable from, aligned with, or causally mediated by internal activations and gradients. Backpropagated `-loss` gradients are local marginal-value signals relative to the training objective and provide one bridge to test.

**Research route:** begin only after external recovery is stable; test probes, representation similarity, activation interventions, gradient/value alignment, and changed-objective transfer. A negative mechanistic result would leave the behavioral value artifact intact while narrowing the internal interpretation.

## Minimum publishable path and expansion path

The **minimum publishable path** is Tasks 23–78: finish the PyTorch migration, establish the formal value-form/quotient results, build the recovery and stability framework, and complete exact discrete-domain experiments plus the first interpretive-value atlas.

Tasks 79 onward form the **generalization and interpretability path**: continuous control, time series, sequence/LLM policies, agents, and mechanistic tests. Each checkpoint may reorder, narrow, or remove later tasks based on the strongest completed evidence.

## Work and evidence protocol

For each numbered task:

1. Read this file, the files named by the task, and the most recent checkpoint.
2. Complete only that task’s stated scope.
3. Add tests or deterministic fixtures for executable claims.
4. Run `pytest -q`; run any task-specific validation command; record commands and results.
5. Update this TODO entry with completion date, result, files changed, and claim impact.
6. Commit the task as one cohesive commit. Do not automatically push unless explicitly requested.

For empirical claims:

- freeze the population, seeds, metrics, comparison rules, and stopping rule before the final run;
- distinguish exploratory, pilot, and confirmation data;
- report seed-level results and uncertainty, not only a favored run;
- retain failed hypotheses and implementation defects in the record; and
- never let behavior agreement alone establish return, preference, causal, or human interpretability semantics.

At each checkpoint, audit all work since the previous checkpoint, update the thesis/claim ledger, revise later tasks when justified, set the next-task pointer, and stop.

---

# Completed foundation

## Initial policy-to-value pipeline

- [x] **1.** Add on-policy and off-policy tic-tac-toe state-space sampling utilities.
- [x] **2.** Implement a small tic-tac-toe MLP policy and training loop.
- [x] **3.** Freeze trained policies and generate configurable Monte Carlo value targets.
- [x] **4.** Implement and train `V_phi(s)` on rollout labels.
- [x] **5.** Add action agreement, top-k agreement, gameplay, and calibration evaluation.
- [x] **6.** Add `Q_phi(s,a)` recovery and compare it with successor-state `V` recovery.
- [x] **7.** Add tic-tac-toe symmetry augmentation/reduction.
- [x] **8.** Add CLI entry points for generation, training, and evaluation.
- [x] **9.** Port the environment/pipeline structure to Connect Four.
- [x] **10.** Document baseline reproducibility settings.

## Monitoring, checkpoints, and resumability

- [x] **11.** Standardize telemetry across policy, `V`, and `Q` runs.
- [x] **12.** Add append-safe CSV/JSONL telemetry backends.
- [x] **13.** Add training-curve plotting utilities.
- [x] **14.** Add periodic gameplay evaluation during training.
- [x] **15.** Add gameplay/score-over-time reporting.
- [x] **16.** Save human-readable gameplay samples at checkpoints.
- [x] **17.** Save periodic policy, `V`, and `Q` checkpoints.
- [x] **18.** Implement deterministic resume/continuation where feasible.
- [x] **19.** Add CLI support for monitoring/checkpoint workflows.
- [x] **20.** Add monitoring/checkpoint tests and smoke demos.

## PyTorch migration decision

- [x] **21.** Audit manual training and define PyTorch migration boundaries.
- [x] **22.** Record the PyTorch-first decision and compatibility constraints.

---

# Phase 0 — Finish the PyTorch foundation

- [ ] **23. Introduce minimal internal interfaces for PyTorch-backed models and training steps.**
  - Add explicit model, optimizer, batch, device, and prediction boundaries shared by policy, `V`, and `Q` without building a large framework.
  - Preserve pure data preparation and evaluation functions.
  - **Done when:** interfaces support the current one-hidden-layer models and are covered by focused shape/type tests.

- [ ] **24. Port policy, value, and Q MLPs to PyTorch with parity fixtures.**
  - Preserve public APIs, legal-action masking, perspective/sign conventions, and telemetry.
  - Keep the first architecture equivalent to the current tanh MLP.
  - **Done when:** forward behavior and small deterministic training fixtures match the old implementation within declared tolerances.

- [ ] **25. Replace manual optimization with configurable `torch.optim.AdamW`.**
  - Add device selection, dtype, seeding, and deterministic settings where available.
  - Expose optimizer parameters through existing CLI paths.
  - **Done when:** all three training paths run end to end and baseline commands are documented.

- [ ] **26. Evaluate Muon only after AdamW parity is complete.**
  - Keep this optional and config-gated; do not fork the training stack.
  - Compare maintenance burden, convergence, wall time, and final behavior on a frozen small benchmark.
  - **Done when:** retain or reject Muon with a recorded result; rejection completes the task.

- [ ] **27. Migrate checkpoint/resume to PyTorch model and optimizer state.**
  - Version the artifact schema and provide a clear legacy policy.
  - Preserve latest/specified checkpoint behavior and telemetry continuation.
  - **Done when:** round-trip and interrupted-run tests pass for policy, `V`, and `Q`.

- [ ] **28. Expand tests and smoke demos for the PyTorch stack.**
  - Add CPU end-to-end tests for policy training, rollout labeling, `V`, `Q`, evaluation, and resume.
  - Mark any GPU-only checks separately.
  - **Done when:** `pytest -q` and a documented smoke command pass from a fresh editable install.

- [ ] **29. Update dependencies, README, reproducibility docs, and examples for the PyTorch-first stack.**
  - Replace stale “no neural nets/no dependencies” language in `AGENTS.md`, `DESIGN.md`, `REFERENCES.md`, and package metadata.
  - Document optional dependency groups planned for control, sequence, LLM, and visualization work.
  - **Done when:** a new contributor can reproduce the current baseline without relying on historical Task 1 instructions.

## Checkpoint A — Backend and scope readiness

- [ ] Audit Tasks 23–29, run the full suite, and freeze the package interfaces that the research phases may rely on. Confirm that the new philosophical stance does not require changing existing value-sign conventions before the formalism is written.

---

# Phase I — Freeze the research constitution and literature boundary

- [ ] **30. Create `RESEARCH_STANCE.md`.**
  - State the value-first position, semantic ladder, bounded-interval affine collapse, loss/value duality, exact Boolean/readout recovery from real-valued computation, four levels of value form, and the distinction between existence, usefulness, identifiability, and mechanism.
  - Cross-link the companion `value_logic` project without making it a code dependency.
  - **Done when:** the document makes clear that reward nonuniqueness narrows a claim rather than refuting the base representation thesis.

- [ ] **31. Create `CLAIM_LEDGER.md`.**
  - Give PV-H1–PV-H7 stable IDs, exact scoped statements, theorem/experiment routes, support and falsification conditions, dependencies, and project impact.
  - Split broad theses into approximately 20 atomic claims.
  - **Done when:** no planned empirical claim is marked supported merely because it is motivating.

- [ ] **32. Define the policy-access and value-semantics taxonomy.**
  - Create `docs/policy_value_taxonomy.md` covering deterministic action access, samples, probabilities/logits, constrained-choice access, rollout access, environment access, and white-box internals.
  - Define decoder, revealed, return, and mechanistic value artifacts.
  - **Done when:** every later adapter can declare capabilities without overloading “value function.”

- [ ] **33. Audit the primary literature and expand `REFERENCES.md`.**
  - Cover policy evaluation/improvement, policy distillation, inverse decision/reward modeling, maximum-entropy control, reward invariances, revealed preference/choice, value equivalence, imitation distribution shift, representation similarity, explainable RL, many-valued/fuzzy neural interpretations, and the historical activation/loss story.
  - Use primary papers and record exactly what is imported versus merely analogous.
  - **Done when:** each load-bearing concept in PV-H1–PV-H7 has a verified source or an explicit original-proof obligation.

- [ ] **34. Freeze notation and the recovery contract.**
  - Create `formalism/00_notation.md` defining histories, candidate sets, policies, value forms, decoders, recovery procedures, equivalence, normalization, and `ValueArtifact`.
  - Include discrete, continuous, and sequence examples.
  - **Done when:** later formal files link to one authoritative notation source.

- [ ] **35. Define prospective success metrics and failure modes.**
  - Create `docs/evaluation_contract.md` separating behavioral, ordinal, cardinal, geometric, return, stochastic-completion, representational, causal, and human-facing evidence.
  - Define what would count as genuine interpretive leverage.
  - **Done when:** “similar V,” “what P likes,” and “understand P” each have operational tests.

- [ ] **36. Produce a small hand-worked policy/value atlas.**
  - For one tic-tac-toe state, manually show the source action, all legal alternatives, exact/rollout/revealed values, action gaps, normalizations, and one softmax completion.
  - **Done when:** the example exposes every ambiguity the formalism must handle before more code is written.

## Checkpoint B — Thesis and notation freeze

- [ ] Review Tasks 30–36 and decide the exact theorem spine, minimum experiment suite, and terminology. Remove any claim that accidentally equates decoder score, return, latent reward, and mechanism.

---

# Phase II — Prove the value-form and equivalence results

- [ ] **37. Prove finite deterministic policy-to-score representation.**
  - For finite feasible action sets and deterministic `P`, construct an exact real action-score representation with fixed tie-breaking.
  - Characterize the encoder image and inverse decoder.
  - **Done when:** theorem, proof, examples, and executable finite tests agree.

- [ ] **38. Prove discrete stochastic policy-to-log-score representation.**
  - Represent a stochastic policy by logits/energies up to state-dependent additive constants; handle zero support with extended reals or a declared approximation.
  - State the role of temperature.
  - **Done when:** softmax recovery and invariance properties are proved and tested.

- [ ] **39. Extend exact representation to deterministic metric-action policies.**
  - Use a separating distance/energy such as `-d(a,P(h))` and state assumptions for unique decoding, measurability, and candidate restriction.
  - **Done when:** finite-dimensional continuous-action examples and edge cases are explicit.

- [ ] **40. Prove action-value to state-value lifting.**
  - Show how `Q(h,a)` can be represented as state value on an action-augmented history or successor object.
  - State when an ordinary `V(h)` plus a transition/action harness is sufficient.
  - **Done when:** the theorem covers the current deterministic game convention and a general history-based form.

- [ ] **41. Define decoder-relative value equivalence and prove the quotient result.**
  - Two values are equivalent when every declared decoder/query consumer sees the same result.
  - Prove that exact representations factor through the response quotient and identify common invariant transformations.
  - **Done when:** deterministic argmax, softmax with known temperature, and threshold consumers have separate examples.

- [ ] **42. Prove normalization and scale/temperature lemmas.**
  - Analyze additive anchors, positive affine transforms, rank/quantile normalization, and the equivalence between value scale and softmax temperature.
  - **Done when:** the project has a justified comparison protocol rather than raw-output RMSE between arbitrary gauges.

- [ ] **43. Prove revealed ranking recovery from constrained choice sets.**
  - If a deterministic policy chooses the unique maximum of one stable ordering from every offered finite subset, prove that pairwise or elimination queries recover the total order.
  - Record consistency tests and counterexamples when context effects violate the assumption.
  - **Done when:** this yields an implementable black-box preference-probing algorithm.

- [ ] **44. Prove score-difference recovery for soft/stochastic choice.**
  - Under a declared Luce/softmax model, recover pairwise score differences from probability or repeated-choice ratios.
  - Separate ordinal recovery, cardinal differences, unknown temperature, and finite-sample uncertainty.
  - **Done when:** exact and sampled fixtures reproduce the derivation.

- [ ] **45. Formalize return-value recovery as an added contract.**
  - Define environment, state/history sufficiency, payoff/loss, horizon/discount, perspective, and evaluation distribution.
  - Show precisely how decoder value and policy-induced `V^P/Q^P` relate and differ.
  - **Done when:** the current rollout code is an instance of the formal contract.

- [ ] **46. Prove stability bounds linking value error, action gaps, and policy agreement.**
  - Include deterministic argmax bounds and, if tractable, a softmax distribution bound after gauge alignment.
  - **Done when:** the bounds directly motivate the empirical convergence metrics.

- [ ] **47. Build the constructive non-identifiability suite.**
  - Include equivalent argmax scores, unknown softmax temperature, unqueried alternatives, reward shaping/return ambiguity, insufficient state, and off-support behavior.
  - For each case, record the stronger interpretation that fails and the value-form result that survives.
  - **Done when:** future work cannot mistake one counterexample for a refutation of PV-H1.

- [ ] **48. Implement executable formal fixtures.**
  - Add a small exact module and tests for Tasks 37–47, independent of neural training.
  - **Done when:** one command checks every finite theorem example and normalization invariant.

## Checkpoint C — Formal spine audit

- [ ] Adversarially check every proof, boundary convention, and use of “isomorphism.” Freeze the exact positive theorem claims and the quotient language before building the generalized recovery API.

---

# Phase III — Build the generalized recovery and comparison framework

- [ ] **49. Define the `PolicyAdapter` protocol.**
  - Support history encoding, feasible/candidate actions, deterministic action, sampled action, optional probabilities/logits, optional constrained choice, optional transition/rollout, and optional internals.
  - **Done when:** tic-tac-toe and Connect Four satisfy the protocol without changing their game logic.

- [ ] **50. Define `RecoveryContract` and `ValueArtifact` dataclasses.**
  - Record semantics level, access mode, query/state distribution, candidate generator, decoder, normalization, provenance, uncertainty, and supported scope.
  - **Done when:** artifacts are serializable and reject semantically incomplete combinations.

- [ ] **51. Implement exact decoder-value encoders.**
  - Add deterministic indicator/gap scores, stochastic log scores, and metric-action distance energies.
  - **Done when:** round-trip tests recover the source behavior exactly on supported domains.

- [ ] **52. Implement policy-to-score distillation.**
  - Train value/action-score networks directly from actions, probabilities, logits, or pairwise choices according to access level.
  - **Done when:** each supervision mode is a separate named recovery method with matched evaluation.

- [ ] **53. Generalize rollout return-value recovery.**
  - Separate Monte Carlo, exact tabular evaluation where available, and optional TD/bootstrapped estimation.
  - Preserve outcome/value provenance and policy perspective.
  - **Done when:** recovery method and semantic contract cannot be silently mixed.

- [ ] **54. Implement constrained-choice and active preference probing.**
  - Recover action rankings from masked candidate sets; add deterministic consistency diagnostics and a stochastic pairwise model.
  - **Done when:** the probe budget and query history are part of the artifact.

- [ ] **55. Implement candidate and neighborhood generators.**
  - Cover legal discrete actions, nearby reachable states, sampled continuous actions, and generated sequence candidates.
  - **Done when:** candidate-generation bias is measurable and reproducible.

- [ ] **56. Implement value alignment and canonicalization.**
  - Support anchor subtraction, standardized/robust scale, affine fit to shared anchors, isotonic/rank alignment, and temperature-aware alignment.
  - **Done when:** comparison refuses raw cardinal metrics across incompatible gauges.

- [ ] **57. Implement the multi-axis value comparison suite.**
  - Behavioral: agreement, KL, regret.
  - Ordinal: Kendall/Spearman and top-k overlap within state.
  - Cardinal: aligned error and action-gap calibration.
  - Geometric: local finite-difference, neighborhood, and contour agreement.
  - Return/trajectory: calibration and induced rollout differences.
  - **Done when:** each metric declares invariances and data requirements.

- [ ] **58. Build the repeated-recovery experiment orchestrator.**
  - Factor seeds for data, rollouts, initialization, optimizer order, architecture, and candidate generation.
  - Store manifests and pairwise/barycenter comparison tables.
  - **Done when:** a small matrix resumes safely and detects accidental seed reuse.

- [ ] **59. Build the first `ValueAtlas` reporting API.**
  - Emit per-context action tables, value gaps, uncertainty across recoveries, local state/action neighborhoods, and source-policy comparison.
  - **Done when:** a deterministic policy yields more inspectable information than a single chosen action.

## Checkpoint D — Infrastructure and experiment freeze

- [ ] Validate Tasks 49–59 on tiny fixtures. Freeze the exact discrete-domain experiment populations, seed matrix, metrics, and report schema before final runs.

---

# Phase IV — Exact discrete-domain recovery and convergence experiments

- [ ] **60. Create a diverse finite policy suite.**
  - Include optimal, scripted suboptimal, random, stochastic mixtures, trained MLP checkpoints, and deliberately context-inconsistent choice policies.
  - **Done when:** the suite spans exact-representable, noisy, and assumption-violating cases.

- [ ] **61. Enumerate the tic-tac-toe ground-truth state/action graph.**
  - Compute exact policy outputs, reachable-state distributions, exact returns where defined, symmetries, and action gaps.
  - **Done when:** all recovery methods can be scored against exhaustive references rather than sampled anecdotes.

- [ ] **62. Freeze the first confirmation protocol.**
  - Predeclare policy suite, access modes, budgets, seed counts, alignments, primary endpoints, and stopping rules.
  - **Done when:** the final run can execute without discretionary metric changes.

- [ ] **63. Run exact decoder-score recovery baselines.**
  - Demonstrate the representation theorems and quantify compression/model approximation error separately.
  - **Done when:** exact encoder and learned-distillation results are not conflated.

- [ ] **64. Run constrained-choice ranking recovery.**
  - Measure query complexity, ordering recovery, inconsistency diagnostics, and active-vs-random probing.
  - **Done when:** deterministic policies acquire a tested ranking over unchosen alternatives under the declared query model.

- [ ] **65. Run stochastic/log-score recovery.**
  - Compare direct probabilities/logits with sample-only estimation across sample budgets and temperatures.
  - **Done when:** gauge-aligned score differences and induced-policy error are reported.

- [ ] **66. Run exact and Monte Carlo return-value recovery.**
  - Compare rollout budgets, on/off-policy states, `V` vs `Q`, and approximation architectures against exact tabular policy evaluation.
  - **Done when:** convergence curves separate target noise from network fit.

- [ ] **67. Run the repeated-recovery stability matrix.**
  - Train enough independent recoveries to estimate within-method and between-method variability.
  - **Done when:** behavioral, ordinal, cardinal, and geometric stability are reported together.

- [ ] **68. Analyze convergence to a representative, quotient, or cluster structure.**
  - Test scaling with data/query/rollout budget; identify ensemble barycenters or stable modes after alignment.
  - **Done when:** PV-H3 receives a scoped disposition rather than a single seed anecdote.

- [ ] **69. Produce exhaustive value-atlas visualizations.**
  - Show action values, runner-up gaps, nearby reachable-state values, uncertainty, and disagreement clusters on the exact state graph.
  - **Done when:** examples include both clear and ambiguous “what P likes” cases.

- [ ] **70. Test value-derived stochastic completions.**
  - Sweep normalization, base measure, and temperature; compare stability across recovered values and zero-temperature recovery.
  - **Done when:** the project can state what stochastic behavior is implied, chosen by convention, or unsupported.

- [ ] **71. Test sensitivity to controlled policy edits.**
  - Modify source behavior at known states or action neighborhoods and measure whether recovered value changes localize and scale appropriately.
  - **Done when:** value artifacts demonstrate intervention sensitivity beyond global action agreement.

- [ ] **72. Publish the finite-domain result report.**
  - Create a machine-readable result bundle and a narrative report with all hypotheses, negative results, and representative atlases.
  - **Done when:** the formal claims and empirical results are cross-referenced to the claim ledger.

## Checkpoint E — First research adjudication

- [ ] Decide whether the project has established a stable and useful value-form core in exact discrete domains. Select the strongest recovery method(s) for the next phase; do not carry every method forward automatically.

---

# Phase V — Interpretive leverage in larger neural game policies

- [ ] **73. Establish PyTorch Connect Four policy/value baselines.**
  - Train multiple policy qualities and checkpoints under a frozen protocol.
  - **Done when:** source policies are behaviorally characterized before value recovery.

- [ ] **74. Recover multiple value forms from each neural policy.**
  - Compare decoder distillation, constrained-choice revealed value, and return value under matched state distributions.
  - **Done when:** semantics are separated in artifacts and reports.

- [ ] **75. Measure run-to-run and architecture-to-architecture stability.**
  - Use output-function metrics first; use hidden-representation metrics only as secondary evidence.
  - **Done when:** stable action rankings and local geometry are distinguished from parameter similarity.

- [ ] **76. Test off-support and counterfactual state coverage.**
  - Generate legal but rare states, adversarial neighborhoods, and trajectories induced by recovered policies.
  - **Done when:** value-atlas confidence reflects query/data coverage.

- [ ] **77. Test active counterfactual probing.**
  - Select states and candidate actions that maximally reduce ensemble disagreement or ordering uncertainty.
  - **Done when:** active probing is compared prospectively with random/query-frequency baselines.

- [ ] **78. Test prospective interpretability tasks.**
  - Predict runner-up actions, policy response to action removal, local policy edits, changed sampling temperature, and trajectory consequences.
  - Compare with direct policy probabilities/logits and standard attribution baselines.
  - **Done when:** PV-H4 receives a quantitative scoped disposition.

- [ ] **79. Build interactive/static neural-policy value atlases.**
  - Provide state views, action profiles, gaps, uncertainty, counterfactual masks, and trajectory summaries.
  - **Done when:** reports remain usable without inspecting raw tensors.

- [ ] **80. Test value-derived policy sampling in Connect Four.**
  - Compare diversity, source-policy fidelity, playing strength, and cross-recovery stability over temperature.
  - **Done when:** PV-H5 receives a scoped empirical disposition.

- [ ] **81. Compare value explanations with direct surrogate explanations.**
  - Include behavior cloning, policy distillation/logits, and simple local surrogate baselines.
  - **Done when:** any claimed interpretive advantage is incremental rather than merely a consequence of fitting another network.

- [ ] **82. Write the neural-game empirical paper/report draft.**
  - Lead with the formal representation/quotient results, stability evidence, and interpretive tasks; keep mechanistic claims out unless tested.
  - **Done when:** every figure and claim points to a reproducible artifact.

## Checkpoint F — Minimum publishable program

- [ ] Audit Tasks 73–82 and decide whether to release a first paper/software benchmark before broader adapters. Freeze which results are established and which remain conjectures.

---

# Phase VI — Continuous controllers and time-series policies

- [ ] **83. Generalize adapters to history-based and continuous-action policies.**
  - Add metric actions, candidate samplers, action bounds, and trajectory histories without forcing finite enumeration.
  - **Done when:** discrete adapters remain unchanged and a continuous toy fixture round-trips.

- [ ] **84. Add a lightweight continuous-control environment.**
  - Start with a reproducible Gymnasium-class task such as Pendulum; avoid MuJoCo as the first dependency.
  - **Done when:** scripted and trained deterministic/stochastic controllers are available.

- [ ] **85. Recover continuous action-energy landscapes.**
  - Compare exact distance-to-action encodings, local policy distillation, sampled candidate rankings, and return critics.
  - **Done when:** 1D/2D slices and local geometry metrics are validated.

- [ ] **86. Measure continuous-domain stability and stochastic completion.**
  - Account for base measures, action scaling, candidate density, and temperature.
  - **Done when:** induced densities and controller behavior are compared across independent recoveries.

- [ ] **87. Gate a simulated robot-controller case study.**
  - Select a simple simulator only if Tasks 84–86 show stable value recovery and the added dependency buys a genuinely new test.
  - **Done when:** either complete the scoped case study or document a justified rejection/defer decision.

- [ ] **88. Define a time-series model as a policy over predictions or interventions.**
  - Make context/history, output candidates, transition/evaluation semantics, and task loss explicit.
  - **Done when:** the mapping is not merely metaphorical and supports counterfactual candidate scoring.

- [ ] **89. Implement a forecasting-policy pilot.**
  - Use a small deterministic and probabilistic forecaster; recover decoder score and negative-loss/return-like value over candidate forecasts.
  - **Done when:** nearby forecast alternatives and uncertainty can be visualized.

- [ ] **90. Test cross-domain invariants.**
  - Compare which stability and interpretability metrics survive games, control, and forecasting.
  - **Done when:** PV-H6 has a capability matrix rather than a universal claim by assertion.

## Checkpoint G — Non-sequence portability

- [ ] Decide whether the shared recovery contract genuinely spans discrete games, control, and forecasting. Split the interface only where the evidence requires it.

---

# Phase VII — Autoregressive models, LLMs, and agents

- [ ] **91. Formalize autoregressive generation as a policy over histories.**
  - Treat tokens, structured outputs, or tool calls as actions and continuations as reachable states.
  - Distinguish token score, sequence score, outcome value, and agent return.
  - **Done when:** the notation handles finite candidate sets and open-ended generation.

- [ ] **92. Build a tiny autoregressive sequence-model fixture.**
  - Train or define a small inspectable model where logits and exact sequence probabilities are available.
  - **Done when:** token-level value-form round trips are exact.

- [ ] **93. Recover value from black-box deterministic decoding.**
  - Use constrained candidate sets, prompt/history perturbations, repeated sampling where available, and distillation.
  - **Done when:** the limits and gains of each access mode are experimentally separated.

- [ ] **94. Recover and compare sequence/trajectory values.**
  - Compare cumulative log score, learned continuation ranking, and declared outcome/return value.
  - **Done when:** token and trajectory semantics are not conflated.

- [ ] **95. Run repeated-recovery stability experiments for sequence models.**
  - Measure token ordering, sequence ranking, local semantic neighborhoods, and generated-distribution agreement.
  - **Done when:** candidate-generation variance is separated from value-model variance.

- [ ] **96. Add an optional small open-weight LLM adapter.**
  - Prefer a model small enough for reproducible local experiments; make `transformers` an optional dependency.
  - **Done when:** logits/probabilities, constrained candidates, and black-box-like modes can all be exercised.

- [ ] **97. Add a tool-using agent adapter.**
  - Represent tool calls and structured arguments as actions, with explicit environment transition and outcome contracts.
  - **Done when:** one small agent task yields a value atlas over alternative calls and continuations.

- [ ] **98. Test value-derived sampling and intervention on language/agent policies.**
  - Compare temperature families, alternative ranking, action removal, prompt perturbation, and downstream outcome effects.
  - **Done when:** value-based stochastic completion is tested outside board games.

- [ ] **99. Produce the sequence/agent portability report.**
  - Include access requirements, compute costs, failure modes, and a direct comparison with native logits when available.
  - **Done when:** PV-H6 is supported, narrowed, or split by policy class.

## Checkpoint H — General policy-to-value adjudication

- [ ] Evaluate the original broad objective: what method family, rather than one monolithic algorithm, now transforms arbitrary policy interfaces into declared value forms? Publish the supported adapter/semantics matrix and prune unsupported universality language.

---

# Phase VIII — Neural value semantics and mechanistic interpretability

- [ ] **100. Freeze mechanistic interpretability hypotheses.**
  - Define external value prediction, internal decodability, causal mediation, gradient alignment, and changed-objective transfer as separate claims.
  - **Done when:** no probe accuracy result can be mislabeled causal.

- [ ] **101. Probe policy internals for recovered value variables.**
  - Predict aligned values, rankings, and gaps from hidden layers across states and actions.
  - **Done when:** train/test splits prevent trivial state or action leakage.

- [ ] **102. Compare internal representations across independently trained policies and values.**
  - Use CKA/related measures only alongside functional alignment and output behavior.
  - **Done when:** representation similarity is interpreted at its actual invariance level.

- [ ] **103. Test causal interventions on candidate value features.**
  - Use activation patching, steering, ablation, or controlled latent edits and predict their effect on action values and choices.
  - **Done when:** interventions have preregistered direction/magnitude tests and matched controls.

- [ ] **104. Test backpropagated marginal-value semantics.**
  - Compare `-dL/dh` or objective gradients with recovered local value differences and policy sensitivity.
  - **Done when:** the result distinguishes training-objective marginal value from deployment return or latent preference.

- [ ] **105. Search for interpretable value directions or concepts.**
  - Relate features to stable action/trajectory tradeoffs, not merely labels.
  - **Done when:** concepts predict held-out counterfactual comparisons and survive appropriate controls.

- [ ] **106. Test changed-decoder and changed-objective transfer.**
  - Ask whether retained value representations answer new thresholds, temperatures, constraints, costs, or action sets better than direct policy labels.
  - **Done when:** the experiment directly connects this repo to the value-first thesis.

- [ ] **107. Compare value-based interpretability against standard baselines.**
  - Include logits/probabilities, saliency, local linear surrogates, behavior cloning, and mechanistic probes not trained on value.
  - **Done when:** value adds measurable leverage or the claim is narrowed.

- [ ] **108. Run one preregistered mechanistic confirmation experiment.**
  - Select the most promising policy class and hypothesis only after pilot work, then freeze the final protocol.
  - **Done when:** PV-H7 receives a transparent disposition.

- [ ] **109. Write the neural value-semantics research report.**
  - Separate behavioral representation, external interpretability, and internal mechanism throughout.
  - **Done when:** successful and unsuccessful bridges are equally traceable.

## Checkpoint I — Interpretability adjudication

- [ ] Decide whether “policy-to-value” has become a genuine mechanistic interpretability method, a behavioral/counterfactual interpretability method, or both. Update the project title/subtitle only after this result.

---

# Phase IX — Synthesis, release, and future branches

- [ ] **110. Reconcile the complete claim ledger.**
  - Give every atomic claim a final current disposition, scope, evidence, and next experiment or proof obligation.

- [ ] **111. Package a reproducible policy-to-value benchmark.**
  - Include exact discrete tasks, one continuous task, one sequence task, adapters, recovery contracts, metrics, and artifact schemas.

- [ ] **112. Create end-to-end documentation and tutorials.**
  - Provide one minimal example for decoder value, revealed value, return value, repeated recovery, value atlas, and stochastic completion.

- [ ] **113. Write the formal paper.**
  - Center exact representation, equivalence/normalization, revealed-ranking recovery, stability bounds, and constructive limits.

- [ ] **114. Write the empirical/interpretability paper.**
  - Center repeated-recovery convergence, value-atlas leverage, stochastic completion, cross-policy adapters, and the strongest mechanistic result if any.

- [ ] **115. Conduct an adversarial proof, statistics, and software audit.**
  - Re-run from clean environments, inspect leakage and seed lineage, check theorem/implementation agreement, and remove unsupported language.

- [ ] **116. Publish a versioned release and archival artifacts.**
  - Tag code, freeze reports/configs, archive machine-readable results, and record exact hardware/software environments.

- [ ] **117. Select the next research branch from evidence.**
  - Candidate branches include value-logic integration, active preference interrogation, LLM/agent value atlases, mechanistic value circuits, multi-agent value reconstruction, and value-based policy editing.
  - **Done when:** the selected branch follows from completed evidence rather than the original roadmap alone.
