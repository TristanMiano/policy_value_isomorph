# Policy–Value Isomorph Research Program — TODO

Last updated 2026-11-20

## Resume here

**Next is Task 49**, defining how the project queries a policy (the player's rule for choosing moves). Phase II and Checkpoint C are complete. [Proofs](formalism/48_finite_checks.md) · [Checkpoint findings](CHECKPOINT_C.md)

Complete exactly one numbered task or one checkpoint per focused work session unless Tristan explicitly requests a larger batch. Preserve completed history, record negative results, and treat checkpoint revisions as part of the research rather than as administrative cleanup.

## Main philosophical stance

A **source policy** is the original rule that chooses moves or gives them probabilities. A **value** is a numerical score assigned to a move or its consequences. A **decoder** converts scores into moves or move probabilities. Choosing a highest-scoring move is called **argmax**. A **rollout** plays a game forward under a specified policy. Its average payoff over possible continuations is the **expected return**. Write `V(s)` for that average from position `s` and `Q(s,a)` for the average after first taking move `a` there. A **multilayer perceptron (MLP)** is a neural network that transforms inputs into predictions using successive layers of adjustable weights.

This project investigates **value-first semantics**, the idea that values give meaning to a policy's choices.

> A policy is a discrete or probabilistic behavioral readout from a richer value-like landscape over actions, reachable states, continuations, or trajectories. Values are the primary semantic object; policy choices are obtained from them by `argmax`, softmax/sampling, thresholding, or another declared decoder.

The motivating semantic ladder is:

```text
Boolean truth {0,1}
    -> uncertain truth / expectation [0,1]
    -> general signed and unbounded value R
    -> policy or decision through argmax / sampling
```

Every closed bounded real interval with two distinct endpoints can be rescaled onto `[0,1]`. Open endpoints stay open. Signed real numbers also represent unbounded differences, costs and losses. Minimizing a training loss `L` has the same solutions as maximizing `−L` on the same parameters. A utility of complete plays requires a separate definition. The historical progression from hard thresholds to sigmoid/tanh and then ReLU-like activations is a motivating hypothesis to audit, not a premise to assume.

The long-range conjecture is not that every activation literally is utility. It is that neural networks may owe part of their generality and interpretability potential to computing in a rich real-valued value carrier and postponing discrete truth, class, or action readouts until late in the computation.

The project will not define success as recovering one metaphysically unique reward hidden inside every system. It investigates four meanings of a score, each with its own evidence.

1. **Decoder value:** a score used to produce a move or distribution. A constructed score can reproduce a known policy exactly. A fitted score has measured fidelity on a declared population.
2. **Revealed value:** a ranking or cardinalized score inferred from behavior under counterfactual action sets or perturbations.
3. **Return value:** expected outcome or cumulative return under a declared environment, horizon, perspective, and payoff contract.
4. **Mechanistic value:** an internal representation that causally mediates the source policy’s computation.

Linking any two meanings requires additional assumptions or observations. A supplied game payoff, for example, can be evaluated even when it predicts a different move from the source. Failure to identify a unique reward does **not** refute policy-to-value representation; it identifies the relevant equivalence class, missing query information, or extra assumptions needed for a stronger interpretation.

## Core research objects

The [notation and recovery contract](formalism/00_notation.md) defines these symbols and the record each experiment returns. Later formal files link to that reference and state any local specialization.

```text
h in H                     state, context, or full history
C(h) subset A              feasible candidate actions/outputs
P(. | h, C)                deterministic or stochastic policy
S_P(h, C, a) in extended R recovered move score on the offered set C
D(S_P, h, C)               declared decoder producing behavior
R(P, contract, seed)       recovery procedure returning a ValueArtifact
```

A `policy` is therefore any history-to-action transducer: a tabular controller, MLP, time-series forecaster, robot controller, autoregressive language model, tool-using agent, or another bounded decision procedure. For large or continuous output spaces, `C(h)` may be a generated candidate set and the primary universal object will usually be action/continuation value `Q(h,a)` or energy rather than a scalar state value alone.

A `ValueArtifact` must eventually contain not only a trained network, but also its decoder, normalization, recovery contract, supported domain, provenance, uncertainty/repeated-run information, and evaluation results.

## High-level theses

### PV-H1 — Universal value-form representation

For broad policy classes, an exact value-like representation exists after fixing a decoder and minimal structure on the action space.

- Finite deterministic policies admit exact action-score representations decoded by tie-broken `argmax`.
- Discrete stochastic policies admit log-score/energy representations decoded by softmax, using extended values for zero-probability actions when needed.
- Deterministic policies on metric action spaces admit distance-based scores such as `-d(a,P(h))`, or energy equal to positive distance.
- A history paired with its candidate action can carry the move score. Using only the resulting board requires compatible comparisons wherever different histories merge.

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
pi_(T,mu)(a|h) proportional to mu(a|h) * exp(S(h,a)/T),
```

where `S` is the move score, `mu` gives the base weights and `T` is a positive temperature. The finite-action limit as temperature approaches zero should recover a unique highest-scoring move with positive base weight.

**Research route:** prove which changes in scores or temperature preserve the move probabilities. Measure how much the resulting distributions vary across recovered values, how many different moves and states they explore, and how those quantities change with uncertainty in the estimated scores.

### PV-H6 — One recovery contract can span heterogeneous policy classes

The same conceptual interface—history, candidates, policy access, counterfactual query family, decoder, normalization, and evaluation semantics—should apply across discrete games, continuous controllers, forecasting models, autoregressive sequence models, and agents, even when the concrete recovery algorithm differs.

**Research route:** build adapters in increasing difficulty and publish a capability matrix specifying which semantics are supported under black-box, probability/logit, rollout, and white-box access.

### PV-H7 — External value recovery can become a neural interpretability program

For neural policies, externally recovered value structure may be predictable from, aligned with, or causally mediated by internal activations and gradients. Backpropagated `-loss` gradients are local marginal-value signals relative to the training objective and provide one bridge to test.

**Research route:** quantify variation across independent external recoveries. Measure how well internal activity predicts recovered values, how choices change when that activity is altered, and how both relationships change under a new objective.

## Minimum publishable path and expansion path

The first scientific milestone is the exact tic-tac-toe pilot in Tasks 60–61 below. The source policy supplies choices. The game supplies the transitions and win/draw/loss payoff. Measure how much move choice is recovered from that payoff, then test how much a compact utility inferred from choices can explain.

After the pilot, Tasks 23–59 compare the existing and PyTorch implementations, develop formal results, and build recovery methods. Tasks 62–72 repeat and extend the tic-tac-toe measurements with independently trained players. Tasks 73–81 study Connect Four. Task 81a adds 4×4 Reversi, a disc-flipping game in which the player with more discs at the end wins. Task 82 brings the three games together before the first publication decision at Checkpoint F. Broader domains and neural-mechanism studies begin at Task 83. Checkpoints can narrow that work in light of the results.

## Formulations to compare in games

Run each applicable comparison for deterministic and stochastic source policies. Cross the **meaning of the score** with **how it selects a legal move**. Use the same boards, opponents, and evaluation budgets within each comparison. Record what each method can observe during recovery and at deployment. An exact encoding that queries the source at decision time is a reference for a learned model that must act independently.

| Where the move score comes from | Greedy choice | Stochastic choice |
| --- | --- | --- |
| Direct behavioral score fitted to source actions or probabilities | Compare the learned legal argmax with an exact indicator-score encoding and a fixed tie rule. | Compare learned scores with the exact log-probability reference and sample the legal distribution. |
| Learned frequency of an observed transition, estimated from normalized visit counts | With the same weights compare the largest immediate score with the largest average learned whole-game utility over sampled continuations. Record both players' continuation rules and time decay. | Compare sampling from immediate scores with sampling from expected whole-game utilities at specified temperatures, with the continuation rule held fixed. |
| Expected return under the supplied game payoff | Compare exact tabular returns, learned successor-state `V`, and direct `Q`. Take the best move for the player to act. | Apply the same temperature sampler to expected returns. Measure the effect of changing the choice rule while holding the returns fixed. |
| Expected utility under one compact utility fitted from choices | Fit a shared utility of declared consequences, compute its expectation for each move, and maximize it. | Compare mixing among expected-utility maximizers, a specified random-utility model, and temperature sampling of the expected utilities. Each makes a different prediction about choice frequencies. |

For a von Neumann–Morgenstern (VNM) expected-utility model, each move induces a lottery over completed games under a declared environment and continuation policy. Take the expectation of the **same** cardinal utility over each lottery. Training targets based on final game payoffs already include an expectation over the source's play. A local visit frequency instead measures how often a move was chosen. A whole-game utility built from those learned scores must be evaluated over continuations. Classical maximization permits mixtures over tied best moves. Probabilities among ties require a mixing rule. Positive probability on a strictly worse move requires an additional choice model, such as variation in utility across decisions or an explicit soft-choice rule. Test these assumptions with observable choices over lotteries where available. [Selection semantics](docs/selection_semantics.md) · [Utility factorization](docs/policy_reconstruction_vs_utility_factorization.md)

For temperature sampling use `p(a|s) ∝ μ(a|s) exp(S(s,a)/T)` on legal moves, with `S = -E` under the energy convention. Record the base measure `μ`, support, score scale, temperature, and tie rule. Distinguish randomness in the source, randomness in the recovered choice, and Monte Carlo error in the score estimate.

## Work and evidence protocol

Keep each measurable question quantitative from design through reporting. Give counts, magnitudes, differences and applicable uncertainty, and define terms before use. Explore several readings of an open thesis. Pair the search for constructions with the search for limitations, and record a self-review of what the evidence establishes. [Mandatory research protocols](AGENTS.md#mandatory-research-protocols) · [How to apply the two adapted protocols](RESEARCH_PROTOCOLS.md)

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

At each checkpoint, audit work since the previous checkpoint, record the decision, revise later tasks when justified, set the next-task pointer, and stop. Keep early decisions in this file until Task 31 creates the claim ledger, then link them from the ledger.

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

# First milestone — Exact tic-tac-toe reconstruction on the current backend

- [x] **60. Freeze the source policies, opponents, and pilot protocol.**
  - Define and save five original players. The **priority-rule player** takes an immediate win, blocks one, takes the center, takes a corner, or takes any legal cell, in that order. The **perfect-play player** chooses the best guaranteed outcome assuming the opponent always makes its best reply. The **missed-win player** deliberately takes the first move that does not win immediately when an immediate win is also available, otherwise it follows the priority rule. The **trained neural player** uses a network taught the priority player's choices on 528 boards and plays its highest-scoring legal move. These four players resolve ties by the lowest cell number, counting across rows from 0. The **randomized priority player** independently uses the priority rule with probability 3/4 on each turn and otherwise draws uniformly from legal cells.
  - Save four equally weighted opponent groups. One chooses a legal cell uniformly each turn. The second contains five untrained networks, each with fixed randomly initialized weights, choosing its highest-scoring legal cell with the lowest-numbered one breaking ties. The other groups are the priority-rule and perfect-play players. Reserve independent random seeds (integers that reproduce random choices) for confirmation.
  - Fix the opening, equal weights for playing as X and O, opponent weights, tie rules and numerical comparisons. **Move agreement** is the fraction of boards with the same chosen move. For a randomized source, **total variation distance** is half the sum of the absolute differences in legal-move probabilities, averaged over boards. Report equal weights per reachable board and weights proportional to the probability that the source reaches each board playing both sides. Keep rotations and reflections together when assigning boards to training, validation and test groups. Preserve choices for each orientation and give distinct board and visit counts.
  - Deliver a reproducible pilot manifest that identifies every source, opponent, split, game payoff, and continuation policy. Record how to reproduce both the full-board exact reference and the held-out learned comparison.
  - Completed 2026-11-17. Five sources, eight individual opponents, independent confirmation seeds, and 2,980 training, 760 validation and 780 test decision boards were fixed before the pilot. The reporting rules and measure descriptions were revised after data collection. [Study settings and definitions](experiments/tictactoe_pilot/manifest.json) · [Saved player weights](experiments/tictactoe_pilot/weights.json)
  - Rebuilt the checkpoints with `python scripts/freeze_tictactoe_pilot.py experiments/tictactoe_pilot/weights.json`. The protocol and network helpers are in `pilot_protocol.py` and `policy_mlp.py`, with coverage in `test_pilot_protocol.py`. The demo passed and all 50 tests passed on Windows after correcting the path separator assertion in `test_telemetry_io.py`.

- [x] **61. Measure exact reconstruction and playing strength across the tic-tac-toe graph.**
  - Enumerate every reachable nonterminal board and legal move. Compute exact continuation values for each frozen source, using both players' source policies in self-play and the existing payoff perspective. For stochastic play, integrate over legal moves with their probabilities. Verify successor `V` and direct `Q` agree under this deterministic transition and terminal-payoff convention.
  - On each board retain the source action or distribution, the exact behavioral decoder output, every return-maximizing move for the acting player, and the move selected by the fixed tie rule. Distinguish exact matches, alternative tied moves, and strictly better one-move deviations. For stochastic sources also record source probability mass outside the maximizing set.
  - Run the existing learned policy, `V`, and `Q` comparison on distinct held-out positions against those exact references. Generate each set of rollout labels with the very source being reconstructed, including the frozen learned policy. Report calibration and which actions change when learned values replace exact ones.
  - Play the **original source and each reconstructed policy** for whole games against the same opponents, openings, and seats. Report absolute win, draw, and loss probabilities per opponent and seat, expected game score `P(win) - P(loss)`, and paired differences from the source. Compute exact probabilities where the move distributions are known. For sampled games report counts and paired uncertainty, keeping opponent-network seeds distinct from game seeds. Include separate head-to-head rows against the source.
  - Publish a per-board table and a per-matchup table with source rows and both state weightings for fidelity. Include a worked board where the tie rule matters and a deliberate mistake where exact return maximization changes the move.
  - Completed 2026-11-17. Computing the expected payoff from the board after a move and computing it directly for the board and move gave a difference of zero on all 16,167 legal moves for each source. On the 780 equally weighted test boards, a network trained to copy the trained neural player's moves matched 709 (90.90%). A network trained to predict payoff from the resulting board matched 394 (50.51%). A network trained to predict payoff from the current board and move matched 414 (53.08%). [Pilot findings and worked boards](TICTACTOE_PILOT.md) · [Move comparisons](experiments/tictactoe_pilot/results/fidelity.csv) · [Game outcomes](experiments/tictactoe_pilot/results/matchups.csv)
  - `python scripts/run_tictactoe_pilot.py` generated the six result files. A fresh replay reproduced them byte for byte. The exact, recovery, and reporting code is in `pilot_exact.py`, `pilot_recovery.py`, and `pilot_metrics.py`. The demo passed and all 56 tests passed on Windows and Ubuntu, including `test_pilot_exact.py` and `test_pilot_results.py`. [Published fixture hashes](experiments/tictactoe_pilot/results/run.json)

## Checkpoint before the PyTorch migration

Treat the size of the latest tic-tac-toe training gains as provisional. The frozen player takes the lowest-numbered move that preserves the best outcome against perfect replies. The reconstruction scores each candidate move given the board and takes its highest-scoring move. Increasing its training set from 2,980 to 3,740 boards at the same number of training updates raised its average agreement with the frozen player from 96.92% to 97.78% on the same 780 equally weighted boards withheld from training. The increase was 0.85 percentage points with a standard error of 1.06 points (the sample standard deviation of the three paired differences divided by the square root of three). Each fit's agreement is an exact count on those boards. The uncertainty here concerns variation across training runs with this one player and board split. Task 62 plans independent repetitions. [Training-set comparison](TICTACTOE_CHOICE_TRAINING.md#how-much-does-the-number-of-distinct-training-boards-matter)

The neural follow-up verified the trained player's guaranteed outcome on every reachable decision board. It then sampled complete games under the fixed player and trained networks to predict normalized transition counts. It compared direct move selection with the expected learned utility of sampled plays using the same weights. Keep the empirical counts, coverage, paired decoders, different network inputs and separate payoff-trained reference in Tasks 62, 66 and 71 and in the independent confirmation. [Training procedure and side-by-side results](NEURAL_TICTACTOE_CONTROL.md)

On 2026-11-19 we compared starting-board rules. The frozen neural player takes the lowest-numbered move with the best guaranteed outcome on every decision board. At 8,192 observed source moves, favoring the least visited of 16 random legal starting positions increased direct move agreement for a network scoring a board and candidate move by 3.82 percentage points over ordinary random starts. The standard error of the mean difference across five paired fits on the same 780 test boards was 0.44 points. One start at each enumerated board reproduced the earlier 32-start move frequencies exactly for this player. The report gives the costs, tests on rarely reached positions, results for randomized choices and game outcomes. [Experiment and archived measurements](TICTACTOE_SAMPLING.md)

- [x] **Pilot checkpoint completed 2026-11-17.** On all 4,520 boards, choosing the move with the best exact return matched the perfect-play, missed-win, priority-rule and trained neural players on 100.00%, 37.70%, 75.93% and 56.62% of boards respectively. Weighting boards by visits in the source's own games gave 100.00%, 33.33%, 77.78% and 55.56%. For the randomized priority player the corresponding total variation distances were 0.2764 and 0.5381. The report gives the learned models' results on the same 780 test boards and the number reached by each source. [Move agreement and probability differences](TICTACTOE_PILOT.md#move-agreement-and-probability-differences)
  - For the frozen neural source, 1,094 of 4,520 exact choices differed only by a return tie, and 867 source moves had strictly lower return. Replacing exact scores with learned V restored the source move on 95 test boards and lost it on 145. The exact behavioral decoder matched all five source distributions. The fixed game payoff therefore explains move choice to different degrees across these policies. [Board-level attribution](TICTACTOE_PILOT.md#why-the-moves-differed)
  - The source weights were fixed in revision `eaf79b3`, and the measurements were first recorded in `bc74ccf`. Replaying the revised study preserved all previously measured values in the 240 move-comparison rows and all 900 game-outcome rows. The added payoff comparison measures the average points forgone by each player's chosen move relative to the best legal move, assuming the source plays both sides thereafter. An independent calculation from all 22,600 board records reproduced the two added averages for all 240 rows. All 58 software tests passed on Windows and Ubuntu, and the demo printed three sample boards, their payoffs and the chosen moves. [Reproducing the measurements](TICTACTOE_PILOT.md#reproducing-the-measurements)

---

# Phase 0 — Finish the PyTorch foundation

- [x] **23. Introduce minimal internal interfaces for PyTorch-backed models and training steps.**
  - Add explicit model, optimizer, batch, device, and prediction boundaries shared by policy, `V`, and `Q` without building a large framework.
  - Preserve pure data preparation and evaluation functions.
  - **Done when:** interfaces support the current one-hidden-layer models and are covered by focused shape/type tests.
  - Completed 2026-11-20. The shared model reads the original weight layout, accepts rows of board or board-and-move inputs, and exposes an update step that accepts the optimizer (the rule for changing weights). Tests cover all three input/output shapes, legal moves, scalar targets, CPU placement, 64-bit precision and independent weight snapshots. All 77 tests passed with `python -m pytest -q`, and `python scripts/demo_tictactoe.py` printed the three sample boards, payoffs and moves. [Model and training step](src/policy_value_isomorph/torch_mlp.py) · [Interface tests](tests/test_torch_mlp.py)

- [x] **24. Port policy, value, and Q MLPs to PyTorch with parity fixtures.**
  - Preserve public APIs, legal-action masking, perspective/sign conventions, and telemetry.
  - Keep the first architecture equivalent to the current tanh MLP.
  - Verify forward behavior and small deterministic training fixtures within declared tolerances. Replay the Task 61 frozen inputs and checkpoints to compare legal choices, ties, value signs, and resulting matchup scores across backends.
  - Completed 2026-11-20. The public trainers and prediction functions now use the shared model. For each of the three network types, the highest-scoring legal move for the player to act matched on 22,600/22,600 archived player–board cases. The maximum absolute win, draw or loss probability difference over 900 matchup rows was 2.22 × 10⁻¹⁶. A separate four-example training comparison measured a largest absolute weight change between implementations of 5.55 × 10⁻¹⁷ across 170 weights and biases. All 82 tests passed with `pytest -q`, including saved-weight replay and continuation from old and new checkpoints. The installed training command and `python scripts/demo_tictactoe.py` also ran. [Implementation, definitions and quantitative comparisons](PYTORCH_MIGRATION_AUDIT.md#replaying-the-frozen-tic-tac-toe-networks) · [Saved replay](experiments/tictactoe_pilot/torch_replay.json)

- [x] **25. Add configurable AdamW training through `torch.optim.AdamW`.**
  - Add device selection, dtype, seeding, and deterministic settings where available.
  - Expose optimizer parameters through existing CLI paths.
  - **Done when:** all three training paths run end to end and baseline commands are documented.
  - Completed 2026-11-20. All three trainers accept the same device, precision, seed, repeatability and optimizer settings. Their adaptive updates and losses were checked against PyTorch's own AdamW on three passes for each network. The command-line training example exercised all three paths. All 85 tests and the tic-tac-toe demo ran. [Configuration](src/policy_value_isomorph/training_config.py) · [Numerical tests](tests/test_adamw.py) · [Baseline command](README.md#pytorch-first-migration-decision-task-22)

- [x] **26. Evaluate Muon only after AdamW parity is complete.**
  - Keep this optional and config-gated; do not fork the training stack.
  - Compare maintenance burden, convergence, wall time, and final behavior on a frozen small benchmark.
  - **Done when:** retain or reject Muon with a recorded result; rejection completes the task.
  - Completed 2026-11-20. Compared AdamW with Muon on the hidden matrix in 24 CPU fits. Over four paired initializations, Muon changed average move agreement by +1.57, +2.08 and −2.88 percentage points for the move, board and board-and-move networks on 780 withheld boards. Retained the opt-in comparison. The report gives uncertainty, all losses, times and game outcomes. All 85 tests and the demo ran. [Benchmark, results and repeat command](experiments/optimizer_comparison/README.md)

- [x] **27. Migrate checkpoint/resume to PyTorch model and optimizer state.**
  - Version the artifact schema and provide a clear legacy policy.
  - Preserve latest/specified checkpoint behavior and telemetry continuation.
  - **Done when:** round-trip and interrupted-run tests pass for policy, `V`, and `Q`.
  - Completed 2026-11-20. Version 2 saves native PyTorch model and optimizer state, example order, random state, dataset identity and elapsed time. The interrupted and uninterrupted AdamW fits produced identical final weights, optimizer memory and subsequent losses for each of the three networks using both latest and named files. Historical JSON continuation also ran. All 92 tests and the demo passed. [Checkpoint implementation](src/policy_value_isomorph/checkpointing.py) · [Continuation tests](tests/test_training_resume.py) · [Resume commands](REPRODUCIBILITY.md#continuing-a-training-run)

- [x] **28. Expand tests and smoke demos for the PyTorch stack.**
  - Add CPU end-to-end tests for policy training, rollout labeling, `V`, `Q`, evaluation, and resume.
  - Verify telemetry paths on Windows and POSIX when porting the checkpoint stack.
  - Mark any GPU-only checks separately.
  - **Done when:** `pytest -q` and a documented smoke command pass from a fresh editable install.
  - Completed 2026-11-20. A short end-to-end program freezes a trained player, generates rollout labels with it, fits and resumes both payoff networks, and plays both seats. Tests read all three trainers’ checkpoints and both log formats through native paths. With fresh editable installations, Windows passed 93 tests and skipped the GPU check on its CPU-only PyTorch build. Ubuntu passed all 94, including the separately marked NVIDIA GPU check. The short program and the three-board demo ran on both platforms. [Runnable check](scripts/smoke_torch_training.py) · [Cross-platform tests](tests/test_torch_smoke.py)

- [x] **29. Update dependencies, README, reproducibility docs, and examples for the PyTorch-first stack.**
  - Replace stale “no neural nets/no dependencies” language in `AGENTS.md`, `DESIGN.md`, `REFERENCES.md`, and package metadata.
  - Document optional dependency groups planned for control, sequence, LLM, and visualization work.
  - **Done when:** a new contributor can reproduce the current baseline without relying on historical Task 1 instructions.
  - Completed 2026-11-20. Updated package metadata, contributor instructions, design, references, installation and reproduction examples. The guide identifies installed extras and future dependency groups separately. In a fresh Windows environment the documented editable installation, dependency check, train, evaluation, checkpoint/resume, short workflow and demo commands ran. The suite passed 93 tests with the one GPU-only check skipped. Ubuntu passed all 94 tests, including that check, during Task 28. [Quickstart](README.md#run-the-current-baseline) · [Reproduction guide and planned groups](REPRODUCIBILITY.md#dependencies-and-optional-groups)

## Checkpoint A — Backend and scope readiness

- [x] Audit Tasks 23–29, run the full suite, and freeze the package interfaces that the research phases may rely on. Confirm that the new philosophical stance does not require changing existing value-sign conventions before the formalism is written.
  - Completed 2026-11-20. Later experiments can rely on the documented calls for board representation, generating game-payoff labels, training and freezing players, selecting moves, evaluating games and continuing saved fits. Freezing now copies the player's weights. Connect Four now lets the caller identify whose win has a positive payoff, with X as the default. [Frozen interfaces and regression tests](CHECKPOINT_A.md)
  - Retain +1 for a win, 0 for a draw and −1 for a loss from one named player's viewpoint. For the tic-tac-toe rule that wins immediately, blocks a win, takes the center, takes a corner or takes any legal move, in that order with lowest-cell ties, changing that viewpoint and reversing the direction of selection changed 0 of 4,520 decisions. Playing forward and working backward from terminal boards gave identical payoffs on all 16,167 legal board–move pairs for each viewpoint. A move's observed share of visits measures how often the mover chose it. The checkpoint provides an implementation check for the payoff calculation. [Exact calculation](tests/test_pilot_exact.py)
  - Windows passed 96 software tests and skipped the one graphics-processor test on a central-processor-only installation. Ubuntu passed all 97, including that test. The three-board demo and short training workflow ran. Task 30 is next.

---

# Phase I — Freeze the research constitution and literature boundary

- [x] **30. Create `RESEARCH_STANCE.md`.**
  - State the value-first position, semantic ladder, bounded-interval affine collapse, loss/value duality, exact Boolean/readout recovery from real-valued computation, four levels of value form, and the distinction between existence, usefulness, identifiability, and mechanism.
  - Cross-link the companion `value_logic` project without making it a code dependency.
  - **Done when:** the document makes clear that reward nonuniqueness narrows a claim rather than refuting the base representation thesis.
  - Completed 2026-11-20. Defined the four interpretations of a score, the bounded-interval change of coordinates and the scope of behavioral existence. Adapted exploration and aim preservation for this project and linked the mandatory protocols from the contributor guide and README. [Research stance](RESEARCH_STANCE.md) · [Protocols](RESEARCH_PROTOCOLS.md)

- [x] **31. Create `CLAIM_LEDGER.md`.**
  - Give PV-H1–PV-H7 stable identifiers, scoped statements, dependencies and project impact. For mathematical claims state the proof obligation or counterexample. For empirical claims specify the measurements, their expected direction and magnitude, and how the estimates and their uncertainty bear on the claim.
  - Split broad theses into approximately 20 atomic claims.
  - Track exact behavioral representation, recovery from the supplied payoff, and recovery of a compact utility from choices as separately testable claims. Link each to its population, evidence, and possible negative result.
  - **Done when:** each empirical entry states its planned measurement or gives the measured size, uncertainty and scope of an observed effect.
  - Completed 2026-11-20. Recorded 21 scoped claims in seven families, with assumptions, dependencies, proof obligations and quantitative predictions. Linked the pilot and Checkpoint A and distinguished full-table imitation, supplied payoff and compact learned utility. [Claim ledger](CLAIM_LEDGER.md)

- [x] **32. Define the policy-access and value-semantics taxonomy.**
  - Create `docs/policy_value_taxonomy.md` covering deterministic action access, samples, probabilities/logits, constrained-choice access, rollout access, environment access, and white-box internals.
  - Define decoder, revealed, return, and mechanistic value artifacts.
  - **Done when:** every later adapter can declare capabilities without overloading “value function.”
  - Completed 2026-11-20. Defined seven forms of access and four score meanings. Documented how to declare what each procedure sees during fitting and deployed play, including controls for changed menus and source-assisted continuations. [Access and value taxonomy](docs/policy_value_taxonomy.md)

- [x] **33. Audit the primary literature and expand `REFERENCES.md`.**
  - Cover policy evaluation/improvement, policy distillation, inverse decision/reward modeling, maximum-entropy control, reward invariances, revealed preference/choice, value equivalence, imitation distribution shift, representation similarity, explainable RL, many-valued/fuzzy neural interpretations, and the historical activation/loss story.
  - Use primary papers and record exactly what is imported versus merely analogous.
  - **Done when:** each load-bearing concept in PV-H1–PV-H7 has a verified source or an explicit original-proof obligation.
  - Completed 2026-11-20. Audited primary work on decisions, imitation, reward ambiguity, explanation, neural representations and logic. Each of the seven claim families has a source boundary and a project-specific proof or measurement to pursue. [Annotated literature audit](REFERENCES.md)

- [x] **34. Freeze notation and the recovery contract.**
  - Create `formalism/00_notation.md` defining histories, candidate sets, policies, value forms, decoders, recovery procedures, equivalence, normalization, and `ValueArtifact`.
  - Include discrete, continuous, and sequence examples.
  - **Done when:** later formal files link to one authoritative notation source.
  - Completed 2026-11-20. Fixed terminology for histories, measures, decoders, menu-relative equivalence and utility. Specified the recovery record and worked through discrete, continuous and sequence inputs. Linked the existing formal notes to the authoritative notation. [Notation and recovery contract](formalism/00_notation.md)

- [x] **35. Define numerical comparisons for imitation, playing strength and interpretability.**
  - In `docs/evaluation_contract.md` define three decisions in advance. For **fidelity**, use held-out action agreement for deterministic sources and mean total variation distance between legal action distributions for stochastic sources. Also report ties, action gaps, cross-entropy and support violations where applicable.
  - For **playing strength**, rank by expected game payoff against the predeclared opponent mix. Publish the separate ranking by win probability, plus per-opponent, per-seat and head-to-head results. Deploy each recovered policy for the whole game. One-step return under the source continuation is a separate measurement.
  - For **interpretability**, measure how accurately an explanation predicts new source choices after moves are removed or outcome probabilities change. Compare with the source network's raw move scores, a model trained to copy its moves, and a simple decision rule given the same information. Report prediction error, explanation size, and variation across independent fits for each. Use the measured trade-offs to explain a choice of formulation.
  - Specify the held-out populations, paired differences (measurements made on the same positions or opponents), sampling uncertainty, and how exact ties will be reported for each objective. Relate differences in move order, score magnitude and score patterns across nearby boards to these measurements. Claims about human usefulness or internal mechanism require their own evidence.
  - Completed 2026-11-20. Defined separate prospective rankings for imitation, points per game, win probability and predictions of changed choices. Fixed comparison players, populations, weights, tie handling, matched access and paired uncertainty. [Evaluation contract](docs/evaluation_contract.md)

- [x] **36. Produce a small hand-worked policy/value atlas.**
  - For one tic-tac-toe state, manually show the source action, all legal alternatives, exact/rollout/revealed values, action gaps, normalizations, and one softmax completion.
  - **Done when:** the example exposes every ambiguity the formalism must handle before more code is written.
  - Completed 2026-11-20. Worked through every legal move and complete continuation on one board. The source picks cell 5 before 8 before 2 on that menu. Cells 2 and 5 both win when two copies of the source continue. Under uniformly random legal replies by both seats their win probabilities are 1 and ½. Normalized counts, restricted-menu rankings, sampled choices and VNM complete-play utilities give distinct comparisons. [Hand-worked example](docs/hand_worked_atlas.md) · [Executable verification](tests/test_hand_worked_atlas.py)
  - Phase I verification. For each of Tasks 30–35 the Windows suite ran 96 tests successfully and skipped the graphics-processor test, and the three-board demo ran. With the Task 36 example included, `python -m pytest -q` passed 97 tests on Windows with that one skip and all 98 on Ubuntu, including the graphics-processor test. `python scripts/demo_tictactoe.py` printed all three sample boards, payoffs and moves. The exact atlas calculations cover one illustrative position. Checkpoint B is pending.

## Checkpoint B — Thesis and notation freeze

- [x] Reviewed Tasks 30–36 on 2026-11-20. The proof plan covers exact encoding, limits on sharing board scores, equivalence under declared queries, expected utility, and stability of decoded behavior. Each is a planned obligation with assumptions and limiting examples. [Frozen proof plan](CHECKPOINT_B.md#proofs-to-pursue)
  - The minimum game comparison retains tic-tac-toe, Connect Four and 4×4 Reversi. Compare direct and expected-utility choices from the same fitted networks, with separate measurements for imitation, full-game performance and predicting fresh choices under changed conditions. Separate known game points, utility inferred from choices and claims about internal causes. [Experiment decisions](CHECKPOINT_B.md#minimum-experiments)
  - Corrected an unsupported ranking of fitted methods and a sign error in the definition of energy. Recounting the fixed-rule tic-tac-toe pilot gave 853 disagreements at equal return and 235 at lower return among 4,520 boards for the win/block/center/corner player defined in the review. Task 37 is next. [Findings, evidence and verification](CHECKPOINT_B.md)

---

# Phase II — Prove the value-form and equivalence results

- [x] **37. Prove finite deterministic policy-to-score representation.**
  - For finite feasible action sets and deterministic `P`, construct an exact real action-score representation with fixed tie-breaking.
  - Characterize the encoder image and inverse decoder.
  - **Done when:** theorem, proof, examples, and executable finite tests agree.
  - Completed 2026-11-20. Characterized the canonical indicator tables, their inverse and every score tie that preserves a chosen move. [Statement and checks](formalism/37_finite_choices.md). Verified with the executable examples, the three-board demo and the full Windows suite (97 passed, 1 skipped).

- [x] **38. Prove discrete stochastic policy-to-log-score representation.**
  - Represent a stochastic policy by logits/energies up to state-dependent additive constants; handle zero support with extended reals or a declared approximation.
  - State the role of temperature.
  - **Done when:** softmax recovery and invariance properties are proved and tested.
  - Completed 2026-11-20. Derived the score family for a fixed temperature and base measure, including zero support and a finite-score approximation with explicit probability error. [Statement and checks](formalism/38_random_choices.md). Verified with the executable examples, the three-board demo and the full Windows suite (97 passed, 1 skipped).

- [x] **39. Extend exact representation to deterministic metric-action policies.**
  - Use a score equal to negative distance or an energy equal to positive distance and state assumptions for unique decoding, measurability, and candidate restriction.
  - **Done when:** finite-dimensional continuous-action examples and edge cases are explicit.
  - Completed 2026-11-20. Proved unique distance decoding and quantified the error from restricting the candidate set, including absent maxima and measurability conditions. [Statement and checks](formalism/39_metric_actions.md). Verified with the executable examples, the three-board demo and the full Windows suite (97 passed, 1 skipped).

- [x] **40. Prove action-value to state-value lifting.**
  - Show how `Q(h,a)` can be represented as state value on an action-augmented history or successor object.
  - State when an ordinary `V(h)` plus a transition/action harness is sufficient.
  - Characterize when one score per resulting board can reproduce every source choice under a fixed rule for ties. Give an exact example and a counterexample with conflicting move orders. In the counterexample, give the highest attainable fraction of reproduced moves or bounds on it. Compare with scores that also take the action as input.
  - **Done when:** the theorem covers the current deterministic game convention and a general history-based form.
  - Completed 2026-11-20. Separated exact score lifting from recovering the same choice. Characterized finite successor feasibility with strict and weak comparisons and bounded a conflicting example exactly. [Statement and checks](formalism/40_successor_scores.md). Verified with the executable examples, the three-board demo and the full Windows suite (97 passed, 1 skipped).

- [x] **41. Define decoder-relative value equivalence and prove the quotient result.**
  - Two values are equivalent when every declared decoder/query consumer sees the same result.
  - Prove that exact representations factor through the response quotient and identify common invariant transformations.
  - **Done when:** deterministic argmax, softmax with known temperature, and threshold consumers have separate examples.
  - Completed 2026-11-20. Identified the response quotient and the inverse on its attainable responses. Different consumers preserve different information. [Statement and checks](formalism/41_response_equivalence.md). Verified with the executable examples, the three-board demo and the full Windows suite (97 passed, 1 skipped).

- [x] **42. Prove normalization and scale/temperature lemmas.**
  - Analyze additive anchors, positive affine transforms, rank/quantile normalization, and the equivalence between value scale and softmax temperature.
  - **Done when:** the project has a justified comparison protocol rather than raw-output RMSE between arbitrary gauges.
  - Completed 2026-11-20. Proved the effects of offsets, scale, temperature and rank normalization on choices, probabilities and lotteries. [Statement and checks](formalism/42_score_units.md). Verified with the executable examples, the three-board demo and the full Windows suite (97 passed, 1 skipped).

- [x] **43. Prove revealed ranking recovery from constrained choice sets.**
  - If a deterministic policy chooses the unique maximum of one stable ordering from every offered finite subset, prove that pairwise or elimination queries recover the total order.
  - Record consistency tests and counterexamples when context effects violate the assumption.
  - **Done when:** this yields an implementable black-box preference-probing algorithm.
  - Completed 2026-11-20. Proved pairwise and elimination recovery under a stable strict order, with query counts and explicit tests for context effects. [Statement and checks](formalism/43_menu_queries.md). Verified with the executable examples, the three-board demo and the full Windows suite (97 passed, 1 skipped).

- [x] **44. Prove score-difference recovery for soft/stochastic choice.**
  - Under a declared Luce/softmax model, recover pairwise score differences from probability or repeated-choice ratios.
  - Separate ordinal recovery, cardinal differences, unknown temperature, and finite-sample uncertainty.
  - **Done when:** exact and sampled fixtures reproduce the derivation.
  - Completed 2026-11-20. Derived adjusted log odds and connected-menu identifiability. Checked exact probabilities and finite repeated draws with uncertainty. [Statement and checks](formalism/44_probability_ratios.md). Verified with the executable examples, the three-board demo and the full Windows suite (97 passed, 1 skipped).

- [x] **45. Formalize expected utility over trajectories and its limits for imitation.**
  - Define `Q_U^κ(s,a) = E[U(trajectory) | s,a,environment,κ]` with a declared continuation rule, horizon, information, player perspective and evaluation distribution. Derive the current rollout target as a special case. Compare a supplied game payoff with a single consequence utility inferred from choices.
  - For deterministic sources establish when exact expected-utility maximization reproduces the action under the declared tie rule. For stochastic sources derive the best distributional match possible when choices must stay within the maximizing set. Compare specified tie mixing and random utility with soft choice, whose additional preference over choice distributions gives the objective `E_q[Q] − T KL(q || μ)`.
  - Give finite examples of an exact match, a missed win under the game payoff, and two actions inducing the same consequence lottery. Show how restrictions on the information available to the utility change what the policy can reveal.
  - Completed 2026-11-20. Characterized deterministic and stochastic recovery from shared utilities. Derived the unavoidable probability discrepancy and the temperature-sampling objective. [Statement and checks](formalism/45_expected_utility.md). Verified with the executable examples, the three-board demo and the full Windows suite (97 passed, 1 skipped).

- [x] **46. Prove stability bounds linking value error, action gaps, and policy agreement.**
  - For a unique best legal move, show why an action gap larger than twice the largest per-action score error on the board preserves that move. Give examples where the same mean squared error preserves or reverses a ranking. Treat ties explicitly.
  - Bound or exhibit the sensitivity of soft-choice probabilities to score error at fixed temperature and base measure. State which bounds concern agreement with the exact decoder and when that decoder also matches the source. Test the finite examples against the diagnostics planned in Task 67.
  - Completed 2026-11-20. Proved deterministic margin and probability-shift bounds, including exact ties. Compared equal squared errors with different move effects. [Statement and checks](formalism/46_stability.md). Verified with the executable examples, the three-board demo and the full Windows suite (97 passed, 1 skipped).

- [x] **47. Build the constructive non-identifiability suite.**
  - Include equivalent argmax scores, unknown softmax temperature, unqueried alternatives, reward shaping/return ambiguity, insufficient state, and off-support behavior.
  - For each case, record the stronger interpretation that fails and the value-form result that survives.
  - **Done when:** future work cannot mistake one counterexample for a refutation of PV-H1.
  - Completed 2026-11-20. Constructed six pairs of indistinguishable explanations and recorded what each preserves, what is hidden and which extra observations can help. [Statement and checks](formalism/47_ambiguities.md). Verified with the executable examples, the three-board demo and the full Windows suite (97 passed, 1 skipped).

- [x] **48. Implement executable formal fixtures.**
  - Add a small exact module and tests for Tasks 37–47, independent of neural training.
  - **Done when:** one command checks every finite theorem example and normalization invariant.
  - Completed 2026-11-20. Unified executable examples and independent enumeration checks in a standard-library verification command. Linked theorem results to the claim ledger. [Statement and checks](formalism/48_finite_checks.md). Verified with the executable examples, the three-board demo and the full Windows suite (107 passed, 1 skipped, 11 subtests passed).

## Checkpoint C — Formal spine audit

- [x] Adversarially check every proof, boundary convention, and use of “isomorphism.” Freeze the exact positive theorem claims and the quotient language before building the generalized recovery API.
  - Completed 2026-11-20. The review fixes the reversible correspondence between groups of scores giving the same observable answers and the answer patterns themselves. It corrects the assumptions for sharing a board payoff and for interpreting repeated random choices. On all 4,520 tic-tac-toe decision boards the archived player chose the lowest-numbered move giving the best guaranteed outcome. Three distinct scores are necessary and enough to copy it with lowest-cell ties, compared with seven when each winner must lead outright. [Decision, counts, assumptions and verification](CHECKPOINT_C.md)

---

# Phase III — Build the generalized recovery and comparison framework

- [ ] **49. Define the `PolicyAdapter` protocol.**
  - Support history encoding, feasible/candidate actions, deterministic action, sampled action, optional probabilities/logits, optional constrained choice, optional transition/rollout, and optional internals.
  - **Done when:** tic-tac-toe and Connect Four satisfy the protocol without changing their game logic.

- [ ] **50. Define `RecoveryContract` and `ValueArtifact` dataclasses.**
  - Record semantics level, access mode, query/state distribution, candidate generator, decoder, normalization, provenance, uncertainty, and supported scope.
  - For game artifacts link the frozen source, utility specification, continuation and opponent policies, tie or mixing rule, temperature, split, and opponent-panel manifest to the evaluation rows.
  - **Done when:** artifacts are serializable and reject semantically incomplete combinations.

- [ ] **51. Implement exact decoder-value encoders.**
  - Add deterministic indicator/gap scores, stochastic log scores, and positive-distance energies minimized over metric actions.
  - **Done when:** round-trip tests recover the source behavior exactly on supported domains.

- [ ] **52. Implement policy-to-score distillation.**
  - Train value/action-score networks directly from actions, probabilities, logits, or pairwise choices according to access level.
  - **Done when:** each supervision mode is a separate named recovery method with matched evaluation.

- [ ] **53. Implement return evaluation and fitting a compact trajectory utility.**
  - Reuse the exact tic-tac-toe reference and compare Monte Carlo and optional bootstrapped estimates under the same payoff and continuation. Expose the induced outcome lotteries so a new utility can be evaluated without summing continuation values along a trajectory.
  - Fit one nonconstant utility shared across boards to training choices under a declared decoder. Start with anchored win/draw/loss utilities and a small prespecified extension such as game length. Restrict inputs to consequences and publish the features, parameter count, scale anchors and policy-independent tie rule. Keep source actions and board identities out of the utility inputs.
  - Verify the implementation on a source with a known utility and on choices the restricted family cannot explain. Retain multiple compatible utilities or bounds when the observations admit them. Task 71 tests predictive value on held-out choices.

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
  - Implement Task 35's fidelity and strength measures, including a source row, absolute rates and paired differences. For return models recompute residual error at each saved checkpoint on fixed training targets and error on held-out boards with exact targets. Log action disagreement, action gaps and matchup outcomes at the same checkpoint.
  - Retain within-state rank correlations and top-k overlap, aligned cardinal error, local geometry, return calibration, and counterfactual prediction scores. Each metric declares its population, invariances, access requirements, and units.

- [ ] **58. Build the repeated-recovery experiment orchestrator.**
  - Factor seeds for data, rollouts, initialization, optimizer order, architecture, and candidate generation.
  - Store manifests and pairwise/barycenter comparison tables.
  - **Done when:** a small matrix resumes safely and detects accidental seed reuse.

- [ ] **59. Build the first `ValueAtlas` reporting API.**
  - Emit per-context action tables, value gaps, uncertainty across recoveries, local state/action neighborhoods, and source-policy comparison.
  - **Done when:** a deterministic policy yields more inspectable information than a single chosen action.

## Checkpoint D — Infrastructure and experiment freeze

- [ ] Validate Tasks 49–59 against the Task 61 fixtures. Verify that the source, direct decoder, and expected-utility variants can be evaluated in the same tables. Task 62 freezes the confirmation protocol before final runs.

---

# Phase IV — Exact discrete-domain recovery and convergence experiments

Tasks 60–61 run before the backend migration. Use their graph, manifests, and pilot findings to design independent confirmation runs here.

- [ ] **62. Freeze the first confirmation protocol.**
  - Carry forward the [tic-tac-toe starting-position comparison](TICTACTOE_SAMPLING.md). Reserve games starting from the natural opening and games starting from positions generated by legal play at varied depths. Compare uniform draws from those candidates with preference for the least observed among several candidates. Count only source moves. Fix the budget for starting-position generation, source observations, validation and independent repeats per board. Report distinct boards observed, fractions of normal play represented, agreement on common, rare and unreachable positions under the declared opponents, and error in move probabilities for randomized sources. Report uncertainty across independent collections and fits. In a game small enough to enumerate, include one start at each training board as a reference for deterministic players.
  - Extend the pilot with independently initialized source and opponent networks, randomized players with known move probabilities, and players whose choice ordering depends on context. Before collecting confirmation data, specify the quantities and differences to estimate, board and opponent weights, numbers of simulated games and requests for source choices, independent repetitions, and how to estimate uncertainty. Fix the data-collection budget in advance.
  - Lock the three Task 35 selection rules. Match deployable models on available information, training positions, and capacity or compute budget where feasible. Record extra access such as logits or an environment model. Select checkpoints, utility features, temperatures and other settings on validation data. Reserve the test split and held-out interventions for the final comparisons.

- [ ] **63. Run exact decoder-score recovery baselines.**
  - Report exact behavioral encoding and learned score distillation side by side, with access to the source at inference identified. Add the frozen source as the reference row for both game strength and fidelity. Report approximation error on unseen boards at matched training budgets.

- [ ] **64. Run constrained-choice ranking recovery.**
  - Measure query complexity, ordering recovery, inconsistency diagnostics, and active-vs-random probing.
  - **Done when:** the report gives the fraction of alternative move pairs whose order was recovered, the number left undetermined and the number of requests for source choices.

- [ ] **65. Run stochastic/log-score recovery.**
  - Compare access to true probabilities or logits with repeated action samples at fixed query budgets. Include a mixture with zero support on a legal action and one that gives positive mass to a lower-return move. Check exact log-score recovery with the declared support, base measure, and temperature.
  - Compare full legal distributions using total variation and source-to-recovered KL, reporting infinite KL when source support is missed. Report conditional probabilities and calibration by board. Independent draws can disagree even when distributions match exactly.

- [ ] **66. Compare behavior and gameplay under exact and learned return values.**
  - Compare the raw behavioral decoder with greedy expected game utility on identical boards and matchups. Score exact returns, rollout estimates before fitting, learned successor `V`, and learned direct `Q` on all legal alternatives. Sweep rollout budgets on on-policy and off-policy positions. Include exact best-response game scores against known individual opponents as strength references and record the opponent information they require.
  - Give per-source counts for each Task 61 disagreement category and the average payoff forgone. Measure how often estimated scores restore or lose the source's move. Publish original and recovered win probabilities, paired game-score differences, move agreement or total variation distance, and return errors. Report values and differences for both equal board weights and weights proportional to the source's visits.

- [ ] **67. Measure how residual training error affects fidelity and playing strength.**
  - The 2026-11-20 tic-tac-toe investigation compared losses, inputs and tie handling on two board partitions. With the frozen neural player that chooses the earliest cell preserving its best guaranteed outcome, a network using board and move geometry reached 96.45% direct move agreement on 780 withheld boards, up from 83.72%. Extending three fits to 120 passes recovered all 2,980 training moves in each and averaged 96.92% on those withheld boards. Carry forward the separate measurement of same-board fit, withheld-board prediction, moves corrected or lost by VNM expected-utility selection, and game outcomes. Repeat on independently trained sources with other tie conventions. [Training losses, example count predictions and results](TICTACTOE_CHOICE_TRAINING.md)
  - At fixed training data and targets, compare checkpoints and model capacities across independent seeds. In separate paired sweeps vary data coverage or rollout budget while holding evaluation boards and opponents fixed. Plot fidelity and game score against training residual and held-out exact-target error. Break down move disagreements by the exact best-to-runner-up action gap.
  - Include a zero-estimation-error reference. Perturb exact scores at matched error magnitudes with a common offset or a change that crosses an action margin. This isolates the effect of ranking errors from correlations along training. Track probability shifts under soft choice at fixed temperature.
  - Report the size and direction of the change in move agreement or probability distance for each reduction in training error. Include variation in estimated returns, error on unseen boards, and variation across fits in rankings, values on the same scale and score patterns across nearby boards.

- [ ] **68. Establish attainable fidelity and explain the gaps for each formulation.**
  - For every score family and decoder in the game comparison, separate best validation-selected performance on held-out data, fidelity with its exact scores, and any certified optimum within that family. Compute a finite optimum or give justified bounds where possible. Label unresolved gaps and record when independent recoveries converge to different aligned solutions.
  - Include exact encoding controls and constructive limits. For a stochastic source `P`, the best deterministic action has total variation error `1 - max_a P(a|s)` on that board. A mixture restricted to expected-utility maximizers must miss at least the source mass outside that set. A fixed tie or temperature rule can impose further error.
  - Measure the reduction and remaining gap as training data, simulated games, network size, or training time increases. Derive bounds from indistinguishable inputs, shared resulting boards, restricted utility features, unobserved moves, conflicting rankings, or restrictions on move probabilities. Quantify the change in fit when each assumption changes and record what extra information the method uses.

- [ ] **69. Produce exhaustive value-atlas visualizations.**
  - Show action values, runner-up gaps, nearby reachable-state values, uncertainty, and disagreement clusters on the exact state graph.
  - **Done when:** examples include both clear and ambiguous “what P likes” cases.

- [ ] **70. Test value-derived stochastic completions.**
  - Cross deterministic and stochastic sources with greedy and temperature-based decoding of raw scores, fixed-payoff returns, and fitted expected utilities. Sweep temperature with score scale and base measure recorded. Include the low-temperature limit, ties, support masks, and a validation-selected temperature.
  - For a known stochastic source, compare each score with the score predicted by its move probability. In `S(s,a) = T log(P(a|s)/μ(a|s)) + c(s)`, `S` is the score, `P` the source probability, `μ` the reference move weight, `T` a shared positive temperature governing the spread of choices, and `c` an offset common to that board's moves. On moves with positive probabilities and weights report score residuals and the resulting probability differences. Report probability mass assigned or omitted where the sets of possible moves differ. Compare a network that predicts expected utility directly with one that predicts outcome probabilities and averages their utilities.
  - Compare these distributions with classical expected-utility choice under a specified tie mixture and one declared random-utility model that samples a utility then maximizes its expectation. Report distributional fidelity, win probability and game score for each, with uncertainty across independent recoveries.

- [ ] **71. Measure how closely a compact utility predicts held-out choices over moves and lotteries.**
  - Compare direct scores, the supplied payoff, and the shared utility fitted in Task 53 at matched access and budgets. Measure the change in held-out fidelity and game performance from requiring scores to factor through expected utility. Report utility complexity and stability up to the allowed normalization.
  - Before querying the source, predict its response to action removal, controlled policy edits, and changed consequence lotteries, such as a sure draw versus a specified win/loss gamble. Define how the source can observe and choose between each offered alternative. Use explicit lottery-choice fixtures when an ordinary board policy lacks that interface. Report which comparisons the original source actually supports.
  - Compare predictions under changed conditions with direct scores and simple decision rules. Include policies generated from a known utility and deliberately incompatible choices. Measure errors in known utility trade-offs after aligning scale and offset. Record equivalent explanations and the change in prediction error when applying each fitted utility in a new setting.

- [ ] **72. Publish the finite-domain result report.**
  - Publish a machine-readable table by source, score, decoder, access, and budget. Link the opponent/seat results, error and action-gap curves, fidelity bounds, counterfactual predictions, representative atlases, and negative results to the claim ledger.
  - For each of the three goals in Task 35, give all methods' measured values, differences on the same cases, and applicable uncertainty. Name the formulation and move rule favored by each objective with the size of its advantage or disadvantage. Include the source, ties, uncertainty in rankings, and trade-offs between goals.

## Checkpoint E — First research adjudication

- [ ] Answer the original README question for each policy class and state weighting. Record which limits are structural, which shrink with training, and which remain open. Use the three objective-specific results from Task 72 to select methods for Connect Four and 4×4 Reversi, including a direct behavioral control and an expected-utility comparison.

---

# Phase V — Interpretive leverage in larger neural game policies

- [ ] **73. Establish PyTorch Connect Four policy/value baselines.**
  - Freeze source networks of different strengths and include both greedy and stochastic sources. Adapt the Task 60 opponent protocol to Connect Four, including uniform random legal play and independently seeded untrained neural opponents. Record each source's absolute W/D/L by seat with uncertainty before evaluating its reconstructions.

- [ ] **74. Recover multiple value forms from each neural policy.**
  - Repeat the selected direct-score, revealed-value, and supplied or fitted expected-utility comparisons on common positions and against the Task 73 panel. Include achievable fidelity by access and compute budget, residual-error curves, and whole-game outcomes. Identify which tic-tac-toe exact bounds transfer and where only sampled estimates are available.

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
  - Freeze predictions of runner-up choices, responses to action removal or policy edits, and trajectory trade-offs before observing the source responses. Compare direct probabilities/logits, revealed rankings, and compact expected utilities at matched query budgets. Score held-out predictive accuracy, cross-seed stability, and explanation complexity.
  - Compare prediction errors and their differences for every formulation under each intervention, then name the most predictive formulation given those results, including ties and uncertainty. Use separate blinded human evaluation if claiming improved human understanding. Use the later intervention studies for claims about internal mechanism.

- [ ] **79. Build interactive/static neural-policy value atlases.**
  - Provide state views, action profiles, gaps, uncertainty, counterfactual masks, and trajectory summaries.
  - **Done when:** reports remain usable without inspecting raw tensors.

- [ ] **80. Test value-derived policy sampling in Connect Four.**
  - Repeat Task 70's stochastic comparisons for direct scores and expected utilities. Compare diversity, distributional fidelity, win probability, expected game score, and cross-recovery stability over temperature on held-out sources and opponents.

- [ ] **81. Compare value explanations with direct surrogate explanations.**
  - Include behavior cloning, policy distillation/logits, and simple local surrogate baselines.
  - **Done when:** any claimed interpretive advantage is incremental rather than merely a consequence of fitting another network.

- [ ] **81a. Measure how well the same methods recover a frozen player's choices in 4×4 Reversi.**
  - Use the Othello rules on a four-by-four board. Start with White on the northwest and southeast squares of the central four, Black on the other two, and Black to move. A placement must bracket at least one uninterrupted line of opposing discs between the new disc and a friendly disc. Flip all discs bracketed horizontally, vertically or diagonally. A player with no placement must pass if the opponent can play. When neither can play, the player with more discs wins, and equal counts draw. Record the side to move so a pass is an observable transition. [World Othello Federation rules adapted for this board size](https://www.worldothello.org/about/about-othello/othello-rules/official-rules/english)
  - Enumerate positions reachable from that opening and compute the best outcome each side can guarantee against perfect replies. Train a neural player to choose the first placement in row order, left to right, that preserves that outcome. Freeze it and record its actual choices and any guaranteed outcome given up. Sample complete games from the opening and from varied reachable positions. Retain move counts and visits per position. Train both the network scoring the board after a move and the network scoring a board and candidate move on the fraction of visits taking each move.
  - Compare choosing the highest immediate score with **von Neumann–Morgenstern (VNM) selection**, which chooses the move with the highest average learned utility of a completed game. Use the same fitted scores for both rules. Carry forward the methods selected in Task 72 and specify how scores along a game determine its utility, how each side continues play, how later moves are weighted, how ties are resolved and how many continuations are sampled per move. Report exact-move agreement separately for positions with multiple legal placements, one placement, and a forced pass, with a count and denominator for each. Alongside agreement, report win, draw and loss rates for the original and reconstructed players against the same opponents as Black and as White. On positions with multiple moves, also measure how often each method predicts the frozen player's next choice when its favourite is removed. Report uncertainty for sampled quantities and variation across independent fits.

- [ ] **82. Write the neural-game empirical paper/report draft.**
  - For tic-tac-toe, Connect Four and 4×4 Reversi, report how often each method predicts the original player's choices, how its win, draw and loss rates change, and how accurately it predicts responses to removing a move. Give within-game differences between the Task 72 methods, their denominators and the uncertainty on sampled results. Relate the differences to the original player's move rule, the opponent, and each method's access to the original player. Include how the learned scores vary across training runs and network designs. Link every figure and claim to its reproducible artifact.

## Checkpoint F — Minimum publishable program

- [ ] Audit the three-game comparison in Task 82, including the Reversi measurements from Task 81a. Review Tasks 73–82 and decide whether to release a first paper/software benchmark before broader adapters. Record the scope and uncertainty of each empirical claim.

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
  - Apply the three selection criteria using the domain's actual return or forecasting loss for task performance. State which counterfactuals can be checked for each source.
  - **Done when:** PV-H6 has a capability matrix rather than a universal claim by assertion.

## Checkpoint G — Non-sequence portability

- [ ] Report coverage across discrete games, control, and forecasting. Give the tasks and measurements supported in each domain, errors under the shared assumptions, and the changes required for uncovered cases. Use those results to decide any changes to the interface.

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
  - Report distributional fidelity, downstream task score, and held-out interpretive predictions separately. Identify where the game-based choice of formulation transfers.
  - **Done when:** the report gives the measured error, cost and coverage in each policy class and states the resulting scope of PV-H6.

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
  - **Done when:** predicted direction and size are recorded before intervention and observed changes are compared numerically with matched controls.

- [ ] **104. Test backpropagated marginal-value semantics.**
  - Compare `-dL/dh` or objective gradients with recovered local value differences and policy sensitivity.
  - **Done when:** the result distinguishes training-objective marginal value from deployment return or latent preference.

- [ ] **105. Search for interpretable value directions or concepts.**
  - Relate features to stable action/trajectory tradeoffs, not merely labels.
  - **Done when:** the report gives prediction errors for choices under changed conditions, with paired differences from the comparison methods on the same unseen examples.

- [ ] **106. Test changed-decoder and changed-objective transfer.**
  - Compare errors in predicting choices from retained values and from direct policy labels when the rule for taking an action, sampling temperature, constraints, costs, or available moves change. Give the magnitude and direction of the difference on the same cases.
  - **Done when:** the experiment directly connects this repo to the value-first thesis.

- [ ] **107. Compare value-based interpretability against standard baselines.**
  - Include logits/probabilities, saliency, local linear surrogates, behavior cloning, and mechanistic probes not trained on value.
  - **Done when:** the report gives the size, direction and uncertainty of the differences between methods on the same interpretability tasks.

- [ ] **108. Run one preregistered mechanistic confirmation experiment.**
  - Select the most promising policy class and hypothesis only after pilot work, then freeze the final protocol.
  - **Done when:** the confirmation reports measured effects, uncertainty and tested conditions for PV-H7.

- [ ] **109. Write the neural value-semantics research report.**
  - Separate behavioral representation, external interpretability, and internal mechanism throughout.
  - **Done when:** every proposed connection to an internal mechanism links to its measured effects, uncertainty and conditions, including estimates near zero.

## Checkpoint I — Interpretability adjudication

- [ ] Quantify how accurately recovered values predict behavior under changed conditions and how much altering the proposed internal mechanism changes decisions. Report both effects, their uncertainty and the tested conditions before updating the project title or subtitle.

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
  - Lead with the original-policy baseline and the three objective-specific choices. Explain attainable fidelity, residual training effects, and fixed versus inferred utility using the confirmed game results. Trace extensions to stochastic choice, other policy classes, and supported mechanistic findings.

- [ ] **115. Conduct an adversarial proof, statistics, and software audit.**
  - Re-run from clean environments, inspect leakage and seed lineage, check theorem/implementation agreement, and remove unsupported language.

- [ ] **116. Publish a versioned release and archival artifacts.**
  - Tag code, freeze reports/configs, archive machine-readable results, and record exact hardware/software environments.

- [ ] **117. Select the next research branch from evidence.**
  - Candidate branches include value-logic integration, active preference interrogation, LLM/agent value atlases, mechanistic value circuits, multi-agent value reconstruction, and value-based policy editing.
  - **Done when:** the selected branch follows from completed evidence rather than the original roadmap alone.
