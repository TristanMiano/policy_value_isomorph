# Notation and recovery contract

This is the notation reference for later proofs and experiments. A **policy** specifies behavior given its information and available actions. A **recovery** uses stated observations of a source policy to produce scores and a **decoder**, the rule that turns those scores back into behavior. Every artifact specifies the meaning of its scores and how they are used.

## Domains and observations

| Symbol | Meaning |
| --- | --- |
| `h ∈ H` | The history, including observations, previous actions and known chance outcomes available to the policy. A state `s` can replace it when the retained information is adequate for the specified prediction. |
| `A`, `C`, `C(h)` | The action space, a particular offered subset and the usual feasible set at a history. Decisions use nonempty sets. A terminal history ends a play and has no action distribution. Record whether a subset was offered to the source or generated only for evaluating a model. |
| `P(· given h,C)` | Source action distribution on the offered set. Write `P(h,C)` for its single chosen action in the deterministic case. Distinct menus can elicit distinct responses to the same history. |
| `M`, `κ`, `τ` | The environment's transition law, the continuation rules for all actors after a forced action, and one resulting complete play. For games, specify both seats, chance and the stopping horizon. `τ` denotes a trajectory, never temperature in this notation. |
| `μ`, `T` | The base measure (prior weights or a reference for density over possible moves) and positive temperature controlling the spread of sampled choices. Inputs and units for `T` must match those of the score. |
| `w(h)` | Nonnegative evaluation weight for each history. Normalize weights over the reported population and give the original counts or weight rule. Uniform over boards and proportional to visits answer different questions. |

For finite games, `N(h,a)` counts observed source transitions by move and `N(h)=Σa N(h,a)`. The visit label is `f(h,a)=N(h,a)/N(h)` when the denominator is positive. A missing board has an unobserved distribution. Two moves can lead to the same next state. Counts by successor identify their sum unless the move itself was recorded. The mover, legality and which actor generated each observation belong to the data record.

## Scores, utility and decoding

- Write `S(h,C,a)` for a general move score, with the menu omitted only when fixed or irrelevant. A board network outputs `V(s)`. A board-and-move network outputs `Q(h,a)`. State whether those outputs predict frequency, points, ranks or some other target. In deterministic games `s′=M(s,a)` denotes the successor board. A history extended by the action, `(h,a)`, retains information that may be lost in `s′`.
- A **utility** `U(τ)` assigns a number to a complete play. A **VNM** (von Neumann–Morgenstern) expected-utility score is `S_U(h,a)=E[U(τ) given h, forced a, M, κ]`. Here the expectation is over the specified continuation and environment after the move. State whether utility refers to the full history or the suffix beginning at the decision and keep that convention identical for competing actions. Example suffix utilities use the last learned transition score, or `Σj γʲ u_j` where `j=0` is the forced move, `u_j` is a signed learned transition score and `γ` is a time discount in [0,1]. A declared terminal game payoff gives another choice of utility.
- With terminal points and no intermediate reward or discount, fixed-policy board return is mean final payoff starting from a board. Move return under those conditions equals expected return of its successor under the transition law. Define the **root** as the player whose win gets +1, draw 0 and loss −1. For this zero-sum payoff, maximize on that player's turn and minimize on the other turn. When scores count how often the mover chose each move, both seats prefer larger frequencies. Declare the orientation before applying any decoder.
- `D(S,h,C)` denotes the deployed action law. Largest-score decoding chooses a maximizer with a specified rule for ties, such as the smallest legal cell. **Softmax** on finite actions gives probability `μ(a) exp(S(a)/T) / Σb μ(b) exp(S(b)/T)`, with positive denominator. For a density, replace the weighted sum by an integral with respect to the stated base measure. Zero mass is recorded explicitly. With the usual strictly positive base weights, a probability zero requires exclusion or score −∞. A stochastic completion is a sampling rule supplied for a previously deterministic ordering. State which probabilities were chosen by the researcher and which were observed from the source.

## Equivalence, normalization and recovery

A **query** requests an observable response, for example the full-menu move, a pairwise choice or a predicted payoff. For a family of queries `F`, write `S₁ ∼F S₂` when every query in `F` receives the same response from both scores under the fixed contract. The equivalence class `[S]F` collects those scores. Adding queries can distinguish members. Agreement of maxima on one menu places fewer restrictions than equality of all sampled probabilities or cardinal (numerically meaningful difference-scale) utilities.

A **normalization** chooses coordinates for comparison. Subtracting each board's maximum leaves softmax probabilities unchanged. For a nonconstant finite row, dividing by its range puts its max–min difference at one. This preserves order, and preserves softmax probabilities if temperature is divided by the same range. A constant row retains its tie flag and can be stored as zeros. Report any excluded infinities. A utility common to lotteries allows a shared positive affine change of units, `U′=aU+b` with `a>0`, which preserves expected-utility comparisons. Renormalizing each candidate's consequences separately can change their order.

Let `K` be the recovery contract and `r` the random seed, an integer specifying reproducible draws. Write `R(P,K,r) → ValueArtifact` for the procedure and its result. This signature denotes controlled access to the source through `K`. It also covers stored observations made before fitting. A **ValueArtifact** is the following research record. The field names specify meanings for subsequent work without prescribing a serialization format.

| Field | Required content |
| --- | --- |
| Domain and queries | Histories, candidate generator, ties, empty/terminal cases, covered population, query family and permitted access at collection, fitting and deployment. |
| Representation and decoder | Stored scores or fitted model, inputs, units, perspective, support, normalization, selection and sampling rules, temperature and base measure. |
| Return assumptions when used | Environment, starts, horizon, two continuation rules in games, utility of a whole play, discount, rollout count per forced move and source calls at selection. |
| Evidence and provenance | Source version, move rule, observations and denominators, data splits, model architecture, training loss and weighting, optimization settings, seeds, saved weights and code revision. |
| Evaluation and uncertainty | Populations and weights, imitation and full-game outcomes, queries under changed conditions, independent repeat results, uncertainty and limits of coverage. Link results and record unavailable fields with reasons. |

## Three instantiations

1. **Discrete game.** History is a board and side to move, candidates are numbered empty cells, the source chooses its highest legal score with lowest-cell ties. A fitted `Q(h,a)` predicts observed move frequency. Direct selection requires the artifacts and legal moves. VNM selection using source copies also calls the source to continue games after each candidate.
2. **Continuous action.** A controller chooses a steering angle on a closed interval. Negative absolute angular difference from the source angle gives a score with a unique best angle. This construction requires access to the source angle. For a fitted approximation, declare angle units, search resolution and error. For sampling, state whether density is per radian or per degree and transform the measure consistently.
3. **Sequence.** A prefix is the history, allowed next tokens are the candidates and the source supplies conditional probabilities. Log probabilities give one direct score at unit temperature with equal base weights. Truncating to generated candidates omits mass. A utility over complete sequences additionally declares how future tokens are drawn and how the sequence ends.

**Self-review.** Using one symbol for any numerical output would hide both score perspective and access to the source. The contract makes them explicit while permitting visit-based, payoff-based and learned whole-play explanations. Which representations transfer to unobserved menus and larger domains remains measurable. [Capability declarations](../docs/policy_value_taxonomy.md) · [Frozen package calls](../CHECKPOINT_A.md)
