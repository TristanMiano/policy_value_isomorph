# What can be observed, and what a recovered value means

A **source policy** gives a choice or a probability distribution over the moves offered at a history (the information available before choosing). An **adapter** is the interface through which an experiment queries this source or its environment. A **value artifact** bundles a score with its meaning, selection rule, evidence and domain of use. Write down access separately during data collection, fitting, choosing evaluation settings and deployed play. A simulator, a source and an internal network expose different information. [Research stance](../RESEARCH_STANCE.md) · [Claim register](../CLAIM_LEDGER.md)

## Access an adapter can declare

| Access | What the caller receives | What can be inferred, and which ambiguity remains |
| --- | --- | --- |
| Deterministic action | One legal move for a position and its offered moves. Repeated identical queries have the same answer. | Records the chosen move. With a largest-score rule it constrains comparisons against that move, subject to ties. Lower choices can have many orders. |
| Samples of a random policy | Independent draws conditional on the same history and offered moves, or a sequence with its dependence recorded. | Counts estimate probabilities at visited positions. Record the number of draws and zero counts. A move never drawn can still have positive probability. |
| Probabilities or raw move scores | A probability for each legal move, or the numbers before the source converts them into probabilities (logits). | Probabilities determine log-probability differences on positive support at known temperature. Raw outputs also expose an implementation's particular scale. Record masks, units and the actual sampling rule. |
| Choices under constraints | A fresh source choice after changing the set of moves on offer, while keeping the original history fixed. | Pair queries reveal an order if choices use one stable ordering. If the second-best move changes with the menu, record that context dependence. Deleting the top logit after the fact models one response and supplies no fresh source observation. |
| Rollouts | Complete plays after forcing an initial move, with a declared continuation rule for each player. | Estimate expected utility under that rule. Record starts, random draws, horizon, payoff or learned path utility, and whether source calls are required during selection. Unforced source games alone may never visit alternative moves. |
| Environment | Legal moves, transitions, chance probabilities or a simulator, and any supplied payoff. | Supports counterfactual plays and exact enumeration in small games. Knowledge of game rules gives no direct observation of which terminal play the source prefers. Record whether exact transition probabilities or only draws are available. |
| Internal network access | Weights, hidden activity or derivatives. A separate capability permits intervening on a hidden quantity and rerunning the source. | Supports predictions from internal activity and, with controlled interventions, tests of causal involvement. A read-only score predictor measures association. |

Capabilities may coexist. A missing query is **unavailable**. A query the adapter supports but the experiment did not issue is **unused**. A renormalized distribution or simulated outcome produced by the experiment is **derived**. These distinctions let a later comparison charge for information actually used.

## Meanings the evidence can support

| Artifact meaning | What is represented | Allowed comparison and required conditions |
| --- | --- | --- |
| Decoder score | A number for each candidate with a rule converting numbers to choices. | Compare decoded behavior on declared histories. Scoring the chosen move 1 and the others 0 reproduces an observed deterministic choice with limited information about alternatives. |
| Revealed preference | Relative priority inferred from choices after changing menus or lotteries (distributions of outcomes). | Predict new choices under comparable changes. Order is observable under stable comparisons. Relative utility differences require assumptions and observations about trade-offs under uncertainty. |
| Expected return or expected utility | Mean cumulative payoff, or mean learned utility of the complete play, conditional on an initial move and continuation. | Specify the environment, perspective, time horizon, later moves, units and any time discount. Changing any of these changes the question. |
| Mechanistic quantity | An internal computation proposed to influence the source's choice. | Compare choice changes after a targeted intervention with matched controls. Keep predictive accuracy, causal effect and an outside observer's ability to use an explanation separate. |

**Direct selection** reads fitted move scores immediately. **Von Neumann–Morgenstern (VNM) expected-utility selection** scores each candidate by the mean of one declared complete-play utility over its possible continuations. Either set of move scores can be maximized with a tie rule or turned into probabilities by exponential sampling at a specified temperature. Record both the score's origin and how it produces the deployed move. A board network scores a resulting board. A board-and-move network scores the candidate in its original context. The input shape alone cannot determine any of the four meanings.

## Example declarations

| Setting | Data and fitting | Selection and scope |
| --- | --- | --- |
| Frozen tic-tac-toe player choosing its highest legal score with lowest-cell ties | Sample its turns in complete games and normalize counts within each parent board. The environment provides all legal successors. Withhold complete board groups from fitting. | A direct network needs legal moves and its own weights. A VNM version rolling out two copies of the frozen source also needs the source and simulator at selection. Rolling out two fitted copies uses the simulator and fitted weights instead. Compare both under explicit budgets. |
| Continuous controller choosing a steering angle | Observe its angle at sampled sensor histories. Record the angle range, units, distance and sampling coverage. | Negative distance from the source angle is an exact score if one can query that angle. Independent deployment needs a fitted predictor and a rule for searching allowed angles. A probability density also needs a reference measure, such as length in radians. |
| Sequence model assigning probabilities to the next token | Query a prefix (tokens seen so far), permitted next tokens and their probabilities. Record the probability mass omitted by a candidate generator. | Direct sampling chooses from the declared token set. Scoring completed sequences additionally needs a completion rule, stopping length and utility. A prefix table alone leaves utility over completions open. |

For a new adapter, record the domain, candidate generator and omitted alternatives, access in each phase, supported queries and reset/independence assumptions. Then give the score meaning, units, decoder, independent or source-assisted deployment, unsupported comparisons and evidence location. This is a declaration for later implementations.

**Self-review.** A source answer on a restricted menu and a program deleting an action from saved scores can yield the same number with different evidential meaning. The declaration distinguishes the two and preserves a constructive comparison when fresh access is available. Prediction after an actual change of outcome probabilities remains an open measurement.
