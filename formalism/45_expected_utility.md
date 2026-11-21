# 45 · When following expected utility reproduces a player

**In plain language.** Once the rules for finishing a game and scoring its outcome are fixed, each move has an average utility. A player can be copied by maximizing that average exactly when its choices are among the best, with the right treatment of ties. For a random player, its probability of choosing worse moves is exactly the smallest possible probability discrepancy for *any* rule confined to the best moves. A utility learned from choices can yield a different answer from the game's supplied points. [Common notation](00_notation.md)

**Contract.** A history h is what the player knows at the decision. Force a legal first move a. The environment M supplies transitions and chance probabilities. The continuation κ specifies who plays both seats thereafter and their information. Play to the declared horizon, including how unfinished plays are scored. Let τ be a complete play and U(τ) a common numerical utility with finite expectation. The **von Neumann–Morgenstern (VNM) expected-utility score** is `Q(h,a) = E[U(τ) given h, forced a, M, κ]`. Use a fixed perspective: in zero-sum games maximize a fixed player's points on its turn and minimize them on the other turn. Below, orient Q for maximization by the mover. Fix a population weight w(h) before averaging across decisions. The sufficiency results concern selecting a move under this contract, independent of how U was obtained.

**Rollout special case.** If U gives +1, 0, −1 to a fixed player's win, draw and loss, the expectation is win probability minus loss probability. Averaging n independent complete simulations after the forced move estimates this quantity without bias, with variance equal to per-play variance divided by n. Under deterministic transitions and continuation there is just one resulting payoff. The project's alternative learned utilities assign a score to the last transition or sum signed fitted move scores along the path with time decay. Both require the same declared utility on all candidates' continuations. When deploying a reconstructed player for a full game, it must choose on every turn in its seat. The one-forced-move expectation uses κ afterward.

**Theorem for fixed utility.** Let G(h) be the nonempty set of highest-scoring moves on a finite menu, and let d(h) be its first member in the tie priority. A deterministic source is copied on exactly the weight of histories where its move equals d(h). For source law p and a fixed deterministic selection d, the total variation distance (half the sum of absolute probability differences) is 1−p(d). For arbitrary mixing among the maximizers at that history, the minimum distance is
`m(h) = 1 − Σ[a in G(h)] p(a)`.
No such decoder can put mass outside G. Hence the difference in probability assigned to that event is m, a lower bound on total variation. For equality, keep p's existing mass on G and add all outside mass to any member of G. The absolute mass differences inside and outside are both m. This also covers m=1. A tie rule shared across histories or fixed independently can do worse. For a specified tie distribution q on G the distance is exactly `m + Σ[a in G] max(p(a)−q(a),0)`, by summing the positive deficits. Weight these per-history quantities by w for the population result. The achievable optimum allows a freely chosen tie distribution at each history and is an information-rich reference.

**Theorem for a utility inferred from choices.** In a finite problem fix the allowed consequences z and the probability L(h,a,z) of each consequence under M and κ. Seek one real number u(z) per consequence, or a specified smaller family of such vectors. Its move scores are `Σz L(h,a,z)u(z)`. For deterministic observations, exact recovery under first-maximum decoding is equivalent to the following simultaneous feasibility problem. For each observed choice c and rival b, require `Σz (L(h,c,z)−L(h,b,z))u(z) ≥ 0`, strictly positive when b wins a tie against c. Reverse signs for minimizing rows. Necessity follows by subtracting the rival's expected utility. Conversely every vector satisfying all inequalities makes c the selected maximizer at each observed row. This proves existence conditional on the allowed inputs, without implying unique trade-offs or predictive accuracy on new rows. Fix meaningful units and test a compact family on held-out consequences or fresh choices. Allowing the source action or identity of each training row into the utility can encode labels directly.

**Three finite constructions.** Consider a one-step game, so both continuations end immediately. Actions W and D give a win and draw for the mover. The player always choosing W is copied by maximizing supplied points, while a player always choosing D is copied on 0/1 decisions and gives up 1 point here. A learned utility assigning the draw higher than the win can reproduce D if its admissible units allow this ordering. A model constrained to prefer wins cannot. Now let actions L and R lead to identical lotteries over the allowed consequences, with L first in the tie order. Every shared consequence utility gives a tie. A player choosing R is copied on 0/1 decisions. If the outcomes also have distinct durations observable to the player, allowing duration in the consequence features can distinguish them. If the full observable lotteries are identical, alternative evidence would have to concern the move itself or the tie rule. In longer games the same argument applies to the complete continuation distributions.

**Three sources of random choice.** VNM maximization of a fixed expected utility can mix over tied best moves according to a declared rule. A **random-utility model** instead draws a utility before the decision, then maximizes its expectation. For example, draw a type favoring W with chance 1/4 and a type favoring D otherwise, with each type giving its favorite 1 and its rival 0. This produces (1/4,3/4) even though maximizing the average of those two utilities always chooses D. Draws of a utility depending on source labels have less explanatory content than a shared type model tested across contexts.

A third rule, **VNM scores + temperature sampling**, uses base probabilities μ and temperature T>0 to choose a distribution q. Put `KL(q||μ)=Σa q(a) log(q(a)/μ(a))`, with zero-q terms zero and positive mass outside μ's support forbidden. This measures distributional discrepancy in natural-log units. Let Z=Σa μ(a)exp(Q(a)/T) and q*(a)=μ(a)exp(Q(a)/T)/Z. For finite Q on the base support, substitution gives
`Σa q(a)Q(a) − T KL(q||μ) = T log Z − T KL(q||q*)`.
The last discrepancy is nonnegative, with equality only at q=q*: use −log x ≥ 1−x on positive q and sum, accounting for any q* mass outside q. Thus q* uniquely maximizes this objective. At positive temperature it can choose moves below the best expected utility. Its trade-off depends on T and μ. The three rules imply distinct probabilities until their parameters and tie conventions are specified.

```pycon
>>> from fractions import Fraction as F
>>> from math import exp, log, isclose
>>> p = (F(1,4), F(1,2), F(1,4))  # First two moves share highest utility.
>>> tv = lambda p,q: sum(abs(a-b) for a,b in zip(p,q))/2
>>> tv(p,(F(1,2),F(1,2),0)), tv(p,(1,0,0))
(Fraction(1, 4), Fraction(3, 4))
>>> points = (1,0)
>>> [int(points.index(max(points)) == source) for source in (0,1)]
[1, 0]
>>> lotteries = [(F(1,2),F(1,2))]*2
>>> assert all(sum(p*u for p,u in zip(lotteries[0],v)) == sum(p*u for p,u in zip(lotteries[1],v)) for v in [(0,1),(2,-3)])
>>> weights = (F(1,4),F(3,4))
>>> assert weights[1] > weights[0] and sum(weights) == 1
>>> q = (.25,.75); mu = (.5,.5); Z = (exp(1)+1)/2
>>> optimum = (.5*exp(1)/Z, .5/Z)
>>> kl = lambda a,b: sum(x*log(x/y) for x,y in zip(a,b) if x)
>>> assert isclose(q[0]-kl(q,mu), log(Z)-kl(q,optimum))

```

**Self-review.** A fixed game payoff and a freely tailored utility give opposite extremes for imitation. Holding the continuation fixed and restricting which consequences a common utility sees defines the middle question the project can test. Matching today's choices under this restriction still leaves its transfer to unseen lotteries open.
