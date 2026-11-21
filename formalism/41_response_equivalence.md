# 41 · Which score differences are observable?

**In plain language.** Two score tables are indistinguishable if every question you are allowed to ask of them has the same answer. There is an exact correspondence between groups of indistinguishable tables and the answer patterns you can actually obtain. Asking about move probabilities can separate tables that always pick the same best move. [Common notation](00_notation.md)

**Theorem.** Fix a score space X, the contexts, tie conventions, and a family F of queries. A query f returns a specified observable f(S) for scores S. For random choice use the probability law as the observable, or explicitly specify a coupling of individual draws. Write R(S) for the list of answers, one for each f. Define S ~ S′ when R(S) = R(S′). The set X/~ of groups of equivalent scores, called the **quotient**, is in bijection with the attainable response patterns R(X).

**Proof.** Equality is reflexive, symmetric and transitive, so the groups partition X. The map from a group [S] to R(S) is well defined by the grouping rule, injective because equal patterns imply equal groups, and onto R(X) by its definition. Hence R factors as grouping followed by this bijection. More generally, a report g(S) can be computed from the group alone exactly when g is constant within each group. Define its grouped value as g of any representative. Conversely any function on groups takes the same value at all representatives.

If policy behavior is the full response, every exact encoder of a given policy selects a representative in the same group, and all such encoders give the same map into the quotient. If additional queries are included, equality of policy behavior alone can span several groups. Adding a query can split a group and cannot merge formerly distinct ones. An inverse from responses to unrestricted raw scores requires each group to contain exactly one score table, or an explicit choice of one representative per group.

**Three consumers.** A first-maximum decoder reads the winning move on a menu. Strictly increasing transformations preserve this answer, and even reordering two losers can be invisible on that menu. For exponential sampling with fixed positive temperature and base weights, equality of the full probability vector requires the same support and a common additive offset on that support by the log-probability theorem. A threshold consumer returns 1 when a score reaches a fixed numerical cutoff, otherwise 0. Equality then means the same pattern on the two sides, including the convention at the boundary. A generic shift can cross the boundary. This is a mathematical readout of a single score, independent of how the project reports quantitative research measurements. [Probability representation](38_random_choices.md)

```pycon
>>> from itertools import product
>>> tables = list(product(range(3), repeat=2))
>>> def winner(row): return row.index(max(row))
>>> def threshold(row): return tuple(int(x >= 1) for x in row)
>>> len({winner(v) for v in tables}), len({(winner(v), threshold(v)) for v in tables})
(2, 5)
>>> # At unit temperature with equal base weights the difference fixes both probabilities.
>>> len({a-b for a,b in tables})
5
>>> winner((0,1)) == winner((1,2)), threshold((0,1)) == threshold((1,2))
(True, False)

```

**Self-review.** This proves an isomorphism of sets with specified domains: observable classes and attainable answers. It establishes uniqueness at the level of those observations. The project can pursue stronger identification by collecting additional responses under changed conditions.
