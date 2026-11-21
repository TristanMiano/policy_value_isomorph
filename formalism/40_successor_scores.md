# 40 · When a single number per resulting board is enough

**In plain language.** A score for a situation and a proposed move can always be stored on the pair. Storing only the resulting board throws away which situation and move produced it. Sometimes a single score per board still reproduces every choice, even when it cannot reproduce every number. The exact obstruction is a circular chain of required comparisons containing at least one strict improvement. [Common notation](00_notation.md)

**Lifting theorem.** Let x be a context and a a legal action. For any move score Q, the function on the augmented state (x,a) given by V((x,a)) = Q(x,a) preserves every score. More generally let z(x,a) be the object retained after a move. There is a scalar V with Q(x,a) = V(z(x,a)) exactly when Q is constant across all pairs sharing that same z. Necessity follows by evaluating V at one object. For sufficiency assign their common score to the object. A path prefix retaining the original move can satisfy this condition even if a board alone does not.

For game points accumulated over play, fix the rules, payoff perspective and both continuation players. Under deterministic transitions, terminal-only payoff and no discount, the expected payoff after forcing a move equals that of its **extended history**, which includes the move. A board alone carries this return precisely when all reachable histories mapped to it have the same conditional mean payoff under those rules. One way to ensure this is for future transition, continuation and payoff rules to depend only on the retained board. For example, two routes reach the same board. There the continuation player finishes left after the first route and right after the second, earning +1 and −1 respectively. One common board number has largest absolute error at least 1 point across the two histories, attained at zero. Retaining the route restores both returns. If values use the new player to move's viewpoint in an alternating zero-sum game, reverse the sign. For a finite random transition law with additive immediate reward, the successor formula averages the immediate reward plus discounted child value over possible children that retain the information needed by those rules. Shared child values reproduce a given move-score vector exactly when they solve all its resulting linear equations.

**Choice theorem for finitely many decision rows and successor objects.** Assume a known priority order for ties in each row. Orient all scores for maximization by the mover for the moment. For chosen action c and each rival a, draw an arrow from the rival successor to the chosen successor. Its constraint is **strict** when a precedes c in the tie order, and **weak** otherwise. The arrow requires the destination score to be larger, or at least as large, respectively. If a row minimizes a common fixed-player score, reverse its arrows and retain their strictness. Then one common scalar per object reproduces all choices if and only if there is no directed cycle containing a strict arrow. A strict self-loop is included.

**Proof.** The inequalities are necessary and sufficient within each row by first-maximum or first-minimum selection. Chaining a cycle with a strict arrow would force one value to be strictly larger than itself. For the converse group vertices that can reach one another in both directions. All arrows inside each group must be weak. Collapse each group to one vertex. The resulting finite graph is acyclic. Give each group the largest number of strict arrows on any path ending there, allowing a path of length zero. Appending an arrow raises the required count by one for a strict arrow and by zero for a weak one. The assigned integers therefore satisfy every constraint, including the weak equalities inside a group.

**Best attainable fraction.** For nonnegative row weights with positive total weight, maximize the weight of retained rows whose arrows admit the construction, then divide by total weight. Any scalar's correctly decoded rows form such a subset. Conversely the construction realizes all choices in any retained subset. This characterizes the optimum without assuming it can be found cheaply.

**Example.** Use three successor labels X, Y, Z. In each row the first listed action wins a tie. A source choosing the second action from (X,Y), (Y,Z), (Z,X) demands Y above X, Z above Y and X above Z. Every scalar misses at least one row. Giving X, Y, Z scores 0, 1, 2 matches two rows, hence the optimum is **2/3**, versus **3/3** with a score for each context and action. If the source chooses the first listed action in all three rows, all constraints are weak and a constant scalar matches 3/3. In the project's full-table tic-tac-toe example one board score did reproduce 4,520/4,520 choices even though 2,026/5,477 resulting boards had incompatible frequency labels from different parents. [Measured construction and population](../TICTACTOE_CHOICE_TRAINING.md) The minimum number of levels for that archived choice table is three with the specified lowest-cell tie rule and seven when strict superiority is required for every chosen move. [Recount and proof of minimality](../CHECKPOINT_C.md)

```pycon
>>> from itertools import product
>>> rows = [(0,1), (1,2), (2,0)]
>>> def hits(values, target_index):
...     return sum(max(row, key=values.__getitem__) == row[target_index] for row in rows)
>>> max(hits(v,1) for v in product(range(3), repeat=3)), hits((0,0,0),0)
(2, 3)
>>> sum(max(range(2), key=(0,1).__getitem__) == 1 for row in rows)
3
>>> from fractions import Fraction as F
>>> # A continuation remembers which route reached the board. Its two returns are +1 and -1.
>>> worst_error = lambda board_score: max(abs(board_score - r) for r in (1, -1))
>>> [worst_error(x) for x in (F(-1,2), 0, F(1,2))]
[Fraction(3, 2), 1, Fraction(3, 2)]

```

**Self-review.** Requiring all comparisons to be strict would reject feasible tied scores. Merging histories indiscriminately would erase feasible move-specific distinctions. Keeping the tie order gives the exact boundary, and augmenting the state repairs the conflicting example.
