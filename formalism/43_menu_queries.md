# 43 · Learning an ordering by changing the available choices

**In plain language.** If a player always takes its favorite available move from one fixed ordering, ask it to choose, remove its answer, and ask again. This reveals the entire order. Observing the ordinary first choice alone leaves the rest open. [Common notation](00_notation.md)

**Theorem.** At one fixed history let n ≥ 1 distinct moves be available. A query offers any nonempty subset to the same source and observes its choice. Assume the source always chooses the unique highest member of one strict total order (every pair is comparable and comparisons are transitive). Repeated elimination recovers the order in n−1 queries, with the last move inferred. Alternatively asking all unordered pairs takes n(n−1)/2 queries and determines that same order.

**Proof and algorithms.** Start with the whole menu. Record the chosen move as first and remove it. Inductively, if the first k have been removed, the next answer is the highest remaining move. The singleton left at the end fixes the last position. In the pair method draw an arrow from the chosen member to its rival. The observed comparisons orient every pair. Under the assumption, the move at position j, counting from 1, beats precisely n−j rivals. Sorting by that count in descending order reconstructs the full order. Apply the procedure separately at each history. A recorded score mask that changes what an external model selects is a model prediction. The query here measures the source's actual response.

**Checking the assumption.** A directed cycle of pairwise wins rules out a stable strict ordering. If every pair was queried and there is no cycle, sorting gives the unique compatible order: an acyclic orientation of every pair has a topological order (a list respecting all arrows), and each pair fixes its own relative position. Check every additional observed menu against its highest member. Agreement on all queried menus establishes compatibility there. Even an internally consistent pair table can miss a deviation on an unqueried triple. With partial observations, draw the chosen-above-rival arrows from all observed menus. Every acyclic finite constraint graph has an order extension, obtained by repeatedly removing a vertex with no incoming arrows. Uncompared pairs can remain undetermined. An underlying indifference with fixed tie priority can give the same total choice order as a strict preference.

**Example.** Player one always chooses the first available of B, A, C. Its pair winners on AB, AC, BC are B, A, B. Player two gives those same pair answers but takes C when all three are offered. Using player one's order predicts three out of three pair responses and misses the triple, hence matches 3/4 on those equally weighted queries. A third rule choosing A over B, B over C and C over A exhibits a pair cycle. Changing the representation to include the menu preserves such choices. Repeating queries under controlled menu changes measures how often the apparent ordering shifts.

```pycon
>>> from itertools import permutations, combinations
>>> moves = "ABC"
>>> def pick(order, offered): return next(a for a in order if a in offered)
>>> for order in permutations(moves):
...     remaining = list(moves)
...     recovered = []
...     while len(remaining) > 1:
...         chosen = pick(order, remaining)
...         recovered.append(chosen)
...         remaining.remove(chosen)
...     assert tuple(recovered + remaining) == order
...     wins = {a: sum(pick(order,(a,b)) == a for b in moves if b != a) for a in moves}
...     assert tuple(sorted(moves, key=wins.get, reverse=True)) == order
>>> menus = [*combinations(moves,2), tuple(moves)]
>>> sum(pick("BAC", m) == ("C" if len(m)==3 else pick("BAC",m)) for m in menus)
3

```

**Self-review.** A pairwise experiment can reconstruct a stable relation economically and can also miss a menu effect. Count what was queried, record reversals, and retain menu-dependent descriptions alongside the strict-order model when their predictions differ.
