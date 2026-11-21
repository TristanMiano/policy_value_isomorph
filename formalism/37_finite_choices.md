# 37 · A score for every fixed choice

**In plain language.** If you know which move a player takes in each situation, you can give that move one point and every other available move zero. Reading off the highest score reproduces every recorded choice. The numbers have an exact inverse in this particular table format. Predicting a choice you have never seen is a further task. [Common notation](00_notation.md)

**Theorem.** A context is a history together with its offered menu. Let every decision context have a nonempty finite menu, a fixed priority order for resolving ties, and one legal source choice. Define the encoder (the rule producing scores) by assigning 1 to that choice and 0 to every other legal move. Define the decoder (the rule reading them) to take the first maximum in priority order. The encoder is a bijection, meaning a one-to-one correspondence in both directions, between deterministic policies on these contexts and tables having exactly one 1 and otherwise zeros in every legal row.

**Proof.** The chosen entry is the unique maximum, including for a one-entry row. Decoding after encoding therefore returns the source row at every context. Conversely, decoding such a table chooses its unique 1. Re-encoding restores that exact table, establishing both directions. For unrestricted real scores the full set of rows decoded to a given move has a different description. Its score must strictly exceed that of every earlier rival and weakly exceed that of every later rival. These inequalities are necessary by the tie rule and sufficient because the chosen move is then the first maximum. There are no rows to decode at terminal histories.

**Construction and limit.** With menu A, B, C in that priority order, scores (0,1,0) encode choosing B. Scores (−3,4,4) also choose B, while (4,4,−3) chooses A. A source may change its mind after a menu changes. Indexing by the offered menu accommodates this behavior. Using one menu-independent ordering across contexts introduces a further assumption.

```pycon
>>> from itertools import product
>>> menus = [("A",), ("A", "B"), ("C", "A", "B")]
>>> for choices in product(*menus):
...     rows = [[int(move == choice) for move in menu] for menu, choice in zip(menus, choices)]
...     assert tuple(menu[row.index(max(row))] for menu, row in zip(menus, rows)) == choices
>>> for row in product(range(3), repeat=3):
...     assert (row.index(max(row)) == 1) == (row[1] > row[0] and row[1] >= row[2])

```

**Self-review.** The construction settles existence even for unusual choices. The inverse recovers arbitrary scores only if the representation has been restricted to the stated image. Repeated actions on an already observed row and comparisons on new rows address different questions.
