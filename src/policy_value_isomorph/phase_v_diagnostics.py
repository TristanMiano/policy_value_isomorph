"""Paired fit stability, coverage, budgeted questions and changed continuations."""

import hashlib
import numpy as np
from .phase_v import actions, read, save, seed, spec, split_of
from .phase_v_study import decoded, probabilities, scoring, setup_fit, wmean
from .phase_v_evaluate import masked_scores, source_reply


def diagnostics(e, game, b, k, mix, f, folder):
    rng = np.random.default_rng(seed(game, b, 79, k, f))
    with np.load(folder / f"data_{k}_{mix}.npz") as d:
        states = list(map(tuple, d["states"].tolist()))
        ids = d["test"]
        test = [states[i] for i in ids]
        counts = d["counts"]
        w = d["equal"]
        vis = d["reach"]
    seen = {s: int(n) for s, n in zip(states, counts.sum(1)) if n}
    raw = scoring(e, test, spec(0))
    legal = raw > -1e28
    primary = {i: scoring(e, test, spec(i)) for i in (1, 2, 7, 8)}
    setup_fit(e, folder, k, mix, 1 - f)
    other = {i: scoring(e, test, spec(i)) for i in (1, 2)}
    setup_fit(e, folder, k, mix, f)
    logs = []
    examples = []
    answers = []
    curves = []
    for i in (1, 2):
        first = primary[i]
        for label, second in (("new_initialization", other[i]), ("other_architecture", primary[i + 6])):
            pairs = (
                legal[:, :, None]
                & legal[:, None, :]
                & np.triu(np.ones((raw.shape[1], raw.shape[1]), bool), 1)
            )
            rank = np.sign(first[:, :, None] - first[:, None, :]) != np.sign(
                second[:, :, None] - second[:, None, :]
            )
            legal_n = pairs.sum(2).sum(1)
            parents = [
                j for j, s in enumerate(test) if w[j] > 0 and e.legal(e.next(s, int(first[j].argmax())))
            ]
            children = [e.next(test[j], int(first[j].argmax())) for j in parents]
            if label == "new_initialization":
                setup_fit(e, folder, k, mix, 1 - f)
                y = scoring(e, children, spec(i)).max(1)
                setup_fit(e, folder, k, mix, f)
            else:
                y = scoring(e, children, spec(i + 6)).max(1)
            x = scoring(e, children, spec(i)).max(1)
            geom = (x - first[parents].max(1)) - (y - second[parents].max(1))
            logs.append(
                dict(
                    experiment="stability",
                    model=i,
                    comparison=label,
                    n=int((w > 0).sum()),
                    top_change=wmean(first.argmax(1) != second.argmax(1), w),
                    score_rms=float(
                        np.sqrt(wmean(np.where(legal, (first - second) ** 2, 0).sum(1) / legal.sum(1), w))
                    ),
                    order_reversal=wmean((rank & pairs).sum((1, 2)) / legal_n.clip(1), w * (legal_n > 0)),
                    neighbor_rms=float(np.sqrt(wmean(geom**2, w[parents]))) if parents else None,
                )
            )
        for j in np.flatnonzero((w > 0) & (counts[ids].sum(1) > 0) & (first.argmax(1) != raw.argmax(1)))[
            :2
        ]:
            row = int(ids[j])
            n = int(counts[row].sum())
            a = np.flatnonzero(legal[j])
            examples.append(
                dict(
                    model=i,
                    state=test[j],
                    withheld=True,
                    visits=n,
                    moves=a.tolist(),
                    observed=counts[row, a].tolist(),
                    predicted=(first[j, a] * n).tolist(),
                )
            )
    # Sample reserve positions with a positive visit count for the raw-count examples if needed.
    for i in (1, 2):
        if any(r["model"] == i for r in examples):
            continue
        withheld = [
            j
            for j, s in enumerate(states)
            if counts[j].sum() > 0 and split_of(e, s, seed(game, b, 11, k)) == 2
        ]
        if withheld:
            choices = [states[j] for j in withheld]
            z = scoring(e, choices, spec(i))
            truth = scoring(e, choices, spec(0))
            for h in np.flatnonzero(z.argmax(1) != truth.argmax(1))[:2]:
                j = withheld[h]
                a = actions(e, states[j])
                n = int(counts[j].sum())
                examples.append(
                    dict(
                        model=i,
                        state=states[j],
                        withheld=True,
                        visits=n,
                        moves=a,
                        observed=counts[j, a].tolist(),
                        predicted=(n * z[h, a]).tolist(),
                    )
                )
    # Neighbor challenging the current choice: largest discrepancy between two learned Q score profiles.
    challenge = []
    salt = seed(game, b, 11, k)
    for j in np.flatnonzero(w > 0):
        options = [tuple(e.next(test[j], int(a))) for a in np.flatnonzero(legal[j])]
        candidates = [s for s in options if e.legal(s) and split_of(e, s, salt) == 2]
        if candidates:
            a = scoring(e, candidates, spec(2))
            c = scoring(e, candidates, spec(8))
            dist = np.where(a > -1e28, np.abs(a - c), 0).max(1)
            challenge.append(candidates[dist.argmax()])
    visited = {}
    for model, name in ((0, "original_play"), (2, "reconstruction_play")):
        paths = [
            e.trajectory(e.start(), spec(model), spec(model), int(rng.integers(2**32))) for _ in range(32)
        ]
        visited[name] = [tuple(s) for path in paths for s in path[:-1] if split_of(e, s, salt) == 2]
    for subset, pop in [
        ("all_held", [test[j] for j in np.flatnonzero(w > 0)]),
        ("unobserved", [s for j, s in enumerate(test) if w[j] > 0 and not seen.get(s)]),
        ("one_or_two", [s for j, s in enumerate(test) if w[j] > 0 and 0 < seen.get(s, 0) <= 2]),
        ("three_or_more", [s for j, s in enumerate(test) if w[j] > 0 and seen.get(s, 0) > 2]),
        ("challenge_neighbor", challenge),
        *visited.items(),
    ]:
        if pop:
            o = probabilities(e, pop, spec(0))
            p = probabilities(e, pop, spec(2))
            match = float((p * o).sum(1).mean())
            tv = float(np.abs(p - o).sum(1).mean() / 2)
        logs.append(
            dict(
                experiment="coverage",
                population=subset,
                n=len(pop),
                unique=len(set(pop)),
                seen=sum(s in seen for s in pop),
                training_collection_boards=len(seen),
                agreement=match if pop else None,
                tv=tv if pop else None,
            )
        )
    # Query and assess disjoint withheld decisions. A response adjusts the residual move probabilities
    # of a nearby board according to shared color occupancy and legal alternatives.
    multi = np.flatnonzero((w > 0) & (legal.sum(1) > 1))
    perm = rng.permutation(multi)
    pool = perm[: len(perm) // 2]
    held = perm[len(perm) // 2 :]
    avail = legal.copy()
    avail[np.arange(len(test)), raw.argmax(1)] = False
    base = decoded(masked_scores(primary[2], avail, False), 0.25)
    alt = decoded(masked_scores(other[2], avail, False), 0.25)
    strategies = dict(
        active=pool[np.argsort(-np.abs(base[pool] - alt[pool]).sum(1), kind="stable")],
        random=rng.permutation(pool),
        visitation=pool[np.argsort(-vis[pool], kind="stable")],
    )
    for label, order in strategies.items():
        for budget in (8, 24):
            chosen = order[:budget]
            labels = source_reply(e, [test[j] for j in chosen], avail[chosen], mix, False)
            predictions = base[held].copy()
            for j, h in enumerate(held):
                s = test[h]
                dist = np.array(
                    [
                        (
                            (s[0] ^ test[q][0]).bit_count()
                            + (s[1] ^ test[q][1]).bit_count()
                            + 2 * int(s[2] != test[q][2])
                        )
                        for q in chosen
                    ]
                )
                similarity = 1 / (1 + dist) ** 2
                residual = np.average(labels - base[chosen], axis=0, weights=similarity)
                pred = np.where(avail[h], np.maximum(0, base[h] + residual), 0)
                predictions[j] = pred / pred.sum() if pred.sum() else base[h]
            digest = hashlib.sha256(predictions.tobytes()).hexdigest()
            truth = source_reply(e, [test[h] for h in held], avail[held], mix, False)
            logs.append(
                dict(
                    experiment="queries",
                    strategy=label,
                    budget=budget,
                    asked=len(chosen),
                    n=len(held),
                    prior_tv=wmean(np.abs(base[held] - truth).sum(1) / 2, w[held]),
                    tv=wmean(np.abs(predictions - truth).sum(1) / 2, w[held]),
                    queried_tv=wmean(np.abs(base[chosen] - labels).sum(1) / 2, w[chosen]),
                    forecast=digest,
                )
            )
            answers.append(
                dict(
                    strategy=label,
                    budget=budget,
                    queries=chosen.tolist(),
                    labels=labels.tolist(),
                    held=held.tolist(),
                    prediction=predictions.round(7).tolist(),
                    actual=truth.tolist(),
                )
            )
    # Two offered moves induce outcome lotteries. Test the predicted preference for a higher payoff
    # against sampled outcomes after changing the opponent that finishes the hypothetical game.
    case_ids = rng.choice(multi, min(24, len(multi)), replace=False)
    saved = read(folder / f"metrics_{k}_{mix}_{f}.jsonl.gz")
    e.draw_value = next(r["draw_value"] for r in saved if r["method"] == "compact_VNM")
    forecasts = {
        name: scoring(e, [test[j] for j in case_ids], p)
        for name, p in (
            ("V_direct", spec(1)),
            ("Q_direct", spec(2)),
            ("imitation", spec(3)),
            ("distilled", spec(4)),
            ("revealed", spec(11)),
            ("cell", spec(5)),
            ("Q_discounted_VNM", spec(2, 3)),
            ("V_discounted_VNM", spec(1, 3)),
            ("Q_terminal_VNM", spec(2, 1)),
            ("V_terminal_VNM", spec(1, 1)),
            ("supplied_VNM", spec(2, 4)),
            ("payoff_network", spec(6)),
            ("outcome_network", spec(15)),
            ("compact_VNM", spec(2, 5)),
        )
    }
    offered = [rng.choice(np.flatnonzero(legal[j]), 2, replace=False) for j in case_ids]
    for ii, (j, options) in enumerate(zip(case_ids, offered)):
        h = test[j]
        sample = []
        for a in options:
            c = e.next(h, int(a))
            x, o = (spec(0), spec(-1)) if h[2] == 0 else (spec(-1), spec(0))
            sample.append(
                [e.match(c, x, o, seed(game, b, 85, k, f, ii, z))[0] * (1 - 2 * h[2]) for z in range(8)]
            )
        values = np.mean(sample, axis=1)
        for name, z in forecasts.items():
            pick = int(np.argmax(z[ii, options]))
            curves.append(
                dict(
                    method=name,
                    state=h,
                    moves=options.tolist(),
                    prediction=pick,
                    outcomes=sample,
                    forgone=float(values.max() - values[pick]),
                    tied=bool(values[0] == values[1]),
                )
            )
    save(folder / f"diagnostics_{k}_{mix}_{f}.jsonl.gz", logs)
    save(folder / f"examples_{k}_{mix}_{f}.jsonl.gz", examples)
    save(folder / f"queries_{k}_{mix}_{f}.jsonl.gz", answers)
    save(folder / f"lotteries_{k}_{mix}_{f}.jsonl.gz", curves)
