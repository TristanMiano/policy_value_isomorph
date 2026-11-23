"""Evaluate every deployment on matched decisions, actual games and fresh restricted menus."""

import hashlib
import numpy as np

from .phase_v import opening_panel, panel, save, seed, spec
from .phase_v_study import decoded, probabilities, scoring, wmean

TEMPS = (0.0, 0.05, 0.25, 1.0)


def temperature_label(t):
    return "max" if t < 0 else f"T{t:g}"


def methods():
    for model, name in ((1, "V"), (2, "Q")):
        yield name + "_direct", spec(model)
        for utility, label in ((1, "terminal"), (2, "sum"), (3, "decay")):
            for who, continuation in enumerate(("copies", "source")):
                yield f"{name}_VNM_{label}_{continuation}", spec(model, utility, who)
    for name, p in (
        ("source", spec(0)),
        ("imitator", spec(3)),
        ("distilled_logits", spec(4)),
        ("cell_frequency", spec(5)),
        ("revealed_order", spec(11)),
        ("payoff_network", spec(6)),
        ("outcome_network", spec(15)),
        ("supplied_VNM", spec(2, 4)),
        ("compact_VNM", spec(2, 5)),
        ("supplied_VNM_source", spec(2, 4, 1)),
        ("compact_VNM_source", spec(2, 5, 1)),
        ("random_utility_VNM", spec(2, 6)),
        ("masked_VNM", spec(2, 7, 0, 0.25)),
    ):
        yield name, p


def masked_scores(z, mask, edit):
    result = np.where(mask, z, -1e30).copy()
    if edit:
        order = np.argsort(-result, axis=1, kind="stable")
        rows = np.arange(len(z))
        a, b = order[:, 0], order[:, 1]
        result[rows, a], result[rows, b] = result[rows, b].copy(), result[rows, a].copy()
    return result


def source_reply(e, states, allowed, mixed, edit):
    # A fresh query receives a board, its offered legal moves and the stated edit.
    raw = scoring(e, states, spec(0))
    ans = np.zeros_like(raw)
    for i, (row, menu) in enumerate(zip(raw, allowed)):
        options = sorted(np.flatnonzero(menu), key=lambda a: (-row[a], a))
        ans[i, options[1 if edit else 0]] = 1
    return 0.75 * ans + 0.25 * allowed / allowed.sum(1, keepdims=True) if mixed else ans


def evaluate(e, game, b, k, mixed, f, folder):
    tag = f"{k}_{mixed}_{f}"
    rng = np.random.default_rng(seed(game, b, 68, k, f))
    with np.load(folder / f"data_{k}_{mixed}.npz") as d:
        states = d["states"].tolist()
        ids = d["test"]
        test = [states[i] for i in ids]
        weights = d["equal"]
        visit = d["reach"]
        groups = d["groups"]
        counts = d["counts"]
        part = d["partition"]
    original = probabilities(e, test, spec(0))
    raw = scoring(e, test, spec(0))
    legal = raw > -1e28
    va = np.flatnonzero((part == 1) & (counts.sum(1) > 0))
    va = rng.choice(va, min(len(va), 96), replace=False)
    tr = np.flatnonzero((part == 0) & (counts.sum(1) > 0))
    tr = rng.choice(tr, min(len(tr), 128), replace=False)
    valid = [states[i] for i in va]
    train = [states[i] for i in tr]
    valfreq = counts[va] / counts[va].sum(1, keepdims=True)
    trfreq = counts[tr] / counts[tr].sum(1, keepdims=True)
    draws = {}
    for who in (0, 1):
        e.draw_value = 0
        p0 = scoring(e, train, spec(2, 5, who))
        e.draw_value = 1
        pd = scoring(e, train, spec(2, 5, who)) - p0
        fits = [float((decoded(p0 + h * pd, -1) * trfreq).sum(1).mean()) for h in np.linspace(-1, 1, 9)]
        grid = np.linspace(-1, 1, 9)
        among = grid[np.isclose(fits, max(fits), rtol=0, atol=1e-12)]
        draws[who] = float(min(among, key=lambda h: (abs(h), h)))
    pairs = opening_panel(e, game, b, k, f)
    rows = []
    cases = {}
    interventions = []
    multi = np.flatnonzero((legal.sum(1) > 1) & (weights > 0))
    menus = {}
    for event in ("remove_favorite", "remove_other", "swap_first_two"):
        menu = legal[multi].copy()
        indices = np.arange(len(multi))
        fav = raw[multi].argmax(1)
        if event == "remove_favorite":
            menu[indices, fav] = False
        elif event == "remove_other":
            other = np.where(
                menu & (np.arange(menu.shape[1]) != fav[:, None]), np.arange(menu.shape[1]), 999
            ).min(1)
            menu[indices, other] = False
        menus[event] = menu
    predicted = {event: {} for event in menus}
    for name, base in methods():
        model, rule, who, temp = base
        e.draw_value = draws[who]
        z = scoring(e, test, base) if rule not in (6, 7) else scoring(e, test, spec(2, 5))
        cases["score_" + name] = z.astype("f4")
        if model in (1, 2) and rule <= 5 or model in (6, 15):
            temps = (-1, *TEMPS)
            vz = scoring(e, valid, base)
            best_t = min(TEMPS, key=lambda t: np.abs(decoded(vz, t) - valfreq).sum(1).mean())
        else:
            temps = (1.0,) if mixed and model in (3, 4) else (temp,)
            best_t = None
        for t in temps:
            p = [model, rule, who, t]
            label = name + "_" + temperature_label(t)
            batch = e.profiles(test, p, True)
            prob = batch[:, :-2]
            calls = batch[:, -1]
            steps = batch[:, -2]
            match = (prob * original).sum(1)
            tv = np.abs(prob - original).sum(1) / 2
            pairmask = None
            if model in (1, 2) and rule in (1, 2, 3):
                dp = probabilities(e, test, spec(model, temp=t))
                if not mixed and t < 0:
                    a = dp.argmax(1) == original.argmax(1)
                    v = prob.argmax(1) == original.argmax(1)
                    pairmask = [int(((weights > 0) & x).sum()) for x in (a & v, a & ~v, ~a & v, ~a & ~v)]
            games = panel(e, p, pairs)
            outcomes = np.array(games)[:, 0]
            h2h = None
            if model in (1, 2) and rule in (1, 2, 3):
                h2h = panel(e, p, pairs, against=spec(model, temp=t))
            missing = ((prob == 0) * original).sum(1)
            outside = ((original == 0) * prob).sum(1)
            entropy = -(prob * np.log(prob.clip(1e-300))).sum(1)
            record = dict(
                method=name,
                temperature=t,
                selected=t == best_t,
                spec=p,
                agreement=wmean(match, weights),
                visit_agreement=wmean(match, visit),
                tv=wmean(tv, weights),
                visit_tv=wmean(tv, visit),
                win=float((outcomes == 1).mean()),
                draw=float((outcomes == 0).mean()),
                loss=float((outcomes == -1).mean()),
                points=float(outcomes.mean()),
                boards=int((weights > 0).sum()),
                probability_missing=wmean(missing, weights),
                probability_outside=wmean(outside, weights),
                entropy=wmean(entropy, weights),
                distinct_in_eight=wmean((1 - (1 - prob) ** 8).sum(1), weights),
                simulated_moves=wmean(steps, weights),
                source_calls=wmean(calls, weights),
                move_pairs=pairmask,
                strata=[
                    dict(
                        n=int(((groups == g) & (weights > 0)).sum()),
                        agreement=wmean(match, weights * (groups == g)),
                        tv=wmean(tv, weights * (groups == g)),
                    )
                    for g in range(3)
                ],
                draw_value=draws[who],
            )
            if t > 0:
                support = original > 0
                resid = np.where(support, z - t * np.log(original.clip(1e-300)), 0)
                centered = np.where(
                    support, resid - resid.sum(1, keepdims=True) / support.sum(1, keepdims=True), 0
                )
                record["centered_residual_rms"] = float(
                    np.sqrt(np.average((centered**2).sum(1) / support.sum(1), weights=weights))
                )
            rows.append(record)
            cases["p_" + label] = prob.astype("f4")
            cases["games_" + label] = np.asarray(games, dtype="i4")
            if h2h is not None:
                cases["duels_" + label] = np.asarray(h2h, dtype="i4")
            for event, menu in menus.items():
                edit = event == "swap_first_two"
                if rule == 6:
                    e.draw_value = -1
                    l = scoring(e, [test[i] for i in multi], spec(2, 5))
                    e.draw_value = 1
                    h = scoring(e, [test[i] for i in multi], spec(2, 5))
                    e.draw_value = draws[who]
                    answer = (
                        decoded(masked_scores(l, menu, edit), -1)
                        + decoded(masked_scores(h, menu, edit), -1)
                    ) / 2
                else:
                    modified = masked_scores(z[multi], menu, edit)
                    answer = decoded(modified, t)
                    if model == 0 or (rule == 7 and not mixed):
                        answer = decoded(masked_scores(raw[multi], menu, edit), -1)
                        if mixed:
                            answer = 0.75 * answer + 0.25 * menu / menu.sum(1, keepdims=True)
                predicted[event][label] = answer
    for event, menu in menus.items():
        # Fingerprint predictions before fetching the fresh restricted-menu answers.
        digest = {key: hashlib.sha256(p.tobytes()).hexdigest() for key, p in predicted[event].items()}
        truth = source_reply(e, [test[i] for i in multi], menu, mixed, event == "swap_first_two")
        cases["menu_" + event] = menu
        cases["answer_" + event] = truth.astype("f4")
        for key, p in predicted[event].items():
            distance = np.abs(p - truth).sum(1) / 2
            interventions.append(
                dict(
                    event=event,
                    method=key,
                    n=len(multi),
                    tv=wmean(distance, weights[multi]),
                    agreement=wmean((p * truth).sum(1), weights[multi]),
                    forecast=digest[key],
                )
            )
    cases["menu_indices"] = multi
    np.savez_compressed(folder / f"evaluation_{tag}.npz", **cases)
    save(folder / f"metrics_{tag}.jsonl.gz", rows)
    save(folder / f"changes_{tag}.jsonl.gz", interventions)
