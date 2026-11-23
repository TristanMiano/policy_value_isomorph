"""Run matched, fixed-original interventions without advancing the roadmap."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
import torch

from policy_value_isomorph.agreement_diagnosis import (
    classify,
    choice_probabilities,
    finite_endpoints,
    fit_choice,
)
from policy_value_isomorph.phase_iv_game import OUT as OLD, game as finite, seed as old_seed
from policy_value_isomorph.phase_iv_fit import designs, masks as finite_masks, selected
from policy_value_isomorph.phase_iv_metrics import Evaluation
from policy_value_isomorph.phase_iv_source import actors
from policy_value_isomorph.phase_v import (
    OUT,
    SIZE,
    fit_table,
    install_source,
    moved,
    native,
    centered_vnm,
    opening_panel,
    panel,
    prefixes,
    read,
    save,
    seed,
    spec,
    split_of,
)
from policy_value_isomorph.phase_v_study import masks, scoring, setup_fit
from policy_value_isomorph.trajectory_utility import trajectory_scores

DEST = OUT.parent / "agreement_diagnosis"
GAMES = ("tictactoe", "connect4", "reversi4")


def record(rows, arrays, name, z, legal, truth, weights):
    """Save full candidate scores and classify every case before applying population weights."""
    result = classify(z, legal, truth)
    row = dict(method=name, n=len(weights), weight=float(weights.sum()))
    row.update({key: float(np.average(x, weights=weights)) for key, x in result.items()})
    assert np.isclose(row["match"] + row["strict"] + row["lost_tie"], 1)
    rows.append(row)
    arrays[name] = np.where(legal, z, -np.inf)
    return row


def lookup_score(children, legal, truth, ids):
    """Best squared-loss V score on this explicitly labeled finite candidate population."""
    y = (np.arange(legal.shape[1]) == truth[:, None]) & legal
    child = children[legal]
    sums = np.bincount(child, weights=y[legal])
    totals = np.bincount(child)
    values = sums / totals.clip(1)
    return values[children[ids]]


def extra_games(engine, game, block, k, states, counts, part, folder):
    rng = np.random.default_rng(seed(game, block, 151, k))
    salt = seed(game, block, 11, k)
    if game == "reversi4":
        pool = [tuple(s) for s in engine.all() if engine.legal(s) and split_of(engine, s, salt) == 0]
        starts = [pool[i] for i in rng.choice(len(pool), 4096, replace=False)]
    else:
        starts = []
        while len(starts) < 4096:
            starts.extend(s for s in prefixes(engine, 1500, rng) if split_of(engine, s, salt) == 0)
        starts = starts[:4096]
    extra = Counter()
    for s in starts:
        path = engine.trajectory(s, spec(0), spec(0), int(rng.integers(2**32)))
        for h, c in zip(path, path[1:]):
            h = tuple(h)
            if split_of(engine, h, salt) == 0:
                extra[h, moved(h, c, game == "reversi4")] += 1
    original = {tuple(s): i for i, s in enumerate(states)}
    new = sorted({s for s, a in extra if s not in original})
    lookup = dict(original)
    lookup.update({s: i + len(states) for i, s in enumerate(new)})
    combined = np.vstack([counts, np.zeros((len(new), counts.shape[1]), dtype=int)])
    for (s, a), n in extra.items():
        combined[lookup[s], a] += n
    all_states = states + [list(s) for s in new]
    partition = np.r_[part, np.zeros(len(new), dtype=int)]
    np.savez_compressed(
        folder / f"extra_{k}.npz",
        starts=starts,
        states=all_states,
        counts=combined,
        training_partition=partition,
        extra_observations=sum(extra.values()),
    )
    return all_states, combined, partition


def outcomes(row, games):
    games = np.asarray(games)
    row.update(
        win=float(np.mean(games == 1)),
        draw=float(np.mean(games == 0)),
        loss=float(np.mean(games == -1)),
        points=float(games.mean()),
    )


def train_agreement(z, legal, truth, train):
    full = train.any(1) & (train == legal).all(1)
    return dict(train_boards=int(full.sum()), train_match=float((z[full].argmax(1) == truth[full]).mean()))


def inspect_fit(
    rows,
    arrays,
    net,
    raw,
    direct,
    final,
    source_final,
    exact_copies,
    exact_source,
    rooted_copies,
    rooted_source,
    signs,
    legal,
    truth,
    w,
):
    for label, z in (
        ("raw", raw),
        ("direct", direct),
        ("terminal", final),
        ("fitted_score_source_continuation", source_final),
        ("exact_score_fitted_continuation", exact_copies),
        ("exact_score_source_continuation", exact_source),
        ("centered", rooted_copies),
        ("centered_source", rooted_source),
    ):
        record(rows, arrays, net + "_" + label, z, legal, truth, w)
    assert np.array_equal(
        np.where(legal, direct, -np.inf).argmax(1), np.where(legal, rooted_copies, -np.inf).argmax(1)
    )
    a = direct.argmax(1)
    b = final.argmax(1)
    idx = np.arange(len(w))
    # Disjoint reasons for a changed selection. A zero-valued last move can also tie across signs.
    changed = a != b
    diff = signs[idx, a] != signs[idx, b]
    vals = {
        "changed_last_mover": changed & diff,
        "changed_same_mover_strict": changed & ~diff & (final[idx, a] < final[idx, b]),
        "changed_same_mover_tie": changed & ~diff & (final[idx, a] == final[idx, b]),
        "unchanged": ~changed,
        "lost_match": (a == truth) & (b != truth),
        "gained_match": (a != truth) & (b == truth),
    }
    rows.append(
        dict(
            method=net + "_switch",
            n=len(w),
            **{key: float(np.average(v, weights=w)) for key, v in vals.items()},
        )
    )


def structured(rows, arrays, fit, counts, legal, context, train, ids, truth, w, execute):
    for informed, name in ((False, "action_only"), (True, "rule_inputs")):
        x = context if informed else np.zeros_like(context)
        wt, details = fit_choice(counts[train], legal[train], x[train], fit)
        if not informed:
            wt[legal.shape[1] :] = 0.0  # Zero-input coefficients minimize the weight penalty at zero.
        p = choice_probabilities(wt, legal[ids], x[ids])
        r = record(rows, arrays, name, p, legal[ids], truth, w)
        r.update(details)
        arrays[name + "_weights"] = wt
        execute(wt, informed, p, r)


def run_large(game, b, folder):
    eng = native().Engine(game == "reversi4")
    origin = OUT / game / f"block_{b:02}"
    install_source(eng, origin)
    eng.configure(False, 0)
    all_rows = []
    for k in (0, 1):
        with np.load(origin / f"data_{k}_0.npz") as dat:
            states = dat["states"].tolist()
            counts = dat["counts"]
            part = dat["partition"]
            primary = dat["equal"] > 0
            ids = dat["test"][primary]
            w = dat["equal"][primary]
        zsource = scoring(eng, states, spec(0))
        legal = zsource > -1e28
        truth = zsource.argmax(1)
        assert np.all((counts == 0) | (np.arange(counts.shape[1]) == truth[:, None]))
        test = [states[i] for i in ids]
        target = truth[ids]
        menu = legal[ids]
        ctx = eng.rule_inputs(states)
        qmask, vmask = masks(eng, states, part, legal, counts)
        # For Reversi pool every reachable decision and all incoming edges. In Connect Four
        # only this partition's observed and evaluation parents are available.
        population = eng.all() if game == "reversi4" else states
        population = [s for s in population if eng.legal(s)]
        poptruth = scoring(eng, population, spec(0))
        popmask = poptruth > -1e28
        popids = {tuple(s): i for i, s in enumerate(population)}
        childmap = {}
        children = np.zeros(popmask.shape, int)
        for i, a in zip(*np.where(popmask)):
            c = tuple(eng.next(population[i], int(a)))
            children[i, a] = childmap.setdefault(c, len(childmap))
        projection = lookup_score(children, popmask, poptruth.argmax(1), [popids[tuple(s)] for s in test])
        extended, ecount, epart = extra_games(eng, game, b, k, states, counts, part, folder)
        # Features and fitting masks are shared between initializations; confirmation boards stay reserved.
        edata = {}
        ey = ecount / ecount.sum(1, keepdims=True).clip(1)
        feature_sets = {model: eng.features(extended, model == 2, 0) for model in (1, 2)}
        q_extended, v_extended = masks(eng, extended, epart, feature_sets[1][:, :, 0] >= 0, ecount)
        for model in (1, 2):
            features = feature_sets[model]
            tr, va = v_extended if model == 1 else q_extended
            assert np.array_equal(va[: len(states)], (vmask if model == 1 else qmask)[1])
            edata[model] = (features, tr, va)
        for f in (0, 1):
            setup_fit(eng, origin, k, 0, f)
            rows = []
            arrays = dict(states=np.asarray(test), weights=w, truth=target, counts=counts[ids], legal=menu)
            pairs = opening_panel(eng, game, b, k, f)
            record(
                rows,
                arrays,
                "exact_direct",
                (np.arange(menu.shape[1]) == target[:, None]).astype(float),
                menu,
                target,
                w,
            )
            record(rows, arrays, "projection_V", projection, menu, target, w)
            with np.load(origin / f"evaluation_{k}_0_{f}.npz") as published:
                for model, net in ((1, "V"), (2, "Q")):
                    direct = scoring(eng, test, spec(model))
                    raw = eng.raw_scores(states, model)
                    terminal = scoring(eng, test, spec(model, 1))
                    assert np.array_equal(
                        direct.argmax(1), published["p_" + net + "_direct_max"][primary].argmax(1)
                    )
                    assert np.array_equal(
                        terminal.argmax(1),
                        published["p_" + net + "_VNM_terminal_copies_max"][primary].argmax(1),
                    )
                    probes = [eng.terminal_probe(test, model, who) for who in (model, 0)]
                    end, orig = probes
                    centered = scoring(eng, test, centered_vnm(model))
                    other = scoring(eng, test, centered_vnm(model, True))
                    reconstructed = end[:, :, 1] * end[:, :, 3]
                    assert np.array_equal(reconstructed[menu], terminal[menu])
                    ascores = orig[:, :, 1] * orig[:, :, 3]
                    assert np.array_equal(ascores[menu], scoring(eng, test, spec(model, 1, 1))[menu])
                    inspect_fit(
                        rows,
                        arrays,
                        net,
                        raw[ids],
                        direct,
                        terminal,
                        ascores,
                        end[:, :, 1] * end[:, :, 2],
                        orig[:, :, 1] * orig[:, :, 2],
                        centered,
                        other,
                        end[:, :, 1],
                        menu,
                        target,
                        w,
                    )
                    arrays[net + "_endpoints"] = np.stack(probes)
                    directrow = next(r for r in rows if r["method"] == net + "_direct")
                    directrow.update(
                        train_agreement(
                            np.where(legal, raw.clip(0, 1), -1e30),
                            legal,
                            truth,
                            (vmask if model == 1 else qmask)[0],
                        )
                    )
                    base_games = published["games_" + net + "_direct_max"][:, 0]
                    outcomes(directrow, base_games)
                    games = np.asarray(panel(eng, centered_vnm(model), pairs))[:, 0]
                    assert np.array_equal(base_games, games)
                    outcomes(next(r for r in rows if r["method"] == net + "_centered"), games)
                    outcomes(
                        next(r for r in rows if r["method"] == net + "_terminal"),
                        published["games_" + net + "_VNM_terminal_copies_max"][:, 0],
                    )
                    if b == k == f == 0:
                        per_board_example(
                            eng, game, folder, net, model, test, counts[ids], menu, target, direct, terminal
                        )

            def deploy(wt, informed, p, row):
                coeff = np.zeros(SIZE)
                coeff[: len(wt)] = wt
                eng.set(16, True, 0, coeff)
                actual = scoring(eng, test, spec(16))
                assert np.allclose(actual[menu], p[menu], atol=1e-12)
                outcomes(row, np.asarray(panel(eng, spec(16), pairs))[:, 0])
                if informed:
                    v = scoring(eng, test, centered_vnm(16))
                    rr = record(rows, arrays, "rule_inputs_centered", v, menu, target, w)
                    assert np.array_equal(actual.argmax(1), v.argmax(1))
                    outcomes(rr, np.asarray(panel(eng, centered_vnm(16), pairs))[:, 0])

            structured(
                rows,
                arrays,
                seed(game, b, 153, k, f),
                counts,
                legal,
                ctx,
                qmask[0].any(1),
                ids,
                target,
                w,
                deploy,
            )
            fitweights = {}
            for model, net in ((1, "V"), (2, "Q")):
                feats, tr, va = edata[model]
                weights, settings, _ = fit_table(feats, ey, tr, va, seed(game, b, 40 + model, k, f))
                eng.set(model, model == 2, 0, weights)
                fitweights[net] = weights
                z = scoring(eng, test, spec(model))
                r = record(rows, arrays, net + "_expanded", z, menu, target, w)
                r.update(settings)
                outcomes(r, np.asarray(panel(eng, spec(model), pairs))[:, 0])
            np.savez_compressed(folder / f"expanded_weights_{k}_{f}.npz", **fitweights)
            np.savez_compressed(folder / f"boards_{k}_{f}.npz", **arrays)
            all_rows.extend(dict(split=k, fit=f, **r) for r in rows)
    return all_rows


def per_board_example(engine, game, folder, net, model, test, counts, legal, target, direct, terminal):
    ids = np.flatnonzero((direct.argmax(1) == target) & (terminal.argmax(1) != target))
    if not len(ids):
        return
    j = next((i for i in ids if counts[i].sum() > 0), ids[0])
    s = test[j]
    branches = []
    for a in np.flatnonzero(legal[j]):
        path = [s] + engine.trajectory(engine.next(s, int(a)), spec(model), spec(model), 0)
        moves = [moved(h, c, game == "reversi4") for h, c in zip(path, path[1:])]
        branches.append(
            dict(
                move=int(a),
                direct=direct[j, a],
                vnm=terminal[j, a],
                moves=moves,
                final_first_player=(path[-2][2] == 0),
                final_fitted=scoring(engine, [path[-2]], spec(model))[0, moves[-1]],
            )
        )
    (folder / f"example_{net}.json").write_text(
        json.dumps(
            dict(state=s, source=int(target[j]), counts=counts[j].tolist(), branches=branches), indent=2
        )
        + "\n"
    )


def run_finite(block, folder):
    board = finite()
    origin = OLD / f"block_{block:02}"
    _, laws, opponents = actors(block, origin)
    original = laws["fixed"]
    truth = original.argmax(1)
    evaluator = Evaluation(original, opponents)
    context = board.optimal[:, :, None].astype(float)
    all_rows = []

    def finite_games(row, choices):
        seat = evaluator.outcomes(board.point(choices))
        rates = seat.mean(0)[:-1].T @ evaluator.weights
        row.update(win=rates[0], draw=rates[1], loss=rates[2], points=rates[0] - rates[2])

    def centered_scores(z, actor):
        sampler = board.sampler(board.point(actor))
        paths = sampler.rollout_paths(1, 0)
        maxima = np.where(board.legal, z, -np.inf).max(1)
        signed = board.side[board.parent]
        result = trajectory_scores(
            paths, signed * z[board.legal], x_local_best=signed * maxima[board.parent]
        )
        return np.where(board.legal, board.dense(signed * result["vnm_centered"][0]), -1e30)

    curves = read(origin / "curves.jsonl.gz")
    with np.load(origin / "visits.npz") as archived, np.load(origin / "baseline.npz") as baseline:
        for k in (0, 1):
            counts = board.dense(baseline[f"fixed_{k}_counts"])
            assert np.array_equal((counts > 0).astype(float), original)
            splits = board.split(block, k)
            ids = np.flatnonzero((splits == 2) & ~board.ends)
            legal = board.legal[ids]
            target = truth[ids]
            w = np.ones(len(ids))
            proj = lookup_score(board.child, board.legal, truth, ids)
            for f in (0, 1):
                rows = []
                arrays = dict(
                    states=board.names[ids], weights=w, truth=target, counts=counts[ids], legal=legal
                )
                record(rows, arrays, "exact_direct", original[ids], legal, target, w)
                record(rows, arrays, "projection_V", proj, legal, target, w)
                for net in ("V", "Q"):
                    train, valid, _ = finite_masks(splits, net)
                    init = old_seed(block, 51 if net == "V" else 52, k, f)
                    frequency = counts / counts.sum(1, keepdims=True).clip(1)
                    raw, settings = selected(designs()[net], frequency[board.legal], train, valid, init)
                    scores = np.where(board.legal, board.dense(raw.clip(0, 1)), -1e30)
                    direct = scores.argmax(1)
                    key = f"fixed_{k}_{f}_{net}_"
                    assert np.array_equal(archived[key + "direct_choice"], direct[board.dec])
                    probes = []
                    for actor in (direct, truth):
                        end, length = finite_endpoints(board, actor)
                        sign = board.side[:, None] * board.side[board.parent[end]]
                        probes.append(
                            np.stack(
                                (length, sign, original[board.legal][end], scores[board.legal][end]), axis=2
                            )[ids]
                        )
                    end, orig = probes
                    terminal = np.where(legal, end[:, :, 1] * end[:, :, 3], -1e30)
                    term_all = np.zeros(board.n, int)
                    term_all[board.dec] = archived[key + "vnm_terminal_copies_choice"]
                    assert np.array_equal(terminal.argmax(1), term_all[ids])
                    centered = centered_scores(scores, direct)
                    inspect_fit(
                        rows,
                        arrays,
                        net,
                        board.dense(raw)[ids],
                        scores[ids],
                        terminal,
                        orig[:, :, 1] * orig[:, :, 3],
                        end[:, :, 1] * end[:, :, 2],
                        orig[:, :, 1] * orig[:, :, 2],
                        centered[ids],
                        centered_scores(scores, truth)[ids],
                        end[:, :, 1],
                        legal,
                        target,
                        w,
                    )
                    arrays[net + "_endpoints"] = np.stack(probes)
                    current = {r["method"]: r for r in rows}
                    current[net + "_direct"].update(
                        train_agreement(scores, board.legal, truth, board.dense(train))
                    )
                    current[net + "_direct"]["training"] = settings["train_loss"]
                    for label, m in (
                        (net + "_direct", direct),
                        (net + "_centered", centered.argmax(1)),
                        (net + "_terminal", term_all),
                    ):
                        finite_games(current[label], m)
                    half = next(
                        r
                        for r in curves
                        if (r["source"], r["split"], r["fit"], r["network"], r["method"])
                        == ("fixed", k, f, net, "half_training")
                    )
                    rows.append(
                        dict(
                            method=net + "_half",
                            match=half["agreement"],
                            win=half["win"],
                            training=half["train_loss"],
                        )
                    )
                    if block == k == f == 0:
                        wrong = np.flatnonzero((direct[ids] == target) & (terminal.argmax(1) != target))
                        if len(wrong):
                            jj = wrong[0]
                            i = ids[jj]
                            branches = []
                            for a in np.flatnonzero(board.legal[i]):
                                h = i
                                act = a
                                seq = []
                                while True:
                                    seq.append(int(act))
                                    c = board.child[h, act]
                                    if board.ends[c]:
                                        break
                                    h = c
                                    act = direct[c]
                                branches.append(
                                    dict(
                                        move=int(a),
                                        direct=scores[i, a],
                                        vnm=terminal[jj, a],
                                        moves=seq,
                                        final_first_player=bool(board.side[h] == 1),
                                        final_fitted=scores[h, act],
                                    )
                                )
                            (folder / f"example_{net}.json").write_text(
                                json.dumps(
                                    dict(
                                        state=str(board.names[i]),
                                        source=int(target[jj]),
                                        counts=counts[i].tolist(),
                                        branches=branches,
                                    ),
                                    indent=2,
                                )
                                + "\n"
                            )

                def deploy(weights, informed, p, row):
                    full = np.zeros_like(original)
                    x = context if informed else np.zeros_like(context)
                    full[board.dec] = choice_probabilities(weights, board.legal[board.dec], x[board.dec])
                    actor = full.argmax(1)
                    finite_games(row, actor)
                    if informed:
                        centers = centered_scores(full, actor)
                        rr = record(rows, arrays, "rule_inputs_centered", centers[ids], legal, target, w)
                        assert np.array_equal(centers[board.dec].argmax(1), actor[board.dec])
                        finite_games(rr, centers.argmax(1))

                structured(
                    rows,
                    arrays,
                    old_seed(block, 153, k, f),
                    counts,
                    board.legal,
                    context,
                    (splits == 0) & ~board.ends,
                    ids,
                    target,
                    w,
                    deploy,
                )
                np.savez_compressed(folder / f"boards_{k}_{f}.npz", **arrays)
                all_rows.extend(dict(split=k, fit=f, **r) for r in rows)
    return all_rows


def run(job):
    game, b = job
    torch.set_num_threads(1)
    folder = DEST / game / f"block_{b:02}"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "diagnosis.jsonl.gz"
    if not output.exists():
        rows = run_finite(b, folder) if game == "tictactoe" else run_large(game, b, folder)
        tmp = folder / "pending.gz"
        save(tmp, rows)
        tmp.replace(output)
    return job


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+", choices=GAMES, default=GAMES)
    parser.add_argument("--blocks", nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    jobs = [
        (g, b)
        for g in args.games
        for b in (args.blocks if args.blocks is not None else range(32 if g == "tictactoe" else 24))
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        for finished in pool.map(run, jobs):
            print(*finished, flush=True)
