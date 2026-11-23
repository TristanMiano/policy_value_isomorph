"""Freeze exact-count fits, then compare move rules on their own training boards."""

from concurrent.futures import ProcessPoolExecutor
import numpy as np
from policy_value_isomorph.phase_iv_game import game, OUT as OLD
from policy_value_isomorph.phase_iv_source import actors
from policy_value_isomorph.phase_iv_metrics import Evaluation
from policy_value_isomorph.agreement_diagnosis import finite_endpoints
from policy_value_isomorph.phase_v import OUT, SIZE, centered_vnm, native, spec, seed
from policy_value_isomorph.phase_v import install_source, opening_panel, panel, save
from policy_value_isomorph.phase_v_study import scoring
from policy_value_isomorph.trajectory_utility import trajectory_scores
from diagnose_agreement import DEST, GAMES, outcomes


def disjoint_training(child, legal):
    """Take parents whose successor inputs are still unused; the function receives no labels."""
    used = set()
    keep = []
    for i, row in enumerate(child):
        candidates = set(row[legal[i]])
        if used.isdisjoint(candidates):
            used.update(candidates)
            keep.append(i)
    return np.array(keep, dtype=int)


def child_fit(children, legal, truth, parents):
    y = np.arange(legal.shape[1]) == truth[:, None]
    keys = children[parents][legal[parents]]
    totals = np.bincount(keys, minlength=children.max() + 1)
    selected = np.bincount(keys, weights=y[parents][legal[parents]], minlength=len(totals))
    return selected / totals.clip(1), totals > 0


def measure(
    rows, arrays, label, k, f, direct, terminal, centered, coverage, mask, truth, visits, game_rates, loss
):
    direct = np.where(mask, direct, -np.inf)
    terminal = np.where(mask, terminal, -np.inf)
    n = len(truth)
    idx = np.arange(n)
    all_paths = (coverage | ~mask).all(1)
    highest = terminal.max(1, keepdims=True)
    assert np.array_equal(direct.argmax(1), np.where(mask, centered, -np.inf).argmax(1))
    tied_choice = np.where(terminal == highest, direct, -np.inf).argmax(1)
    for rule, chosen in (
        ("direct", direct.argmax(1)),
        ("terminal", terminal.argmax(1)),
        ("centered", centered.argmax(1)),
        ("direct_ties", tied_choice),
    ):
        hit = chosen == truth
        row = dict(
            method=label,
            rule=rule,
            split=k,
            fit=f,
            boards=n,
            visits=int(visits.sum()),
            match=float(hit.mean()),
            visit_match=float(np.average(hit, weights=visits)),
            all_paths_covered=float(all_paths.mean()),
            covered_matches=int((hit & all_paths).sum()),
            covered_boards=int(all_paths.sum()),
            mse=float(loss),
        )
        if rule == "terminal":
            strict = terminal[idx, truth] < highest[:, 0]
            row.update(
                strict=float(strict.mean()),
                lost_tie=float((~strict & ~hit).mean()),
                changed=float((chosen != direct.argmax(1)).mean()),
            )
        if rule in game_rates:
            row.update(game_rates[rule])
        rows.append(row)
    arrays.update(
        {
            label + "_direct": direct,
            label + "_terminal": terminal,
            label + "_covered": coverage,
            label + "_truth": truth,
            label + "_visits": visits,
        }
    )


def large(g, b, folder):
    e = native().Engine(g == "reversi4")
    origin = OUT / g / f"block_{b:02}"
    install_source(e, origin)
    records = []
    for k in (0, 1):
        with np.load(origin / f"data_{k}_0.npz") as d:
            observed = d["counts"].sum(1) > 0
            states = d["states"][observed]
            counts = d["counts"][observed]
        roots = states.tolist()
        truth = counts.argmax(1)
        tot = counts.sum(1)
        observed_y = counts / tot[:, None]
        legal = scoring(e, roots, spec(0)) > -1e28
        assert np.array_equal(observed_y, (np.arange(counts.shape[1]) == truth[:, None]).astype(float))
        assert np.array_equal(truth, scoring(e, roots, spec(0)).argmax(1))
        child = np.zeros(legal.shape, int)
        index = {}
        for i, a in zip(*np.where(legal)):
            c = tuple(e.next(roots[i], int(a)))
            child[i, a] = index.setdefault(c, len(index))
        children = np.array(list(index), dtype=np.uint64)
        all_ids = np.arange(len(roots))
        subset = disjoint_training(child, legal)
        more = e.continuation_visits(roots)
        # Union of freely observed training moves, including moves on continuation paths.
        src = {tuple(s): int(a) for s, a in zip(roots, truth)}
        for x, o, t, a, n in more.tolist():
            old = src.setdefault((x, o, t), a)
            assert old == a
        np.savez_compressed(
            folder / f"control_visits_{k}.npz",
            states=states,
            counts=counts,
            disjoint_parent_ids=subset,
            continuation_counts=more,
        )
        closed = list(src)
        closed_moves = list(src.values())
        for f in (0, 1):
            rng = np.random.default_rng(seed(g, b, 162, k, f))
            for m in (1, 2):
                e.set(m, m == 2, 0, rng.normal(0.5 / 12, 0.012, SIZE))
            arrays = {}
            for label, m, ids in (
                ("Q_observed", 18, all_ids),
                ("Q_paths", 18, all_ids),
                ("V_pooled", 19, all_ids),
                ("V_disjoint", 19, subset),
            ):
                if label == "Q_observed":
                    e.fit_observed_choices(roots, truth.tolist())
                elif label == "Q_paths":
                    e.fit_observed_choices(closed, closed_moves)
                else:
                    values, seen = child_fit(child, legal, truth, ids)
                    e.fit_observed_children(children[seen].tolist(), values[seen].tolist())
                test = [roots[i] for i in ids]
                mask = legal[ids]
                direct = scoring(e, test, spec(m))
                z = scoring(e, test, spec(m, 1))
                centered = scoring(e, test, centered_vnm(m))
                covered = e.fitted_paths(test, m)
                rates = {}
                for rule, s in (
                    ("direct", spec(m)),
                    ("terminal", spec(m, 1)),
                    ("centered", centered_vnm(m)),
                ):
                    games = np.array(panel(e, s, opening_panel(e, g, b, k, f)))[:, 0]
                    rates[rule] = {}
                    outcomes(rates[rule], games)
                    arrays[label + "_games_" + rule] = games
                assert np.array_equal(arrays[label + "_games_direct"], arrays[label + "_games_centered"])
                if label == "Q_paths":
                    assert covered[mask].all()
                err = np.square(direct[mask] - observed_y[ids][mask]).mean()
                if label != "V_pooled":
                    assert err == 0 and np.array_equal(direct.argmax(1), truth[ids])
                measure(
                    records,
                    arrays,
                    label,
                    k,
                    f,
                    direct,
                    z,
                    centered,
                    covered,
                    mask,
                    truth[ids],
                    tot[ids],
                    rates,
                    err,
                )
                records[-1]["fitted_parents"] = len(closed) if label == "Q_paths" else len(ids)
            np.savez_compressed(folder / f"control_cases_{k}_{f}.npz", **arrays)
    return records


def small(bid, folder):
    b = game()
    origin = OLD / f"block_{bid:02}"
    _, laws, opponents = actors(bid, origin)
    evaluator = Evaluation(laws["fixed"], opponents)
    legal = b.legal
    ids = b.dec[np.argsort(b.names[b.dec])]
    truth = laws["fixed"].argmax(1)
    subset = ids[disjoint_training(b.child[ids], legal[ids])]
    rows = []
    for k in (0, 1):
        with np.load(origin / "baseline.npz") as d:
            counts = b.dense(d[f"fixed_{k}_counts"])
        totals = counts.sum(1)
        assert (totals[ids] > 0).all() and np.array_equal(counts[ids].argmax(1), truth[ids])
        learned = counts / totals[:, None].clip(1)
        np.savez_compressed(
            folder / f"control_visits_{k}.npz",
            states=b.names[ids],
            counts=counts[ids],
            disjoint_parent_ids=np.flatnonzero(np.isin(ids, subset)),
        )
        for f in (0, 1):
            rng = np.random.default_rng(seed("tictactoe", bid, 162, k, f))
            background = rng.uniform(0, 1, b.n)
            arrays = {}
            for label, parents in (("Q_paths", ids), ("V_pooled", ids), ("V_disjoint", subset)):
                if label == "Q_paths":
                    z = learned
                    seen = legal
                else:
                    values, used = child_fit(b.child, legal, learned.argmax(1), parents)
                    v = np.where(used, values, background[: len(used)])
                    z = v[b.child]
                    seen = used[b.child] & legal
                z = np.where(legal, z, -1e30)
                moves = z.argmax(1)
                last, length = finite_endpoints(b, moves)
                final = np.where(legal, b.side[:, None] * b.side[b.parent[last]] * z[legal][last], -1e30)
                paths = b.sampler(b.point(moves)).rollout_paths(1, 0)
                best = z.max(1)[b.parent]
                sign = b.side[b.parent]
                utility = trajectory_scores(paths, sign * z[legal], x_local_best=sign * best)[
                    "vnm_centered"
                ][0]
                centered = np.where(legal, b.dense(sign * utility), -1e30)
                valid = seen.copy()
                for level in reversed(b.layers):
                    child = b.child[level]
                    valid[level] &= b.ends[child] | (
                        (seen | ~legal).all(1)[child] & valid[child, moves[child]]
                    )
                rates = {}
                for rule, choice in (
                    ("direct", moves),
                    ("terminal", final.argmax(1)),
                    ("centered", centered.argmax(1)),
                ):
                    wdl = evaluator.outcomes(b.point(choice)).mean(0)[:-1].T @ evaluator.weights
                    rates[rule] = dict(win=wdl[0], draw=wdl[1], loss=wdl[2], points=wdl[0] - wdl[2])
                err = np.square(
                    (z[parents] - counts[parents] / totals[parents, None])[legal[parents]]
                ).mean()
                if label != "V_pooled":
                    assert err == 0 and np.array_equal(moves[parents], truth[parents])
                measure(
                    rows,
                    arrays,
                    label,
                    k,
                    f,
                    z[parents],
                    final[parents],
                    centered[parents],
                    valid[parents],
                    legal[parents],
                    truth[parents],
                    totals[parents],
                    rates,
                    err,
                )
                rows[-1]["fitted_parents"] = len(parents)
            np.savez_compressed(folder / f"control_cases_{k}_{f}.npz", **arrays)
    return rows


def run(job):
    g, b, force = job
    folder = DEST / g / f"block_{b:02}"
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "training_control.jsonl.gz"
    if force or not out.exists():
        rows = small(b, folder) if g == "tictactoe" else large(g, b, folder)
        tmp = folder / "pending_control.gz"
        save(tmp, rows)
        tmp.replace(out)
    return job


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+", choices=GAMES, default=GAMES)
    parser.add_argument("--blocks", nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        for job in pool.map(
            run,
            [
                (g, b, args.force)
                for g in args.games
                for b in (args.blocks if args.blocks is not None else range(32 if g == "tictactoe" else 24))
            ],
        ):
            print(job, flush=True)
