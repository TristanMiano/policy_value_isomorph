"""Evaluate the original whole-play sum and 0.9-discounted VNM rules on exact training fits."""

from concurrent.futures import ProcessPoolExecutor
import numpy as np
from policy_value_isomorph.phase_iv_game import game, OUT as OLD
from policy_value_isomorph.phase_iv_source import actors
from policy_value_isomorph.phase_iv_metrics import Evaluation
from policy_value_isomorph.phase_iv_diagnostics import integrated
from policy_value_isomorph.phase_v import (
    OUT,
    SIZE,
    native,
    install_source,
    opening_panel,
    panel,
    seed,
    spec,
    save,
)
from policy_value_isomorph.phase_v_study import scoring
from diagnose_agreement import DEST, GAMES, outcomes
from diagnose_training_control import child_fit


def run(job):
    g, b = job
    folder = DEST / g / f"block_{b:02}"
    records = []
    if g == "tictactoe":
        bk = game()
        _, laws, ops = actors(b, OLD / f"block_{b:02}")
        ev = Evaluation(laws["fixed"], ops)
        ordered = bk.dec[np.argsort(bk.names[bk.dec])]
    else:
        e = native().Engine(g == "reversi4")
        install_source(e, OUT / g / f"block_{b:02}")
    for k in (0, 1):
        with np.load(folder / f"control_visits_{k}.npz") as data:
            roots = data["states"].tolist()
            count = data["counts"]
            sub = data["disjoint_parent_ids"]
            truth = count.argmax(1)
            all_ids = np.arange(len(roots))
            if g != "tictactoe":
                more = data["continuation_counts"]
                closed = {tuple(s): int(a) for s, a in zip(roots, truth)}
                closed.update({(x, o, t): a for x, o, t, a, n in more.tolist()})
                legal = scoring(e, roots, spec(0)) > -1e28
                child = np.zeros(legal.shape, int)
                mapping = {}
                for i, a in zip(*np.where(legal)):
                    s = tuple(e.next(roots[i], int(a)))
                    child[i, a] = mapping.setdefault(s, len(mapping))
                children = np.array(list(mapping), dtype=np.uint64)
        for f in (0, 1):
            rng = np.random.default_rng(seed(g, b, 162, k, f))
            if g != "tictactoe":
                for m in (1, 2):
                    e.set(m, m == 2, 0, rng.normal(0.5 / 12, 0.012, SIZE))
            else:
                background = rng.uniform(0, 1, bk.n)
            arrays = {}
            for label, m, ids in (
                ("Q_observed", 18, all_ids),
                ("Q_paths", 18, all_ids),
                ("V_pooled", 19, all_ids),
                ("V_disjoint", 19, sub),
            ):
                if g == "tictactoe":
                    if label == "Q_observed":
                        continue
                    positions = ordered[ids]
                    trained = np.zeros_like(laws["fixed"])
                    trained[ordered] = count / count.sum(1, keepdims=True)
                    if m == 18:
                        z = trained
                    else:
                        vals, seen = child_fit(bk.child, bk.legal, trained.argmax(1), positions)
                        z = np.where(seen, vals, background[: len(vals)])[bk.child]
                    decisions = bk.greedy(z)
                    utilities = dict(integrated(bk, z, decisions))
                else:
                    if label == "Q_paths":
                        e.fit_observed_choices(list(closed), list(closed.values()))
                    elif m == 18:
                        e.fit_observed_choices(roots, truth.tolist())
                    else:
                        vals, known = child_fit(child, legal, truth, ids)
                        e.fit_observed_children(children[known].tolist(), vals[known].tolist())
                for j, rule in ((2, "sum"), (3, "decay")):
                    if g == "tictactoe":
                        full = bk.greedy(utilities[rule])
                        scores = np.where(bk.legal[positions], utilities[rule][positions], -np.inf)
                        wdl = ev.outcomes(full).mean(0)[:-1].T @ ev.weights
                        rates = dict(win=wdl[0], draw=wdl[1], loss=wdl[2], points=wdl[0] - wdl[2])
                    else:
                        scores = scoring(e, [roots[i] for i in ids], spec(m, j))
                        rates = {}
                        games = np.array(panel(e, spec(m, j), opening_panel(e, g, b, k, f)))[:, 0]
                        outcomes(rates, games)
                        arrays[label + "_" + rule + "_games"] = games
                    chosen = scores.argmax(1)
                    hit = chosen == truth[ids]
                    maxima = scores.max(1)
                    strict = scores[np.arange(len(ids)), truth[ids]] < maxima
                    if label == "Q_paths":
                        assert not strict.any()
                        if rule == "decay":
                            assert hit.all()
                    records.append(
                        dict(
                            method=label,
                            rule=rule,
                            split=k,
                            fit=f,
                            boards=len(ids),
                            match=hit.mean(),
                            visit_match=np.average(hit, weights=count[ids].sum(1)),
                            strict=strict.mean(),
                            lost_tie=(~hit & ~strict).mean(),
                            **rates,
                        )
                    )
                    arrays[label + "_" + rule] = chosen.astype("i1")
            np.savez_compressed(folder / f"control_sums_{k}_{f}.npz", **arrays)
    save(folder / "training_sums.jsonl.gz", records)
    return job


if __name__ == "__main__":
    with ProcessPoolExecutor(6) as pool:
        for job in pool.map(run, [(g, b) for g in GAMES for b in range(32 if g == "tictactoe" else 24)]):
            print(job, flush=True)
