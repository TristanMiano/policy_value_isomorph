"""Fit normalized action scores to visits on the original and expanded training partitions."""

import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from policy_value_isomorph.phase_v import (
    OUT,
    SIZE,
    native,
    install_source,
    centered_vnm,
    opening_panel,
    panel,
    seed,
    spec,
    save,
)
from policy_value_isomorph.phase_v_study import scoring
from diagnose_agreement import DEST, outcomes, record


def fit_frequency_choice(features, counts, start):
    legal = features[:, :, 0] >= 0
    keys, inverse = np.unique(features[legal], return_inverse=True)
    n, groups = features[legal].shape
    x = csr_matrix((np.ones(n * groups), inverse.ravel(), np.arange(n + 1) * groups), shape=(n, len(keys)))
    xt = x.T.tocsr()
    sizes = legal.sum(axis=1)
    idx = np.r_[0, np.cumsum(sizes)[:-1]]
    y = (counts / counts.sum(1, keepdims=True))[legal]

    def objective(w):
        z = x @ w
        centered = z - np.repeat(np.maximum.reduceat(z, idx), sizes)
        exp = np.exp(centered)
        total = np.add.reduceat(exp, idx)
        p = exp / np.repeat(total, sizes)
        nll = np.log(total).sum() - y @ centered
        return nll + 0.05 * (w @ w), xt @ (p - y) + 0.1 * w

    init = np.random.default_rng(start).normal(0, 0.012, len(keys))
    result = minimize(
        objective, init, jac=True, method="L-BFGS-B", options=dict(maxiter=1500, maxls=60, ftol=1e-11)
    )
    if not result.success:
        raise ArithmeticError(result.message)
    w = np.zeros(SIZE)
    w[keys] = result.x
    return w, dict(
        boards=len(sizes),
        candidates=n,
        parameters=len(keys),
        updates=int(result.nit),
        training=float((result.fun - 0.05 * (result.x @ result.x)) / len(sizes)),
        objective=float(result.fun / len(sizes)),
    )


def run(job):
    game, b = job
    folder = DEST / game / f"block_{b:02}"
    origin = OUT / game / f"block_{b:02}"
    out = folder / "choice_fit.jsonl.gz"
    if out.exists():
        return job
    eng = native().Engine(game == "reversi4")
    install_source(eng, origin)
    rows = []
    settings = json.loads((origin / "source.json").read_text())
    for k in (0, 1):
        with np.load(origin / f"data_{k}_0.npz") as base, np.load(folder / f"extra_{k}.npz") as expanded:
            selected = base["equal"] > 0
            test = base["states"][base["test"][selected]].tolist()
            w = base["equal"][selected]
            actual = scoring(eng, test, spec(0))
            legal = actual > -1e28
            truth = actual.argmax(1)
            teacher = np.array([eng.teacher(s, settings["teacher"], settings["priority"]) for s in test])
            for dataset, data, split_name in (
                ("original", base, "partition"),
                ("expanded", expanded, "training_partition"),
            ):
                counts = data["counts"]
                tr = (data[split_name] == 0) & (counts.sum(1) > 0)
                features = eng.features(data["states"][tr].tolist(), True, 0)
                for f in (0, 1):
                    table, loss = fit_frequency_choice(features, counts[tr], seed(game, b, 42, k, f))
                    eng.set(3, True, 0, table)
                    z = scoring(eng, test, spec(17))
                    method = "Q_normalized_" + dataset
                    arr = {}
                    r = record(rows, arr, method, z, legal, truth, w)
                    r.update(split=k, fit=f, **loss)
                    pairs = opening_panel(eng, game, b, k, f)
                    games = np.asarray(panel(eng, spec(17), pairs))[:, 0]
                    outcomes(r, games)
                    score = scoring(eng, test, centered_vnm(17))
                    rr = record(rows, arr, method + "_centered", score, legal, truth, w)
                    rr.update(split=k, fit=f)
                    centered_games = np.asarray(panel(eng, centered_vnm(17), pairs))[:, 0]
                    assert np.array_equal(games, centered_games) and np.array_equal(
                        z.argmax(1), score.argmax(1)
                    )
                    outcomes(rr, centered_games)
                    np.savez_compressed(
                        folder / f"normalized_{dataset}_{k}_{f}.npz",
                        table=table,
                        score=z,
                        games=games,
                        centered_games=centered_games,
                        teacher=teacher,
                    )
    save(out, rows)
    return job


if __name__ == "__main__":
    with ProcessPoolExecutor(5) as pool:
        for job in pool.map(run, [(g, b) for b in range(24) for g in ("connect4", "reversi4")]):
            print(job, flush=True)
