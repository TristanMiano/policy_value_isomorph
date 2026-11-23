"""Choose both the board and the offered pair of moves using predictions alone."""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import numpy as np
from policy_value_isomorph.phase_iv_statistics import holm
from policy_value_isomorph.phase_v import OUT, read, save, seed, native, install_source, spec
from policy_value_isomorph.phase_v_study import scoring
from report_phase_v import paired, summary


def entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log(p) - (1 - p) * np.log1p(-p)


def run(job):
    game, b = job
    folder = OUT / game / f"block_{b:02}"
    rows = []
    engine = native().Engine(game == "reversi4")
    install_source(engine, folder)
    for k, mix in itertools.product(range(2), repeat=2):
        rng = np.random.default_rng(seed(game, b, 117, k, mix))
        with np.load(folder / f"data_{k}_{mix}.npz") as d:
            weight = d["equal"]
            visit = d["reach"]
            states = d["states"][d["test"]]
        with (
            np.load(folder / f"evaluation_{k}_{mix}_0.npz") as f0,
            np.load(folder / f"evaluation_{k}_{mix}_1.npz") as f1,
        ):
            scores = np.stack([f0["score_Q_direct"], f1["score_Q_direct"]]).astype(float)
            # Gather full-precision original rankings only after the menus and both model
            # predictions have been fixed. The original weights are never used to choose a question.
        legal = scores[0] > -1e28
        multi = rng.permutation(np.flatnonzero((weight > 0) & (legal.sum(1) > 1)))
        pool, held = np.array_split(multi, 2)
        options = np.asarray(
            [(i, a, v) for i in pool for a, v in itertools.combinations(np.flatnonzero(legal[i]), 2)]
        )
        fresh = np.asarray(
            [(i, *sorted(rng.choice(np.flatnonzero(legal[i]), 2, replace=False))) for i in held]
        )

        def prob(menu):
            i, a, v = menu.T
            return 1 / (1 + np.exp((scores[:, i, v] - scores[:, i, a]) / 0.25))

        models = prob(options)
        forecasts = prob(fresh)

        def answer(menu):
            i, a, v = menu.T
            raw = scoring(engine, states[i].tolist(), spec(0))
            idx = np.arange(len(menu))
            favorite = (raw[idx, a] >= raw[idx, v]).astype(float)
            return 0.75 * favorite + 0.125 if mix else favorite

        # Use common per-menu uniforms for strategies that ask the same question.
        uniforms = rng.random(len(options))
        random_order = rng.permutation(len(options))
        freq_order = random_order[np.argsort(-visit[options[random_order, 0]], kind="stable")]
        for strategy in ("information", "random", "visitation"):
            prior = 0.5
            available = np.ones(len(options), bool)
            asked = []
            answers = []
            for t in range(24):
                if strategy == "information":
                    mean = prior * models[0] + (1 - prior) * models[1]
                    info = entropy(mean) - prior * entropy(models[0]) - (1 - prior) * entropy(models[1])
                    j = np.where(available, info, -np.inf).argmax()
                else:
                    j = (random_order if strategy == "random" else freq_order)[t]
                available[j] = False
                asked.append(options[j].tolist())
                first = uniforms[j] < answer(options[j : j + 1])[0]
                answers.append(int(first))
                likelihood = models[:, j] if first else 1 - models[:, j]
                prior = prior * likelihood[0] / (prior * likelihood[0] + (1 - prior) * likelihood[1])
                if t + 1 not in (8, 24):
                    continue
                forecast = prior * forecasts[0] + (1 - prior) * forecasts[1]
                fingerprint = hashlib.sha256(forecast.tobytes()).hexdigest()
                actual = answer(fresh)
                error = np.average(np.abs(forecast - actual), weights=weight[held])
                rows.append(
                    dict(
                        split=k,
                        mixed=mix,
                        strategy=strategy,
                        budget=t + 1,
                        error=float(error),
                        asked=asked.copy(),
                        answers=answers.copy(),
                        posterior=float(prior),
                        forecast_hash=fingerprint,
                        held=fresh.tolist(),
                        predicted=forecast.tolist(),
                        actual=actual.tolist(),
                    )
                )
    save(folder / "pair_queries.jsonl.gz", rows)
    return job


def report():
    rows = []
    tests = []
    for game in ("connect4", "reversi4"):
        data = [read(OUT / game / f"block_{b:02}" / "pair_queries.jsonl.gz") for b in range(24)]
        for mix, budget in itertools.product((0, 1), (8, 24)):
            methods = {
                s: np.array(
                    [
                        np.mean(
                            [
                                r["error"]
                                for r in block
                                if (r["mixed"], r["budget"], r["strategy"]) == (mix, budget, s)
                            ]
                        )
                        for block in data
                    ]
                )
                for s in ("information", "random", "visitation")
            }
            for s, x in methods.items():
                row = dict(game=game, mixed=mix, budget=budget, strategy=s, **summary(x), blocks=x.tolist())
                if s != "information":
                    diff = paired(methods["information"] - x)
                    row["information_minus_strategy"] = diff
                    tests.append(diff)
                rows.append(row)
    holm(tests)
    save(OUT / "pair_queries.jsonl.gz", rows)


if __name__ == "__main__":
    with ProcessPoolExecutor(5) as pool:
        for done in pool.map(run, [(g, b) for b in range(24) for g in ("connect4", "reversi4")]):
            print(done, flush=True)
    report()
