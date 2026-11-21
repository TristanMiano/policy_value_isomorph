"""Evaluate direct and complete-game selection on exactly the same visit-trained networks."""

import numpy as np

from .phase_iv_fit import candidates, designs, masks, selected
from .phase_iv_game import game, seed
from .phase_iv_metrics import Evaluation, corrected_counts, dump
from .phase_iv_source import actors
from .trajectory_utility import trajectory_scores


def run(block, dest, tmp):
    b = game()
    _, sources, opponents = actors(block, dest)
    results, curves, examples = [], [], []
    archive = {}
    with np.load(dest / "baseline.npz") as baseline:
        for kind, source in sources.items():
            ev = Evaluation(source, opponents)
            sampler = b.sampler(source)
            repetitions = 1 if kind == "fixed" else 16
            for k in range(2):
                split = b.split(block, k)
                test = split == 2
                ids = np.flatnonzero(test & ~b.ends)
                counts = baseline[f"{kind}_{k}_counts"]
                y, total = sampler.frequencies(counts)
                paths = sampler.rollout_paths(repetitions, seed(block, 50, k))
                for family, x in designs().items():
                    train, val, held = masks(split, family)
                    for fit in range(2):
                        init = seed(block, 51 if family == "V" else 52, k, fit)
                        label = dict(block=block, source=kind, split=k, fit=fit, network=family)
                        key = f"{kind}_{k}_{fit}_{family}"
                        fits = list(candidates(x, y, train, val, init))
                        best, settings = min(fits, key=lambda entry: entry[1]["validation_loss"])
                        for pred, setting in fits:
                            bounded = pred.clip(0, 1)
                            p = b.greedy(b.dense(bounded))
                            row = ev.measure(p, test, **label, method="candidate", **setting)
                            curves.append(
                                dict(
                                    **row,
                                    selected=setting == settings,
                                    test_loss=float(np.square(pred[held] - y[held]).mean()),
                                    test_exact_loss=float(
                                        np.square(pred[held] - source[b.legal][held]).mean()
                                    ),
                                    bounded_train_loss=float(np.square(bounded[train] - y[train]).mean()),
                                )
                            )
                        rng = np.random.default_rng(seed(block, 53, k))
                        half = train & (rng.random(b.n)[b.parent] < 0.5)
                        smaller, smallset = selected(x, y, half, val, init)
                        curves.append(
                            ev.measure(
                                b.greedy(b.dense(smaller.clip(0, 1))),
                                test,
                                **label,
                                method="half_training",
                                **smallset,
                                test_exact_loss=float(
                                    np.square(smaller[held] - source[b.legal][held]).mean()
                                ),
                            )
                        )
                        bounded = best.clip(0, 1)
                        direct = b.greedy(b.dense(bounded))
                        archive[key + "_frequency"] = best.astype("f4")
                        # Scores for all candidates are saved for independent inspection and interventions.
                        copied = b.sampler(direct).rollout_paths(1, seed(block, 50, k))
                        direct_rows = {"direct": (b.dense(bounded), direct)}
                        for who, roll in (("source", paths), ("copies", copied)):
                            for utility, (mean, _) in trajectory_scores(
                                roll, sampler.sign * bounded
                            ).items():
                                scores = b.dense(sampler.sign * mean)
                                name = utility + "_" + who
                                direct_rows[name] = scores, b.greedy(scores)
                                archive[key + "_" + name] = scores[b.legal].astype("f4")
                        for name, (scores, p) in direct_rows.items():
                            row = ev.measure(p, test, method=name, **label, **settings)
                            if kind == "fixed":
                                row["move_pairs"] = corrected_counts(source, direct, p, ids)
                            if name != "direct":
                                row["against_direct"] = np.stack(
                                    [b.endings(p, direct)[0], b.endings(direct, p)[0, ::-1]]
                                ).tolist()
                            results.append(row)
                            archive[key + "_" + name + "_choice"] = p[b.dec].argmax(1).astype("i1")
                        mismatched = ids[direct[ids].argmax(1) != source[ids].argmax(1)]
                        for i in mismatched[:2]:
                            edge = b.parent == i
                            examples.append(
                                dict(
                                    **label,
                                    board=b.names[i],
                                    withheld=True,
                                    moves=b.cell[edge].tolist(),
                                    visits=int(total[edge][0]),
                                    counts=counts[edge].tolist(),
                                    predicted=(bounded[edge] * total[edge]).tolist(),
                                    raw_frequency=best[edge].tolist(),
                                )
                            )
    dump(tmp / "visits.jsonl.gz", results)
    dump(tmp / "curves.jsonl.gz", curves)
    dump(tmp / "examples.jsonl.gz", examples)
    np.savez_compressed(tmp / "visits.npz", **archive)
