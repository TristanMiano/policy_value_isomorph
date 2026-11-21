"""Training checkpoints, cross-fit ordering and geometry, and full-information utility references."""

import numpy as np
from .phase_iv_fit import candidates, designs, masks
from .phase_iv_game import game, seed
from .phase_iv_metrics import Evaluation, dump
from .phase_iv_source import actors


def integrated(b, frequency, law):
    """Enumerate continuations of the supplied law; store suffix utility in X's units."""
    signed = frequency * b.side[:, None]
    for name in ("terminal", "sum", "decay"):
        value = np.zeros(b.n)
        score = np.zeros_like(frequency)
        for d in range(8, -1, -1):
            ids = b.layers[d]
            child = b.child[ids]
            if name == "terminal":
                z = np.where(b.ends[child], signed[ids], value[child])
            else:
                z = signed[ids] + (0.9 if name == "decay" else 1) * value[child]
            value[ids] = (law[ids] * z).sum(1)
            score[ids] = b.side[ids, None] * z
        yield name, score


def run(block, dest, tmp):
    b = game()
    _, sources, opponents = actors(block, dest)
    training, stability, oracles = [], [], []
    with np.load(dest / "visits.npz") as fitdata, np.load(dest / "baseline.npz") as data:
        for kind, law in sources.items():
            sampler = b.sampler(law)
            ev = Evaluation(law, opponents)
            for k in range(2):
                split = b.split(block, k)
                ids = np.flatnonzero((split == 2) & ~b.ends)
                y, _ = sampler.frequencies(data[f"{kind}_{k}_counts"])
                for family, x in designs().items():
                    train, val, test = masks(split, family)
                    key = dict(block=block, source=kind, split=k, network=family)
                    if family == "V":
                        child = np.bincount(b.after, law[b.legal], minlength=b.n) / np.bincount(
                            b.after, minlength=b.n
                        ).clip(1)
                        exact = b.dense(child[b.after])
                    else:
                        exact = law
                    for cont, actor in (("copies", b.greedy(exact)), ("source", law)):
                        for u, z in integrated(b, exact, actor):
                            oracles.append(
                                ev.measure(
                                    b.greedy(np.round(z, 14)),
                                    split == 2,
                                    method="oracle_VNM_" + u + "_" + cont,
                                    fit=-1,
                                    **key,
                                )
                            )
                    scores = []
                    for fit in range(2):
                        z = b.dense(fitdata[f"{kind}_{k}_{fit}_{family}_frequency"].clip(0, 1))
                        scores.append(z)
                        init = seed(block, 51 if family == "V" else 52, k, fit)
                        for pred, setting in candidates(
                            x, y, train, val, init, penalties=(1.0,), steps=(1, 16, 256, None)
                        ):
                            training.append(
                                ev.measure(
                                    b.greedy(b.dense(pred.clip(0, 1))),
                                    split == 2,
                                    method="gradient_steps",
                                    fit=fit,
                                    **key,
                                    **setting,
                                    test_exact_loss=float(
                                        np.square(pred[test] - law[b.legal][test]).mean()
                                    ),
                                )
                            )
                    left, right = np.triu_indices(9, 1)
                    legal = b.legal[ids][:, left] & b.legal[ids][:, right]
                    ranks = [np.sign(z[ids][:, left] - z[ids][:, right])[legal] for z in scores]
                    gaps = []
                    for z in scores:
                        order = np.sort(np.where(b.legal, z, -2), axis=1)
                        gaps.append(order[:, -1] - order[:, -2])
                    edges = (
                        (split[b.parent] == 2)
                        & (b.legal.sum(1)[b.parent] > 1)
                        & (b.legal.sum(1)[b.after] > 1)
                    )
                    diff = gaps[1] - gaps[0]
                    stability.append(
                        dict(
                            **key,
                            legal_pairs=int(legal.sum()),
                            reversed=int(np.sum(ranks[0] * ranks[1] < 0)),
                            changed_ties=int(np.sum((ranks[0] == 0) != (ranks[1] == 0))),
                            neighbors=int(edges.sum()),
                            frequency_rms=float(
                                np.sqrt(np.square((scores[1] - scores[0])[b.legal][test]).mean())
                            ),
                            neighbor_gap_rms=float(
                                np.sqrt(np.square(diff[b.parent[edges]] - diff[b.after[edges]]).mean())
                            ),
                        )
                    )
    dump(tmp / "training_steps.jsonl.gz", training)
    dump(tmp / "stability.jsonl.gz", stability)
    dump(tmp / "oracles.jsonl.gz", oracles)
