"""Recover ordered best-outcome moves from visits, testing both conventions for missing comparisons."""

from concurrent.futures import ProcessPoolExecutor
import numpy as np
from policy_value_isomorph.agreement_diagnosis import order_from_observed_ties
from policy_value_isomorph.phase_iv_source import actors
from policy_value_isomorph.phase_iv_metrics import Evaluation
from policy_value_isomorph.phase_iv_game import game as graph, OUT as OLD
from policy_value_isomorph.phase_v import (
    SIZE,
    OUT,
    native,
    install_source,
    opening_panel,
    panel,
    spec,
    save,
)
from policy_value_isomorph.phase_v_study import scoring
from diagnose_agreement import DEST, record, outcomes


def run(job):
    g, b = job
    folder = DEST / g / f"block_{b:02}"
    records = []
    if g == "reversi4":
        origin = OUT / g / f"block_{b:02}"
        e = native().Engine(True)
        install_source(e, origin)
        states = [tuple(s) for s in e.all() if e.legal(s)]
        lookup = {s: i for i, s in enumerate(states)}
        scores = scoring(e, states, spec(0))
        legal = scores > -1e28
        truth = scores.argmax(1)
        guarantee = e.rule_inputs(states)[:, :, 0]
    else:
        game = graph()
        origin = OLD / f"block_{b:02}"
        _, laws, opponents = actors(b, origin)
        evalgame = Evaluation(laws["fixed"], opponents)
        legal = game.legal[game.dec]
        truth = laws["fixed"][game.dec].argmax(1)
        guarantee = game.optimal[game.dec]
    for k in (0, 1):
        if g == "reversi4":
            with np.load(origin / f"data_{k}_0.npz") as d:
                active = (d["partition"] == 0) & (d["counts"].sum(1) > 0)
                counts = d["counts"][active]
                tr = [lookup[tuple(s)] for s in d["states"][active]]
                primary = d["equal"] > 0
                ids = [lookup[tuple(s)] for s in d["states"][d["test"][primary]]]
                w = d["equal"][primary]
        else:
            with np.load(origin / "baseline.npz") as d:
                split = game.split(b, k)[game.dec]
                tr = np.flatnonzero(split == 0)
                ids = np.flatnonzero(split == 2)
                counts = game.dense(d[f"fixed_{k}_counts"])[game.dec][tr]
                w = np.ones(len(ids))
        for reverse in (False, True):
            pref, edges = order_from_observed_ties(guarantee[tr], legal[tr], counts, reverse)
            n = len(pref)
            z = np.where(legal, 2 * guarantee + pref / (n + 1), -1e30)
            decisions = z.argmax(1)
            for f in (0, 1):
                rr = record(
                    records,
                    {},
                    "ordered_reverse" if reverse else "ordered_first",
                    z[ids],
                    legal[ids],
                    truth[ids],
                    w,
                )
                rr.update(
                    split=k,
                    fit=f,
                    all_count=int((decisions == truth).sum()),
                    all_boards=len(truth),
                    observed_relations=int(edges.sum()),
                    priority=pref.tolist(),
                )
                if g == "reversi4":
                    weights = np.zeros(SIZE)
                    weights[:n] = pref / (n + 1)
                    weights[n] = 2
                    e.set(16, True, 0, weights)
                    outcomes(rr, np.array(panel(e, spec(16), opening_panel(e, g, b, k, f)))[:, 0])
                else:
                    m = np.zeros(game.n, int)
                    m[game.dec] = decisions
                    seat = evalgame.outcomes(game.point(m)).mean(0)[:-1].T @ evalgame.weights
                    rr.update(win=seat[0], draw=seat[1], loss=seat[2], points=seat[0] - seat[2])
    save(folder / "ordered.jsonl.gz", records)
    return job


if __name__ == "__main__":
    with ProcessPoolExecutor(5) as pool:
        for job in pool.map(
            run, [(g, b) for g in ("tictactoe", "reversi4") for b in range(32 if g == "tictactoe" else 24)]
        ):
            print(job, flush=True)
