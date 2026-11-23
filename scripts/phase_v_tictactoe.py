"""Recover matched policy, board and seat records for the common comparisons from Phase IV."""

from concurrent.futures import ProcessPoolExecutor
import numpy as np
from policy_value_isomorph.phase_iv_game import OUT as DATA, game, soft, seed
from policy_value_isomorph.phase_iv_metrics import read
from policy_value_isomorph.phase_v import OUT, save


def run(bid):
    b = game()
    folder = DATA / f"block_{bid:02}"
    target = OUT / "tictactoe" / f"block_{bid:02}"
    target.mkdir(exist_ok=True, parents=True)
    visits = read(folder / "visits.jsonl.gz")
    temperatures = read(folder / "soft.jsonl.gz")
    results = []
    duels = []
    seats = []
    with np.load(folder / "visits.npz") as arrays, np.load(folder / "source.npz") as original:
        fixed = np.zeros_like(b.uniform)
        fixed[b.dec, original["moves"]] = 1
        for mix, kind in enumerate(("fixed", "randomized")):
            src = 0.75 * fixed + 0.25 * b.uniform if mix else fixed
            for k in range(2):
                test = (b.split(bid, k) == 2) & ~b.ends
                testids = np.flatnonzero(test)
                forced = b.legal[test].sum(1) == 1
                source_paths = b.sampler(src).rollout_paths(16 if mix else 1, seed(bid, 50, k))
                source_steps = sum((depth >= 0).sum(1) for depth in source_paths)
                for f in range(2):
                    for family in ("V", "Q"):
                        prefix = f"{kind}_{k}_{f}_{family}_"
                        stored = {
                            r["method"]: r
                            for r in visits
                            if r["fit"] == f
                            and r["source"] == kind
                            and r["split"] == k
                            and r["network"] == family
                        }
                        source_score = b.dense(arrays[prefix + "frequency"].clip(0, 1))
                        m = np.zeros(b.n, int)
                        m[b.dec] = arrays[prefix + "direct_choice"]
                        direct = b.point(m)
                        paths = b.sampler(direct).rollout_paths(1, seed(bid, 50, k))
                        copy_steps = sum((depth >= 0).sum(1) for depth in paths)
                        reference = {
                            t: (direct if t < 0 else soft(b, source_score, t))
                            for t in (-1, 0, 0.05, 0.25, 1)
                        }
                        for tag, r in stored.items():
                            name = family + "_" + tag.replace("vnm", "VNM")
                            variants = [(-1, r)] + [
                                (a["temperature"], a)
                                for a in temperatures
                                if a["method"] == name
                                and a["fit"] == f
                                and a["source"] == kind
                                and a["split"] == k
                                and a["temperature"] is not None
                            ]
                            z = source_score if tag == "direct" else b.dense(arrays[prefix + tag])
                            for t, rec in variants:
                                if t < 0:
                                    move = np.zeros(b.n, int)
                                    move[b.dec] = arrays[prefix + tag + "_choice"]
                                    p = b.point(move)
                                else:
                                    p = soft(b, z, t)
                                info = dict(
                                    game="tictactoe",
                                    block=bid,
                                    mixed=mix,
                                    split=k,
                                    fit=f,
                                    method=name,
                                    temperature=t,
                                    selected=rec.get("selected", False),
                                )
                                match = (src[test] * p[test]).sum(1)
                                tv = np.abs(src[test] - p[test]).sum(1) / 2
                                n = 16 if mix and tag.endswith("_source") else 1
                                cost = source_steps if tag.endswith("_source") else copy_steps
                                steps = (
                                    float(b.dense(cost)[test].sum() / test.sum())
                                    if tag != "direct"
                                    else 0.0
                                )
                                calls = (
                                    steps - float(b.legal[test].sum() * n / test.sum())
                                    if tag.endswith("_source")
                                    else 0.0
                                )
                                strata = [
                                    dict(
                                        n=int(mask.sum()),
                                        agreement=float(match[mask].mean()) if mask.any() else None,
                                        tv=float(tv[mask].mean()) if mask.any() else None,
                                    )
                                    for mask in (~forced, forced, np.zeros(len(forced), bool))
                                ]
                                row = dict(
                                    **info,
                                    **{
                                        v: rec[v]
                                        for v in (
                                            "agreement",
                                            "tv",
                                            "visit_agreement",
                                            "visit_tv",
                                            "win",
                                            "draw",
                                            "loss",
                                            "points",
                                        )
                                    },
                                    boards=len(testids),
                                    strata=strata,
                                    simulated_moves=steps,
                                    source_calls=calls,
                                    move_pairs=None,
                                )
                                if tag != "direct":
                                    if t < 0 and not mix:
                                        row["move_pairs"] = r["move_pairs"]
                                    seat = (
                                        np.asarray(r["against_direct"])
                                        if t < 0
                                        else np.stack(
                                            [
                                                b.endings(p, reference[t])[0],
                                                b.endings(reference[t], p)[0, ::-1],
                                            ]
                                        )
                                    )
                                    for si, triple in enumerate(seat):
                                        duels.append(
                                            dict(
                                                **info,
                                                seat=si,
                                                win=triple[0],
                                                draw=triple[1],
                                                loss=triple[2],
                                            )
                                        )
                                results.append(row)
                                data = np.asarray(rec["seats"])
                                for si in (0, 1):
                                    for j, cols in enumerate(([0], list(range(3, 8)), [1], [2])):
                                        w, d, l = data[si, cols].mean(0)
                                        seats.append(
                                            dict(
                                                **info,
                                                seat=si,
                                                opponent=j,
                                                win=w,
                                                draw=d,
                                                loss=l,
                                                points=w - l,
                                            )
                                        )
    save(target / "metrics.jsonl.gz", results)
    save(target / "duels.jsonl.gz", duels)
    save(target / "seats.jsonl.gz", seats)
    return bid


if __name__ == "__main__":
    with ProcessPoolExecutor(6) as pool:
        for bid in pool.map(run, range(32)):
            print(bid, flush=True)
