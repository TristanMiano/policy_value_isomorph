"""Review archived tic-tac-toe decisions under equal-board and self-play weights."""

import json
from pathlib import Path

import numpy as np

from policy_value_isomorph.phase_iv_game import OUT, game, soft
from policy_value_isomorph.phase_iv_metrics import read
from policy_value_isomorph.phase_iv_statistics import block_means, compare, describe, holm


def measures(original: np.ndarray, prediction: np.ndarray, reach: np.ndarray) -> dict:
    match = (original * prediction).sum(axis=1)
    distance = np.abs(original - prediction).sum(axis=1) / 2
    return {
        "agreement": float(match.mean()),
        "visit_agreement": float(np.average(match, weights=reach)),
        "tv": float(distance.mean()),
        "visit_tv": float(np.average(distance, weights=reach)),
    }


def review_block(block: int, folder: Path) -> tuple[list, list, dict]:
    b = game()
    rows, populations = [], []
    records = {}
    for file in ("baseline", "visits", "soft"):
        for row in read(folder / (file + ".jsonl.gz")):
            name = row.get("network", "") + row["method"]
            if file == "soft":
                if row["temperature"] == 0.05 and name == "Q_direct":
                    name += "_T0.05"
                elif row["temperature"] != 0.25:
                    continue
            records[row["source"], row["split"], row["fit"], name] = row
    with (
        np.load(folder / "source.npz") as frozen,
        np.load(folder / "baseline.npz") as base,
        np.load(folder / "visits.npz") as visit,
        np.load(folder / "sampling.npz") as draws,
        np.load(folder / "returns.npz") as returns,
    ):
        logits = np.zeros_like(b.uniform)
        logits[b.dec] = frozen["scores"]
        preferred = b.greedy(logits)
        certificate = json.loads((folder / "source.json").read_text())
        order = np.array(certificate["priority"])
        best_in_order = np.where(b.legal[:, order], b.optimal[:, order], -np.inf).argmax(axis=1)
        np.testing.assert_array_equal(order[best_in_order[b.dec]], preferred[b.dec].argmax(axis=1))
        ranks = returns["strict_ranks"][b.after]
        ranked = b.greedy(b.dense(ranks))
        np.testing.assert_array_equal(ranked[b.dec], preferred[b.dec])
        construction = dict(
            block=block,
            optimal_decisions=len(b.dec),
            reproduced_choices=len(b.dec),
            distinct_resulting_board_scores=len(np.unique(ranks)),
        )
        second = b.greedy(np.where(preferred > 0, -np.inf, logits))
        restricted = 0.75 * preferred + 0.25 * second
        forced = b.legal.sum(axis=1) == 1
        restricted[forced] = preferred[forced]
        laws = {
            "fixed": preferred,
            "randomized": 0.75 * preferred + 0.25 * b.uniform,
            "top_two": restricted,
        }
        for kind, law in laws.items():
            reach = b.visitation(law)
            for k in range(2):
                test = base[f"partition_{k}"][b.dec] == 2
                ids = b.dec[test]
                weights = reach[ids]
                populations.append(
                    dict(
                        block=block,
                        split=k,
                        source=kind,
                        withheld=len(ids),
                        positive=int((weights > 0).sum()),
                        expected_visits=float(weights.sum()),
                    )
                )
                if kind == "top_two":
                    for budget in (1, 64):
                        p = draws[f"top_two_{k}_{budget}"] / budget
                        rows.append(
                            dict(
                                block=block,
                                source=kind,
                                method=f"counts_{budget}",
                                **measures(law[ids], p, weights),
                            )
                        )
                    continue
                for fit in range(2):
                    key = f"{kind}_{k}_{fit}_"
                    imitate = np.zeros_like(logits)
                    imitate[b.dec] = base[key + "imitation"]
                    players = {
                        "imitation": (b.greedy(imitate) if kind == "fixed" else soft(b, imitate, 1))[ids],
                        "source": law[ids],
                    }
                    for net in ("V", "Q"):
                        for method in ("direct", "vnm_terminal_copies", "vnm_decay_copies"):
                            moves = visit[key + net + "_" + method + "_choice"][test]
                            players[net + method] = np.eye(9)[moves]
                    if kind == "randomized":
                        for name, suffix in (
                            ("Q_direct", "Q_frequency"),
                            ("Q_VNM_decay_copies", "Q_vnm_decay_copies"),
                        ):
                            scores = visit[key + suffix]
                            if name == "Q_direct":
                                scores = scores.clip(0, 1)
                                players["Q_direct_T0.05"] = soft(b, b.dense(scores), 0.05)[ids]
                            players[name] = soft(b, b.dense(scores), 0.25)[ids]
                    for name, p in players.items():
                        record = records[kind, k, -1 if name == "source" else fit, name]
                        metrics = measures(law[ids], p, weights)
                        seats = np.asarray(record["seats"])
                        group_weights = [0.25, 0.25, 0.25, *([0.05] * 5)]
                        outcomes = np.average(seats.mean(axis=0)[:8], axis=0, weights=group_weights)
                        metrics.update(dict(zip(("win", "draw", "loss"), map(float, outcomes))))
                        metrics["points"] = metrics["win"] - metrics["loss"]
                        np.testing.assert_allclose(
                            list(metrics.values()), [record[m] for m in metrics], rtol=0, atol=2e-12
                        )
                        rows.append(dict(block=block, source=kind, method=name, **metrics))
    return rows, populations, construction


def main() -> None:
    rows, populations, constructions = [], [], []
    for block in range(32):
        measured, groups, construction = review_block(block, OUT / f"block_{block:02}")
        rows.extend(measured)
        populations.extend(groups)
        constructions.append(construction)
    metrics = ["agreement", "visit_agreement", "tv", "visit_tv", "win", "draw", "loss", "points"]
    cells = block_means(rows, ["source", "method"], metrics)
    summary = {
        "/".join(key): {m: describe(list(values.values())) for m, values in cell.items()}
        for key, cell in cells.items()
    }
    # Changes in both populations, plus the game-strength and changed-menu decisions.
    contrasts = []
    for kind, method, ref, fields in (
        ("fixed", "Qvnm_decay_copies", "Qdirect", ["agreement", "visit_agreement"]),
        ("fixed", "imitation", "Qvnm_decay_copies", ["agreement", "visit_agreement"]),
        ("fixed", "Vvnm_decay_copies", "Vdirect", ["points", "win"]),
        ("randomized", "Q_VNM_decay_copies", "Q_direct", ["tv", "visit_tv"]),
        ("randomized", "Q_VNM_decay_copies", "imitation", ["tv", "visit_tv"]),
        ("randomized", "Q_direct_T0.05", "imitation", ["tv", "visit_tv"]),
        ("top_two", "counts_64", "counts_1", ["tv", "visit_tv"]),
    ):
        for m in fields:
            differences = [cells[kind, method][m][b] - cells[kind, ref][m][b] for b in range(32)]
            contrasts.append(
                dict(
                    source=kind,
                    method=method,
                    reference=ref,
                    metric=m,
                    block_differences=differences,
                    **compare(differences),
                )
            )
    changed = []
    for block in range(32):
        relevant = [
            r
            for r in read(OUT / f"block_{block:02}" / "interventions.jsonl.gz")
            if r["source"] == "fixed"
            and r["event"] in ("remove_favorite", "remove_other")
            and r["method"] in ("Q_direct", "imitation")
        ]
        errors = {
            name: np.mean([r["tv"] for r in relevant if r["method"] == name])
            for name in ("Q_direct", "imitation")
        }
        changed.append(float(errors["imitation"] - errors["Q_direct"]))
    contrasts.append(
        dict(
            source="fixed",
            method="imitation",
            reference="Q_direct",
            metric="removed_move_error",
            block_differences=changed,
            **compare(changed),
        )
    )
    holm(contrasts)
    result = {
        "definitions": "See CHECKPOINT_E.md. Rates are fractions. Game points = win minus loss. "
        "visit_ weights by the original playing both sides. Contrasts are method minus reference. "
        "One paired difference per original, equal weights within and between originals. Holm family "
        "contains all 15 exploratory contrasts displayed in this file. V and Q score the resulting "
        "board and the board plus move, respectively. direct takes their highest immediate score. "
        "vnm_terminal_copies and vnm_decay_copies maximize signed last-move and discounted cumulative "
        "frequency with both sides playing direct. Q_direct and Q_VNM_decay_copies sample those scores "
        "at temperature 0.25, except Q_direct_T0.05. counts_n uses n original draws per board.",
        "populations": populations,
        "constructions": constructions,
        "methods": summary,
        "contrasts": contrasts,
    }
    (OUT.parent / "checkpoint_e.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
