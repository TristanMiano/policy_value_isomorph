"""Summarize controlled comparisons after averaging fits inside independent original players."""

from collections import defaultdict
import json
import numpy as np
from policy_value_isomorph.phase_iv_statistics import compare, describe, holm
from policy_value_isomorph.phase_iv_game import game as finite
from policy_value_isomorph.phase_v import OUT, native, install_source, spec, read, save
from policy_value_isomorph.phase_v_study import scoring
from policy_value_isomorph.agreement_diagnosis import choice_probabilities
from diagnose_agreement import DEST, GAMES


def summarize():
    summaries = []
    effects = []
    examples = {}
    for game in GAMES:
        n = 32 if game == "tictactoe" else 24
        cells = defaultdict(list)
        if game == "tictactoe":
            ttt = finite()
            ids = ttt.dec
            mask = ttt.legal[ids]
            ctx = ttt.optimal[ids, :, None]
        elif game == "reversi4":
            eng = native().Engine(True)
            states = [s for s in eng.all() if eng.legal(s)]
            mask = np.array([[bool(eng.legal(s) & (1 << i)) for i in range(17)] for s in states])
            ctx = eng.rule_inputs(states)
        for b in range(n):
            folder = DEST / game / f"block_{b:02}"
            rows = read(folder / "diagnosis.jsonl.gz")
            if game != "connect4":
                rows += read(folder / "ordered.jsonl.gz")
            if game != "tictactoe":
                rows += read(folder / "choice_fit.jsonl.gz")
            per = defaultdict(list)
            for r in rows:
                for key, val in r.items():
                    if isinstance(val, (float, int)) and key not in ("fit", "split"):
                        per[r["method"], key].append(val)
            if game == "tictactoe":
                from policy_value_isomorph.phase_iv_game import OUT as TTT_OUT

                with np.load(TTT_OUT / f"block_{b:02}" / "source.npz") as d:
                    alltruth = d["moves"]
            elif game == "reversi4":
                install_source(eng, OUT / game / f"block_{b:02}")
                alltruth = scoring(eng, states, spec(0)).argmax(1)
            for k in (0, 1):
                if game != "tictactoe":
                    base = OUT / game / f"block_{b:02}"
                    with (
                        np.load(base / f"data_{k}_0.npz") as obs,
                        np.load(folder / f"extra_{k}.npz") as more,
                    ):
                        for label, data, part in (
                            ("original_collection", obs, "partition"),
                            ("expanded_collection", more, "training_partition"),
                        ):
                            count = data["counts"]
                            train = (data[part] == 0) & (count.sum(1) > 0)
                            per[label, "boards"].append(int(train.sum()))
                            per[label, "visits"].append(int(count[train].sum()))
                for f in (0, 1):
                    if game != "tictactoe":
                        for net in ("V", "Q"):
                            old = next(
                                r["selected"]
                                for r in read(base / f"training_{k}_0_{f}.jsonl.gz")
                                if r["model"] == (1 if net == "V" else 2)
                            )
                            per[net + "_direct", "training"].append(old["training"])
                    with np.load(folder / f"boards_{k}_{f}.npz") as a:
                        w = a["weights"]
                        truth = a["truth"]
                        legal = a["legal"]
                        ii = np.arange(len(w))
                        if game != "connect4":
                            pm = choice_probabilities(a["rule_inputs_weights"], mask, ctx)
                            per["rule_inputs", "all_count"].append(int((pm.argmax(1) == alltruth).sum()))
                            per["rule_inputs", "all_boards"].append(len(alltruth))
                        else:
                            with np.load(folder / f"normalized_original_{k}_{f}.npz") as chk:
                                teacher = chk["teacher"]
                            structured = a["rule_inputs"].argmax(1)
                            for label, indicator in (
                                ("teacher_agreement", teacher == truth),
                                (
                                    "copied_teacher_original_differs",
                                    (structured == teacher) & (structured != truth),
                                ),
                            ):
                                per["rule_inputs", label].append(np.average(indicator, weights=w))
                        for net in ("Q", "V"):
                            direct = a[net + "_direct"]
                            vnm = a[net + "_terminal"]
                            d = direct.argmax(1)
                            v = vnm.argmax(1)
                            sign = a[net + "_endpoints"][0, :, :, 1]
                            loss = (d == truth) & (v != truth)
                            for label, bucket in (
                                ("lost_changed_mover", sign[ii, d] != sign[ii, v]),
                                (
                                    "lost_same_mover_strict",
                                    (sign[ii, d] == sign[ii, v]) & (vnm[ii, d] < vnm[ii, v]),
                                ),
                                (
                                    "lost_same_mover_tie",
                                    (sign[ii, d] == sign[ii, v]) & (vnm[ii, d] == vnm[ii, v]),
                                ),
                            ):
                                per[net + "_switch", label].append(np.average(loss & bucket, weights=w))
                            mag = a[net + "_endpoints"][0, :, :, 3]
                            per[net + "_unsigned_terminal", "match"].append(
                                np.average(np.where(legal, mag, -np.inf).argmax(1) == truth, weights=w)
                            )
                            for label, primary, secondary in (
                                ("direct_ties", vnm, direct),
                                (
                                    "exact_direct_ties",
                                    a[net + "_exact_score_source_continuation"],
                                    a["exact_direct"],
                                ),
                            ):
                                best = legal & (primary == primary.max(1, keepdims=True))
                                choice = np.where(best, secondary, -np.inf).argmax(1)
                                per[net + "_" + label, "match"].append(
                                    np.average(choice == truth, weights=w)
                                )
                        if game == "tictactoe" and game not in examples:
                            wrong = (a["Q_direct"].argmax(1) != truth) & (
                                a["rule_inputs"].argmax(1) == truth
                            )
                            j = int(np.flatnonzero(wrong)[0])
                            visits = int(a["counts"][j].sum())
                            m = a["legal"][j]
                            examples[game] = dict(
                                block=b,
                                split=k,
                                fit=f,
                                state=str(a["states"][j]),
                                visits=visits,
                                source=int(truth[j]),
                                moves=np.flatnonzero(m).tolist(),
                                observed=a["counts"][j, m].tolist(),
                                predicted=(visits * a["Q_direct"][j, m]).tolist(),
                                rule_inputs=(visits * a["rule_inputs"][j, m]).tolist(),
                            )
                        if game != "tictactoe" and game not in examples:
                            wrong = a["Q_direct"].argmax(1) != truth
                            recovered = a["Q_expanded"].argmax(1) == truth
                            jj = np.flatnonzero(wrong & recovered & (a["counts"].sum(1) > 0))
                            if len(jj):
                                j = int(jj[0])
                                visits = int(a["counts"][j].sum())
                                menu = legal[j]
                                examples[game] = dict(
                                    block=b,
                                    split=k,
                                    fit=f,
                                    state=a["states"][j].tolist(),
                                    visits=visits,
                                    source=int(truth[j]),
                                    moves=np.flatnonzero(menu).tolist(),
                                    observed=a["counts"][j, menu].tolist(),
                                    predicted=(visits * a["Q_direct"][j, menu]).tolist(),
                                    expanded=(visits * a["Q_expanded"][j, menu]).tolist(),
                                )
                        if np.any(legal.sum(1) > 1):
                            sw = w * (legal.sum(1) > 1)
                            for name in (
                                "V_direct",
                                "Q_direct",
                                "Q_terminal",
                                "Q_exact_score_source_continuation",
                                "rule_inputs",
                            ):
                                per[name, "multi_match"].append(
                                    np.average(a[name].argmax(1) == truth, weights=sw)
                                )
            for key, values in per.items():
                cells[key].append(float(np.mean(values)))
        for (name, m), x in cells.items():
            assert len(x) == n
            summaries.append(dict(game=game, method=name, metric=m, **describe(x), blocks=x))
        pairs = [
            ("rule_inputs", "action_only"),
            ("rule_inputs", "Q_direct"),
            ("rule_inputs_centered", "rule_inputs"),
            ("ordered_first", "ordered_reverse"),
            ("ordered_first", "rule_inputs"),
        ]
        for net in ("V", "Q"):
            base = net + "_direct"
            pairs.extend(
                [
                    (net + "_" + key, base)
                    for key in ("raw", "terminal", "centered", "centered_source", "expanded", "half")
                ]
            )
            pairs.extend(
                [("exact_direct", base), ("projection_V", base)] if net == "V" else [("exact_direct", base)]
            )
            pairs.extend(
                [
                    (net + "_exact_score_source_continuation", "exact_direct"),
                    (net + "_exact_score_fitted_continuation", net + "_terminal"),
                    (net + "_fitted_score_source_continuation", net + "_terminal"),
                    (net + "_exact_score_source_continuation", net + "_exact_score_fitted_continuation"),
                    (net + "_exact_score_source_continuation", net + "_fitted_score_source_continuation"),
                    (net + "_centered", net + "_terminal"),
                    (net + "_unsigned_terminal", net + "_terminal"),
                ]
            )
            pairs.extend(
                [
                    (net + "_direct_ties", net + "_terminal"),
                    (net + "_exact_direct_ties", net + "_exact_score_source_continuation"),
                ]
            )
        pairs.extend(
            [
                ("Q_normalized_original", "Q_direct"),
                ("Q_normalized_expanded", "Q_expanded"),
                ("Q_normalized_expanded", "Q_normalized_original"),
                ("Q_normalized_expanded", "Q_direct"),
            ]
        )
        for name, ref in pairs:
            for m in ("match", "win", "draw", "loss", "points", "multi_match"):
                if (name, m) not in cells or (ref, m) not in cells:
                    continue
                x = np.array(cells[name, m]) - cells[ref, m]
                effects.append(
                    dict(
                        game=game,
                        method=name,
                        reference=ref,
                        metric=m,
                        **compare(x),
                        block_differences=x.tolist(),
                    )
                )
    holm(effects)
    if "connect4" not in examples:
        # Some primary sampled boards have zero visits. Find a withheld, actually
        # visited parent for an illustrative count comparison at a common denominator.
        from policy_value_isomorph.phase_v_study import setup_fit

        e = native().Engine(False)
        for b in range(24):
            origin = OUT / "connect4" / f"block_{b:02}"
            folder = DEST / "connect4" / f"block_{b:02}"
            install_source(e, origin)
            setup_fit(e, origin, 0, 0, 0)
            with np.load(origin / "data_0_0.npz") as d:
                h = (d["partition"] == 2) & (d["counts"].sum(1) > 0)
                states = d["states"][h].tolist()
                counts = d["counts"][h]
            old = scoring(e, states, spec(2))
            with np.load(folder / "expanded_weights_0_0.npz") as fit:
                e.set(2, True, 0, fit["Q"])
            new = scoring(e, states, spec(2))
            truth = counts.argmax(1)
            eligible = np.flatnonzero((old.argmax(1) != truth) & (new.argmax(1) == truth))
            if len(eligible):
                j = int(eligible[0])
                m = old[j] > -1e28
                n = int(counts[j].sum())
                examples["connect4"] = dict(
                    block=b,
                    split=0,
                    fit=0,
                    state=states[j],
                    visits=n,
                    source=int(truth[j]),
                    moves=np.flatnonzero(m).tolist(),
                    observed=counts[j, m].tolist(),
                    predicted=(n * old[j, m]).tolist(),
                    expanded=(n * new[j, m]).tolist(),
                )
                break
    save(DEST / "results.jsonl.gz", summaries)
    save(DEST / "contrasts.jsonl.gz", effects)
    (DEST / "count_examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    return summaries, effects


def render(summary, effects):
    lookup = {(r["game"], r["method"], r["metric"]): r for r in summary}
    names = dict(
        exact_direct="Exact original frequency, direct (privileged)",
        projection_V="Mean exact incoming frequency, direct V (privileged)",
        rule_inputs="Action preferences plus game consequences",
        action_only="Action preferences alone",
        rule_inputs_centered="Game consequences, signed local shortfalls VNM",
        Q_normalized_original="Q normalized fit, original visits",
        Q_normalized_expanded="Q normalized fit, added visits",
        Q_normalized_expanded_centered="Q normalized fit, added visits, signed local shortfalls VNM",
        Q_normalized_original_centered="Q normalized fit, original visits, signed local shortfalls VNM",
    )
    for net in ("Q", "V"):
        for suffix, title in (
            ("direct", "direct"),
            ("raw", "direct, before bounding scores"),
            ("terminal", "final-move VNM, fitted continuations"),
            (
                "fitted_score_source_continuation",
                "final-move VNM, fitted scores and original continuations",
            ),
            (
                "exact_score_fitted_continuation",
                "final-move VNM, exact frequencies and fitted continuations",
            ),
            (
                "exact_score_source_continuation",
                "final-move VNM, exact frequencies and original continuations",
            ),
            ("centered", "signed local shortfalls VNM, fitted continuations"),
            ("centered_source", "signed local shortfalls VNM, original continuations"),
            ("expanded", "direct, added visits"),
            ("half", "direct, half training boards"),
            ("unsigned_terminal", "final frequency with last mover's sign removed"),
        ):
            names[net + "_" + suffix] = net + " " + title
    games = dict(tictactoe="Tic-tac-toe", connect4="Connect Four", reversi4="4×4 Reversi")
    names.update(
        ordered_first="Game outcome plus elicited ordering, low-index fallback",
        ordered_reverse="Game outcome plus elicited ordering, high-index fallback",
    )
    for net in ("V", "Q"):
        names[net + "_direct_ties"] = net + " final-move VNM, direct breaks ties"
        names[net + "_exact_direct_ties"] = (
            net + " exact-frequency final-move VNM, exact direct breaks ties (privileged)"
        )

    def display(r, scale=100):
        if r is None:
            return "—"
        return f"{scale * r['mean']:.2f} ± {scale * r['sd']:.2f}"

    st = [
        "# Matched measurements of move recovery",
        "",
        "The original is the frozen neural player. V scores only the resulting board. Q also sees the current board and move. Direct selection takes the highest immediate score. Von Neumann–Morgenstern (VNM) selection compares expected scores of complete plays. The final-move rule uses the last mover’s signed frequency. A local shortfall is the played score minus the mover’s best available score. The alternative VNM utility sums these differences, positive sign for the first player's terms and negative for the second. Both seats use the specified continuing player. Exact frequencies give the original’s move a 1 and every alternative a 0. For each comparison using exact frequencies, the V or Q prefix identifies the original fit being controlled. Each game uses two board splits and two independently initialized fits per original. Values show mean ± sample standard deviation after averaging fits per original. A dash denotes a measurement outside that intervention. [Design](DESIGN.md) · [Report and method definitions](../../AGREEMENT_DIAGNOSIS.md)",
        "",
        "| Game | Originals | Method | Same move (%) | Strictly beaten favorite (%) | Favorite lost a tie (%) | Wins (%) | Draws (%) | Losses (%) | Points/game |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g, game in games.items():
        for name, title in names.items():
            rr = lookup.get((g, name, "match"))
            if not rr:
                continue
            vals = [
                display(lookup.get((g, name, m)), 1 if m == "points" else 100)
                for m in ("match", "strict", "lost_tie", "win", "draw", "loss", "points")
            ]
            st.append("| " + " | ".join([game, str(rr["n"]), title, *vals]) + " |")
    st += [
        "",
        "## Paired changes",
        "",
        "Each difference uses the same original and board or seat cases. Rate differences are percentage points. The 95% intervals describe uncertainty across independent original blocks. Raw p is the two-sided probability of a result at least as extreme under zero mean difference. Adjusted p uses Holm’s procedure over this diagnostic family. Exact identities have zero spread and an undefined Student p-value. The machine-readable data also retain every individual difference.",
        "",
        "| Game | Originals | Method minus comparator | Measurement | Difference ± spread | 95% interval | Raw p | Adjusted p |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]

    def prob(p):
        return "—" if p is None else f"{p:.3g}"

    for r in effects:
        scale = 1 if r["metric"] == "points" else 100
        st.append(
            f"| {games[r['game']]} | {r['n']} | {names[r['method']]} minus {names[r['reference']]} | {r['metric']} | {display(r, scale)} | [{scale * r['ci95'][0]:.2f}, {scale * r['ci95'][1]:.2f}] | {prob(r['p'])} | {prob(r['holm_p'])} |"
        )
    (DEST / "tables.md").write_text("\n".join(st) + "\n", encoding="utf8")


if __name__ == "__main__":
    render(*summarize())
