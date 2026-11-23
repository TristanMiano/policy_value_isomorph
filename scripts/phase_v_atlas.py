"""Generate an offline decision atlas for the first independent source of each new game."""

import json
from pathlib import Path
import numpy as np
from policy_value_isomorph.phase_v import OUT, actions, install_source, moved, native, seed, spec
from policy_value_isomorph.phase_v_study import scoring, setup_fit
from policy_value_isomorph.phase_v_evaluate import methods


def chars(s, rev):
    cells = range(16) if rev else [7 * c + r for r in reversed(range(6)) for c in range(7)]
    return "".join("X" if s[0] & (1 << i) else "O" if s[1] & (1 << i) else "." for i in cells)


def main():
    games = {}
    for game in ("connect4", "reversi4"):
        rev = game == "reversi4"
        e = native().Engine(rev)
        folder = OUT / game / "block_00"
        install_source(e, folder)
        for mix in (0, 1):
            e.configure(bool(mix), seed(game, 0, 50, 0))
            setup_fit(e, folder, 0, mix, 0)
            with np.load(folder / f"data_0_{mix}.npz") as data:
                ids = data["test"]
                boards = data["states"][ids].tolist()
                counts = data["counts"][ids]
            with (
                np.load(folder / f"evaluation_0_{mix}_0.npz") as a,
                np.load(folder / f"evaluation_0_{mix}_1.npz") as b,
            ):
                names = [key[6:] for key in a.files if key.startswith(("score_Q_", "score_V_"))]
                rules = dict(methods())
                scores = {n: [] for n in names}
                for fit, saved in enumerate((a, b)):
                    setup_fit(e, folder, 0, mix, fit)
                    for n in names:
                        z = scoring(e, boards, rules[n])
                        assert np.array_equal(z.argmax(1), saved["p_" + n + "_max"].argmax(1))
                        scores[n].append(z)
                setup_fit(e, folder, 0, mix, 0)
                choices = {
                    n: np.stack([a["p_" + n + "_max"].argmax(1), b["p_" + n + "_max"].argmax(1)])
                    for n in names
                }
                raw = scoring(e, boards, spec(0))
                rows = []
                for j, s in enumerate(boards):
                    legal = actions(e, s)
                    moves = []
                    for a1 in legal:
                        c = e.next(s, a1)
                        rolls = {}
                        for label, m in (("V copies", 1), ("Q copies", 2), ("source copies", 0)):
                            path = e.trajectory(c, spec(m), spec(m), seed(game, 0, 118, j, a1))
                            rolls[label] = dict(
                                moves=[a1] + [moved(x, y, rev) for x, y in zip(path, path[1:])],
                                points=e.outcome(path[-1]) * (1 - 2 * s[2]),
                            )
                        if e.legal(c):
                            neighbor = scoring(e, [c], spec(2))[0]
                            next_move = int(np.argmax(neighbor))
                            next_score = round(float(neighbor.max()), 4)
                        else:
                            next_move = next_score = None
                        moves.append(
                            dict(board=chars(c, rev), next=next_move, score=next_score, rollouts=rolls)
                        )
                    rows.append(
                        dict(
                            board=chars(s, rev),
                            side=s[2],
                            legal=legal,
                            counts=counts[j, legal].tolist(),
                            order=sorted(legal, key=lambda a: (-raw[j, a], a)),
                            moves=moves,
                            choices={n: v[:, j].tolist() for n, v in choices.items()},
                            rankings={
                                n: [sorted(legal, key=lambda a: (-z[j, a], a)) for z in fits]
                                for n, fits in scores.items()
                            },
                            scores={
                                n: [np.round(z[j, legal], 5).tolist() for z in fits]
                                for n, fits in scores.items()
                            },
                        )
                    )
                games[game + ("_randomized" if mix else "_fixed")] = rows
    template = Path(__file__).with_name("phase_v_atlas.html").read_text(encoding="utf8")
    (OUT / "atlas.html").write_text(
        template.replace("/*DATA*/", json.dumps(games, separators=(",", ":"))), encoding="utf8"
    )


if __name__ == "__main__":
    main()
