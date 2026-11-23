"""Check the frozen selection, full-play expectation, and mirror-game denominators."""

import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from policy_value_isomorph.win_selection import FinitePlanner, Selection, standard_vnm

DATA = Path(__file__).resolve().parents[1] / "experiments" / "win_selection"


def test_default_is_the_discovery_choice_and_reuses_no_confirmation_sources():
    lock = json.loads((DATA / "standard.json").read_text())
    assert standard_vnm().name == lock["winner"]
    assert standard_vnm("Q").network == "Q"
    for game, count in (("tictactoe", 32), ("connect4", 24), ("reversi4", 24)):

        def ids(stage):
            return {
                int(p.name.split("_")[1].split(".")[0]) for p in (DATA / stage / game).glob("block_*.json.gz")
            }

        assert ids("discovery") == ids("tuning") == set(range(8))
        assert ids("confirmation") == set(range(8, count))
        with gzip.open(DATA / "confirmation" / game / "block_08.json.gz", "rt") as f:
            rows = [r for r in json.load(f)["rows"] if r["method"] == lock["winner"]]
        assert len(rows) == 4
        for r in rows:
            win, draw, loss = r["self_wdl"]
            assert np.allclose((win, loss), ((1 - draw) / 2,) * 2)
            assert np.isclose(sum(r["wdl"]), 1)
            if game != "tictactoe":
                assert len(r["external"]) == 64 and len(r["self_first"]) == 32


def test_finite_vnm_agrees_with_explicit_complete_paths_in_both_seats():
    from policy_value_isomorph.phase_iv_game import game

    b = game()
    scores = np.random.default_rng(77).uniform(size=b.legal.shape)
    planner = FinitePlanner(b, scores)
    menu = planner.menu("guarded_mixture")
    for depth in (5, 6):
        root = b.layers[depth][13]
        seat = b.side[root]

        def complete(state, opponent):
            if b.ends[state]:
                return np.array(
                    [0.0, 0.0, 0.0, float(seat * b.terminal[state] > 0), float(b.terminal[state] == 0)]
                )
            law = planner.guarded if b.side[state] == seat else opponent
            return sum(law[state, a] * force(state, a, opponent) for a in np.flatnonzero(law[state]))

        def force(state, action, opponent):
            child = b.child[state, action]
            v = complete(child, opponent)
            score = seat * b.side[state] * scores[state, action]
            v[0] = score if b.ends[child] else v[0]
            v[1] += score
            v[2] = score + 0.9 * v[2]
            return v

        for a in np.flatnonzero(b.legal[root]):
            expected = np.mean(
                [force(root, a, op) for op in (planner.direct, planner.guarded, planner.position, b.uniform)],
                axis=0,
            )
            np.testing.assert_allclose(menu[root, a], expected, atol=1e-13)
        offered = b.legal.copy()
        offered[root, planner.law()[root].argmax()] = False
        win, draw, decay = menu[root, :, 3], menu[root, :, 4], menu[root, :, 2]
        best_remaining = np.where(offered[root], win + 0.25 * draw + 0.02 * decay, -np.inf).argmax()
        assert planner.law(offered=offered)[root].argmax() == best_remaining
    np.testing.assert_array_equal(planner.law(), planner.law(standard_vnm()))
    with pytest.raises(ValueError, match="original"):
        planner.law(Selection("V", "original", "hybrid"))


@pytest.mark.skipif(
    sys.platform == "win32", reason="native Phase V simulator requires the Linux/macOS toolchain"
)
def test_native_expected_selection_reproduces_previous_payoff_and_frequency_conventions():
    from policy_value_isomorph.phase_v import SIZE, native, prefixes, spec
    from policy_value_isomorph.win_selection import board_game_player

    rng = np.random.default_rng(912)
    for rev in (True, False):
        engine = native().Engine(rev)
        with pytest.raises(ValueError, match="frozen visit-frequency weights"):
            board_game_player(engine, 43)
        engine.set(1, False, 0, rng.uniform(0, 0.1, SIZE))
        states = prefixes(engine, 32, rng)
        for rule, code in (("last", 1), ("discounted", 3), ("points", 4)):
            choose = board_game_player(engine, 43, Selection("V", "copies", rule))
            expected = engine.profiles(states, spec(1, code), False)[:, :-2].argmax(1)
            np.testing.assert_array_equal([choose(s) for s in states], expected)
        direct = native().WinPlanner(engine, 1, 43, 16)
        args = Selection("V", "copies", "direct").native_args()
        scores = engine.profiles(states, spec(1), False)[:, :-2]
        for state, row in zip(states, scores):
            legal = engine.legal(state)
            if legal.bit_count() > 1:
                remaining = legal ^ (1 << int(row.argmax()))
                options = [a for a in range(len(row)) if remaining & (1 << a)]
                assert direct.choose(state, *args, remaining) == max(options, key=row.__getitem__)
        choose = board_game_player(engine, 43)
        assert engine.legal(engine.start()) & (1 << choose(engine.start()))
        if rev:
            # At these endgames all four projected opponents have the same forced tail.
            # Compute the default utility independently, including passes and second-seat wins.
            checked = [0, 0]
            for root in engine.all():
                seat = root[2]
                moves = engine.legal(root)
                if checked[seat] == 10 or moves.bit_count() < 2:
                    continue
                if (root[0] | root[1]).bit_count() < 13:
                    continue
                utilities = {}
                for first in range(16):
                    if not moves & (1 << first):
                        continue
                    state, a, signed_weight, total = root, first, 0.02, 0.0
                    while True:
                        scores = engine.profiles([state], spec(1), False)[0]
                        total += signed_weight * scores[a]
                        state = engine.next(state, a)
                        legal = engine.legal(state)
                        if not legal:
                            outcome = engine.outcome(state) * (1 - 2 * seat)
                            utilities[first] = total + (outcome > 0) + 0.25 * (outcome == 0)
                            break
                        if legal.bit_count() > 1:
                            break
                        a, signed_weight = legal.bit_length() - 1, -0.9 * signed_weight
                if len(utilities) == moves.bit_count():
                    assert choose(root) == max(utilities, key=utilities.get)
                    remaining = moves ^ (1 << choose(root))
                    other = [a for a in utilities if remaining & (1 << a)]
                    assert direct.choose(root, *standard_vnm().native_args(), remaining) == max(
                        other, key=utilities.get
                    )
                    checked[seat] += 1
                    if checked == [10, 10]:
                        break
            assert checked == [10, 10]
