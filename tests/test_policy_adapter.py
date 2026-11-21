from random import Random

import pytest

from policy_value_isomorph.policy_adapter import ConnectFourState, GameAdapter, TicTacToeState


@pytest.mark.parametrize("game,width", [(TicTacToeState, 9), (ConnectFourState, 7)])
def test_game_queries_and_continuation(game, width):
    state = game.initial()
    first = GameAdapter("first legal", fixed=lambda s: s.legal_moves()[0])
    last = GameAdapter("last offered", law=lambda s: {s.legal_moves()[-1]: 1},
                       choose_menu=lambda s, menu, rng: menu[-1])
    assert first.candidates(state) == tuple(range(width))
    assert first.encode(state)[-1] == 1 and first.sample(state, Random(1)) == 0
    assert last.constrained(state, (0,2), Random(3)) == 2
    assert "constrained" not in first.capabilities
    with pytest.raises(NotImplementedError):
        first.constrained(state, (0,2), Random())
    path = first.rollout(state, 1, Random(2), last)
    assert path[2] == state.apply_move(1).apply_move(width-1)
    assert path[-1].is_terminal() and not first.candidates(path[-1])
    with pytest.raises(ValueError):
        first.action(path[-1])
    with pytest.raises(ValueError):
        last.constrained(state, (0,0), Random())
