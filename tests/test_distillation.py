from random import Random
from math import log

import pytest

from policy_value_isomorph.distillation import fit_scores, observe
from policy_value_isomorph.policy_adapter import GameAdapter, TicTacToeState


def test_named_observations_use_a_common_evaluation():
    state = TicTacToeState.initial()
    source = GameAdapter("highest numbered cell", fixed=lambda s: max(s.legal_moves()),
                         law=lambda s: {max(s.legal_moves()): 1},
                         raw_scores=lambda s: {a: 0 if a==8 else -float("inf") for a in s.legal_moves()},
                         choose_menu=lambda s,m,r: max(m))
    feat = lambda s,a: [a/8]
    full = observe(source,state,feat,"probability",Random(0))
    for mode in ("action","probability","logit","pair"):
        example = observe(source,state,feat,mode,Random(0),pair=(0,8))
        fit = fit_scores([example],mode,epochs=15,seed=8)
        law = fit.evaluate([full])
        assert law["rows"] == law["mean_source_probability_of_selected_move"] == 1
        assert 0 <= fit.loss_nats < log(len(example.target))
    with pytest.raises(NotImplementedError):
        observe(GameAdapter("first",fixed=lambda s:0),state,feat,"pair",Random(),pair=(0,8))
