from math import inf
from types import SimpleNamespace

import pytest

from policy_value_isomorph.comparison_suite import (choice_metrics,ranking,geometry,calibration,
    changed_choices,return_checkpoints,GameMethod)
from policy_value_isomorph.score_alignment import align
from policy_value_isomorph.recovery_contract import RecoveryContract,ValueArtifact
from policy_value_isomorph.tictactoe import TicTacToeState
from policy_value_isomorph.checkpointing import save_checkpoint
from policy_value_isomorph.value_mlp import TinyMLPValue


def test_structure_and_changed_choices():
    row=choice_metrics({0:1},{1:1},(0,1),{0:2,1:2})
    assert (row["tv"],row["cross_entropy"],row["ties"],row["agreement"])==(1,inf,2,0)
    reversed_priority = choice_metrics({1:1},{1:1},(0,1),{0:2,1:2},tie_order=(1,0))
    assert reversed_priority["source_fixed_match"] == 1
    assert choice_metrics({0:1,1:0},{0:1},(0,1),{0:0,1:-inf})["source_score_forgone"] == 0
    assert ranking({0:2,1:1},{0:0,1:1},1)["spearman"]==-1
    r=lambda x:align({0:x,1:0},"raw",units="points",reference="first player")
    assert geometry({0:r(1),1:r(3)},{0:r(2),1:r(3)},[(0,1)])=={"move_pairs":1,"mse":1}
    assert calibration([(0,1),(1,1)],(0,1))[0]["mse"]==0.5
    assert changed_choices({"menu":[({0:1},{1:1})],"lottery":[]})["equally_weighted_observed_families"]==1


def test_saved_weights_recompute_both_errors_and_full_games(tmp_path):
    # A tree containing the single legal continuation used by all three evaluated players.
    state = TicTacToeState.initial()
    children, laws, scores = {},{},{}
    while not state.is_terminal():
        move=state.legal_moves()[0]
        child=state.apply_move(move)
        children[state],laws[state],scores[state]={move:child},{move:1},{move:1}
        state=child
    children[state]={}
    graph=SimpleNamespace(children=children)
    c=RecoveryContract("tic-tac-toe","first legal","behavior","observed","weights","one path",
        "first legal","direct maximum","lowest","raw","points","test path","single path",
        game=True,opponent_panel="first legal only")
    art=ValueArtifact(c,"source","constructed tree","exact",{"rule":"first legal"})
    source=GameMethod(art,laws,scores)
    paths=[save_checkpoint(tmp_path,run_type="value",step=i,model=TinyMLPValue([[0]], [0], [0], b),metadata={})
           for i,b in [(1,0),(2,1)]]
    root=TicTacToeState.initial()
    def method(m):
        a=ValueArtifact(c,"recovered","saved weights","exact",{"bias":m.b2})
        return GameMethod(a,laws,{s:{a:m.b2 for a in row} for s,row in scores.items()})
    output=list(return_checkpoints(paths,TinyMLPValue,lambda m,k:m.b2,[(root,0)],[(root,1)],
        method,graph,source,{"empty":{root:1}},{"first legal":(1,laws)}))
    assert [r["return_error"]["training"]["mse"] for r in output]==[0,1]
    assert [r["return_error"]["heldout_exact"]["mse"] for r in output]==[1,0]
    assert output[1]["decisions"][-1]["gap"] is None
    assert output[1]["game_outcomes"][-1]["delta_win"]==0
