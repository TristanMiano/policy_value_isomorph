from dataclasses import replace

import pytest

from policy_value_isomorph.policy_adapter import GameAdapter
from policy_value_isomorph.recovery_contract import RecoveryContract,ValueArtifact
from policy_value_isomorph.tictactoe import state_from_rows
from policy_value_isomorph.value_atlas import AtlasRecovery,ValueAtlas


def test_atlas_keeps_source_mass_gaps_neighborhoods_and_between_fit_spread():
    s=state_from_rows(["XX.","OO.","XO."],1)
    p=GameAdapter("lowest legal",fixed=lambda s:min(s.legal_moves()))
    c=RecoveryContract("tic-tac-toe","lowest","behavior","choices","scores","fixed board","legal",
        "direct maximum","lowest","fixed units","points","worked example","one position",
        game=True,opponent_panel="uniform legal")
    def recovery(a):
        artifact=ValueArtifact(c,"table","chosen move","constructed",{"chosen":a})
        return AtlasRecovery(artifact,lambda s:{i:int(i==a) for i in s.legal_moves()},lambda s:{a:1})
    first, second=recovery(2),recovery(5)
    atlas=ValueAtlas(p,lambda s:{2:1},[first,second])
    report=atlas.inspect(s,visits={2:8,5:0,8:0})
    assert len(report["neighboring_states"])==3
    assert report["actions"][0]["expected_deployed_count_at_same_visits"]==4
    assert report["actions"][0]["predicted_visit_count"] is None
    assert report["actions"][0]["probability_sd"]==pytest.approx(2**-0.5)
    assert [d["agreement"] for d in report["per_fit_decisions"]]==[1,0]
    altered=replace(second,artifact=replace(second.artifact,contract=replace(c,units="ranks")))
    with pytest.raises(ValueError):
        ValueAtlas(p,atlas.source_law,[first,altered]).inspect(s)
