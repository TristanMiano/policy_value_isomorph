from math import dist

from policy_value_isomorph.candidate_sets import legal_candidates,reachable,sample_continuous,sequence_candidates
from policy_value_isomorph.policy_adapter import GameAdapter,TicTacToeState


def test_coverage_retains_bias_and_history():
    s = TicTacToeState.initial()
    player = GameAdapter("first",fixed=lambda s:s.legal_moves()[0])
    assert len(reachable(player,s,depth=2))==9+72
    subset = legal_candidates(player,s,limit=3,seed=8)
    assert len(subset.values)==3 and subset.coverage({subset.values[0]:0.25,99:0.75})["omitted_probability"]==0.75
    box = sample_continuous([(-1,1),(2,2)],draws=5,seed=8)
    assert box==sample_continuous([(-1,1),(2,2)],draws=5,seed=8)
    assert box.closest_distance((0,2),dist)==min(abs(x) for x,y in box.values)
    tokens = sequence_candidates(("word",),lambda prefix,r:prefix[-1],draws=7,seed=2,generator="repeat last token")
    assert tokens.values==("word",) and tokens.proposals==7
