from fractions import Fraction as F
from random import Random

from policy_value_isomorph.consequence_utility import (ExactLotteries, Utility, UtilityObservation,
    expectation, fit_utility, sample_lottery, utility_move)
from policy_value_isomorph.policy_adapter import GameAdapter
from policy_value_isomorph.tictactoe import state_from_rows
from policy_value_isomorph.pilot_protocol import uniform_policy


def test_reusable_lotteries_exact_rollout_and_conditional_tail():
    state = state_from_rows(["XX.","OO.","X.."], -1)
    exact = ExactLotteries(uniform_policy,uniform_policy)
    actor = GameAdapter("uniform legal",law=uniform_policy)
    for move in state.legal_moves():
        lottery = exact.force(state,move)
        boot = sample_lottery(state,move,actor,actor,draws=3,rng=Random(4),steps=1,suffix=exact.suffix)
        assert lottery == boot.distribution
        for u in [Utility(), Utility(F(1,2),F(-1,4))]:
            assert boot.estimate(u) == (float(expectation(lottery,u)),0)
    sims = sample_lottery(state,5,actor,actor,draws=16,rng=Random(3))
    assert sims.estimate(Utility()) == (-1,0)


def test_outcome_means_match_existing_exact_return_evaluator():
    from policy_value_isomorph.pilot_exact import evaluate_source
    from policy_value_isomorph.pilot_protocol import GameGraph, State

    graph = GameGraph.build({"salt":"small reference","train_end":65,"validation_end":82})
    laws = {s:uniform_policy(s) for s in graph.states if graph.children[s]}
    reference = evaluate_source(graph,laws)
    evaluator = ExactLotteries(uniform_policy,uniform_policy)
    starts = [State.initial(),state_from_rows(["XX.","OO.","X.."],-1)]
    for s in starts:
        for a in s.legal_moves():
            assert expectation(evaluator.force(s,a),Utility()) == reference.action_value[s][a]


def test_known_shared_utility_ambiguity_and_incompatible_choice():
    draw, gamble = {(0,1): F(1)}, {(-1,1): F(1,2),(1,1): F(1,2)}
    laws = {2:gamble,5:draw}
    known = Utility(F(1,2))
    obs = UtilityObservation(laws,utility_move(laws,known,1),1)
    fit = fit_utility([obs],[-1,0,F(1,2),1])
    assert [u.draw for u in fit.best] == [F(1,2),1]
    impossible = UtilityObservation({2:draw,5:draw},5,1)
    fit = fit_utility([obs,impossible],[-1,0,F(1,2),1])
    assert fit.decisions == 2 and max(n for _,n in fit.matches)==1
    durations = {2:{(0,2):F(1)},5:{(0,1):F(1)}}
    assert utility_move(durations,Utility(per_move=F(-1,4)),1) == 5
