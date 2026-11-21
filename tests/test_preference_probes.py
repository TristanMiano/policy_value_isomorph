from policy_value_isomorph.policy_adapter import GameAdapter
from policy_value_isomorph.preference_probes import probe,pair_model
from policy_value_isomorph.tictactoe import state_from_rows


def test_observe_fresh_menus_and_diagnose_cycle():
    board = state_from_rows(["XX.","OO.","XO."],1)
    player = GameAdapter("largest offered except full menu",fixed=lambda s: min(s.legal_moves()),
                         choose_menu=lambda s,m,r: min(m) if len(m)>2 else max(m))
    order = probe(player,board,"example",strategy="elimination",budget=9,seed=4)
    assert len(order.history)==2 and len(order.order_diagnostics()["ordered_pairs"])==3
    rec = probe(player,board,"example",strategy="active pairs",budget=7,seed=4,
                check_menus=[board.legal_moves()])
    assert rec.order_diagnostics()["conflicting_pairs"]
    model = pair_model(rec)
    assert sum(p["draws"] for p in model["pairs"])==6
    assert all(p["model_probability_a"]<0.5 for p in model["pairs"])
    assert all(p["interval"] is not None for p in model["pairs"])
