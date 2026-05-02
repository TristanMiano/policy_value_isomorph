from policy_value_isomorph.policy_mlp import train_policy_mlp
from policy_value_isomorph.sampling import generate_off_policy_dataset


def test_periodic_gameplay_eval_emits_records() -> None:
    dataset = generate_off_policy_dataset(n_episodes=12, seed=5)
    trained = train_policy_mlp(dataset, hidden_dim=12, epochs=6, seed=5, eval_interval=2, eval_n_games=6)

    evals = trained.training_log.gameplay_evals
    assert [r.step for r in evals] == [2, 4, 6]
    assert all(r.n_games == 6 for r in evals)
    assert all(abs((r.win_rate + r.draw_rate + r.loss_rate) - 1.0) < 1e-9 for r in evals)
