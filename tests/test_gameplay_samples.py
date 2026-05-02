from policy_value_isomorph.connect_four import ConnectFourState
from policy_value_isomorph.gameplay_samples import GameplaySample, append_gameplay_samples_text, format_gameplay_sample
from policy_value_isomorph.policy_mlp import train_policy_mlp
from policy_value_isomorph.sampling import generate_on_policy_dataset
from policy_value_isomorph.tictactoe import TicTacToeState


def test_format_gameplay_sample_renders_tictactoe_and_connect_four() -> None:
    t0 = TicTacToeState.initial()
    t1 = t0.apply_move(0)
    rendered_ttt = format_gameplay_sample(GameplaySample(checkpoint_step=2, game_index=0, moves=(0,), states=(t0, t1)))
    assert "checkpoint_step=2" in rendered_ttt
    assert "X . ." in rendered_ttt

    c0 = ConnectFourState.initial()
    c1 = c0.apply_move(3)
    rendered_c4 = format_gameplay_sample(GameplaySample(checkpoint_step=2, game_index=1, moves=(3,), states=(c0, c1)))
    assert "0 1 2 3 4 5 6" in rendered_c4
    assert "(to move: O)" in rendered_c4


def test_train_policy_mlp_writes_checkpoint_gameplay_samples(tmp_path) -> None:
    dataset = generate_on_policy_dataset(lambda s: min(s.legal_moves()), n_episodes=8)
    out_path = tmp_path / "gameplay_samples.txt"
    train_policy_mlp(
        dataset,
        hidden_dim=8,
        epochs=4,
        seed=2,
        eval_interval=2,
        eval_n_games=4,
        gameplay_samples_path=str(out_path),
        gameplay_samples_per_checkpoint=2,
    )
    text = out_path.read_text(encoding="utf-8")
    assert "checkpoint_step=2" in text
    assert "checkpoint_step=4" in text
    assert "state[0]:" in text
    assert "moves=" in text


def test_append_gameplay_samples_text_appends() -> None:
    sample = GameplaySample(
        checkpoint_step=1,
        game_index=0,
        moves=(0,),
        states=(TicTacToeState.initial(),),
    )
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as d:
        path = f"{d}/samples.txt"
        append_gameplay_samples_text(path, [sample])
        append_gameplay_samples_text(path, [sample])
        text = open(path, "r", encoding="utf-8").read()
        assert text.count("checkpoint_step=1") == 2
