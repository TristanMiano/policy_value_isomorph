from policy_value_isomorph.cli import main


def test_cli_data_on(capsys) -> None:
    main(["data", "--mode", "on", "--episodes", "2"])
    out = capsys.readouterr().out
    assert "dataset_mode=on" in out
    assert "samples=" in out


def test_cli_train(capsys) -> None:
    main(["train", "--episodes", "8", "--policy-epochs", "2", "--value-epochs", "2", "--rollouts", "1"])
    out = capsys.readouterr().out
    assert "policy_loss_final=" in out
    assert "value_loss_final=" in out


def test_cli_eval(capsys) -> None:
    main(["eval", "--episodes", "8", "--epochs", "2", "--games", "4", "--rollouts", "1"])
    out = capsys.readouterr().out
    assert "action_agreement=" in out
    assert "wdl=" in out
