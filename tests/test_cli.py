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



def test_cli_plots(tmp_path, capsys) -> None:
    csv_path = tmp_path / "telemetry.csv"
    csv_path.write_text(
        "run_type,game,seed,checkpoint_step,step,loss,wall_time_seconds,hyperparameters_json\n"
        "policy,tic_tac_toe,0,2,1,1.0,0.1,{}\n"
        "policy,tic_tac_toe,0,2,2,0.5,0.2,{}\n",
        encoding="utf-8",
    )
    main(["plots", "--telemetry-csv", str(csv_path), "--output-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "loss_step_plot=" in out
    assert (tmp_path / "loss_by_step.svg").exists()
    assert (tmp_path / "loss_by_wall_time.svg").exists()
