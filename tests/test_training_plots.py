from policy_value_isomorph.policy_mlp import train_policy_mlp
from policy_value_isomorph.sampling import generate_off_policy_dataset
from policy_value_isomorph.telemetry_io import append_telemetry_csv, telemetry_run_dir
from policy_value_isomorph.training_plots import read_training_curve_from_csv, write_loss_plots_from_csv


def test_generate_loss_plots_from_csv(tmp_path) -> None:
    dataset = generate_off_policy_dataset(n_episodes=12, seed=13)
    trained = train_policy_mlp(dataset, hidden_dim=10, epochs=4, seed=13)

    run_dir = telemetry_run_dir(tmp_path, game="tic_tac_toe", run_type="policy", seed=13)
    csv_path = run_dir / "training.csv"
    append_telemetry_csv(csv_path, trained.training_log.telemetry)

    points = read_training_curve_from_csv(csv_path, x_field="step")
    assert len(points) == 4
    assert points[0].x == 1.0

    step_svg, time_svg = write_loss_plots_from_csv(csv_path, run_dir)
    step_text = step_svg.read_text(encoding="utf-8")
    time_text = time_svg.read_text(encoding="utf-8")

    assert "Training loss by step" in step_text
    assert "Training loss by wall-clock time" in time_text
    assert "<polyline" in step_text
