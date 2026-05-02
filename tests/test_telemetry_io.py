import json

from policy_value_isomorph.policy_mlp import train_policy_mlp
from policy_value_isomorph.sampling import generate_off_policy_dataset
from policy_value_isomorph.telemetry_io import (
    append_telemetry_csv,
    append_telemetry_jsonl,
    telemetry_run_dir,
)


def test_telemetry_jsonl_and_csv_append_safe(tmp_path) -> None:
    dataset = generate_off_policy_dataset(n_episodes=12, seed=11)
    trained = train_policy_mlp(dataset, hidden_dim=8, epochs=3, seed=11)

    out_dir = telemetry_run_dir(tmp_path, game="tic_tac_toe", run_type="policy", seed=11)
    jsonl_path = out_dir / "training.jsonl"
    csv_path = out_dir / "training.csv"

    append_telemetry_jsonl(jsonl_path, trained.training_log.telemetry)
    append_telemetry_jsonl(jsonl_path, trained.training_log.telemetry)

    jsonl_lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl_lines) == 6
    first = json.loads(jsonl_lines[0])
    assert first["game"] == "tic_tac_toe"
    assert first["run_type"] == "policy"
    assert first["record"]["step"] == 1

    append_telemetry_csv(csv_path, trained.training_log.telemetry)
    append_telemetry_csv(csv_path, trained.training_log.telemetry)

    csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert csv_lines[0].startswith("run_type,game,seed,checkpoint_step")
    assert len(csv_lines) == 1 + 6


def test_telemetry_run_dir_is_consistent() -> None:
    path = telemetry_run_dir("artifacts", game="connect_four", run_type="value", seed=3)
    assert str(path).endswith("artifacts/connect_four/value/seed_3")
