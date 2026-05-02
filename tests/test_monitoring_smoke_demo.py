from pathlib import Path
import subprocess
import sys


def test_monitoring_checkpoint_demo_script(tmp_path: Path) -> None:
    script = Path("scripts/demo_monitoring_checkpointing.py")
    out_dir = tmp_path / "demo"
    subprocess.run([sys.executable, str(script), str(out_dir)], check=True)

    policy_seed_dir = out_dir / "telemetry" / "tic_tac_toe" / "policy" / "seed_13"
    value_seed_dir = out_dir / "telemetry" / "tic_tac_toe" / "value" / "seed_13"

    assert (policy_seed_dir / "training_telemetry.csv").exists()
    assert (policy_seed_dir / "training_telemetry.jsonl").exists()
    assert (policy_seed_dir / "gameplay_eval.jsonl").exists()
    assert (value_seed_dir / "training_telemetry.csv").exists()
    assert (out_dir / "checkpoints" / "policy" / "policy" / "step_000002.json").exists()
    assert (out_dir / "checkpoints" / "value" / "value" / "step_000004.json").exists()
    assert (out_dir / "plots" / "loss_by_step.svg").exists()
    assert (out_dir / "plots" / "gameplay_score_by_step.svg").exists()
    assert (out_dir / "gameplay_samples.txt").exists()
