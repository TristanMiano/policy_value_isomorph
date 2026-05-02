from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from policy_value_isomorph.cli import main as cli_main


def main(output_dir: str = "artifacts/demo_monitoring") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    telemetry_root = out / "telemetry"
    policy_ckpt = out / "checkpoints" / "policy"
    value_ckpt = out / "checkpoints" / "value"
    gameplay_samples = out / "gameplay_samples.txt"

    cli_main(
        [
            "train",
            "--episodes",
            "16",
            "--policy-epochs",
            "4",
            "--value-epochs",
            "4",
            "--rollouts",
            "1",
            "--seed",
            "13",
            "--eval-interval",
            "2",
            "--eval-games",
            "6",
            "--policy-checkpoint-interval",
            "2",
            "--policy-checkpoint-dir",
            str(policy_ckpt),
            "--value-checkpoint-interval",
            "2",
            "--value-checkpoint-dir",
            str(value_ckpt),
            "--gameplay-samples-path",
            str(gameplay_samples),
            "--gameplay-samples-per-checkpoint",
            "1",
            "--telemetry-root-dir",
            str(telemetry_root),
        ]
    )

    policy_run_dir = telemetry_root / "tic_tac_toe" / "policy" / "seed_13"
    cli_main(
        [
            "plots",
            "--telemetry-csv",
            str(policy_run_dir / "training_telemetry.csv"),
            "--gameplay-jsonl",
            str(policy_run_dir / "gameplay_eval.jsonl"),
            "--output-dir",
            str(out / "plots"),
        ]
    )

    print(f"demo_output_dir={out}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "artifacts/demo_monitoring"
    main(target)
