from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from .gameplay_eval import GameplayEvalRecord
from .telemetry import TrainingTelemetry


def telemetry_run_dir(root_dir: str | Path, game: str, run_type: str, seed: int) -> Path:
    """Return a standardized output directory path for telemetry artifacts."""

    root = Path(root_dir)
    return root / game / run_type / f"seed_{seed}"


def append_telemetry_jsonl(path: str | Path, telemetry: TrainingTelemetry) -> None:
    """Append telemetry records to JSONL, writing one object per telemetry point.

    This writer is append-safe: repeated calls only append lines and never rewrite
    existing content.
    """

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a", encoding="utf-8") as f:
        for record in telemetry.records:
            row = {
                "run_type": telemetry.run_type,
                "game": telemetry.game,
                "seed": telemetry.seed,
                "checkpoint_step": telemetry.checkpoint_step,
                "hyperparameters": dict(telemetry.hyperparameters),
                "record": asdict(record),
            }
            f.write(json.dumps(row, sort_keys=True) + "\n")


def append_telemetry_csv(path: str | Path, telemetry: TrainingTelemetry) -> None:
    """Append telemetry records to CSV with a stable header.

    This writer is append-safe for resumed runs by only writing a header if the
    output file does not already exist.
    """

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_type",
        "game",
        "seed",
        "checkpoint_step",
        "step",
        "loss",
        "wall_time_seconds",
        "hyperparameters_json",
    ]

    write_header = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for record in telemetry.records:
            writer.writerow(
                {
                    "run_type": telemetry.run_type,
                    "game": telemetry.game,
                    "seed": telemetry.seed,
                    "checkpoint_step": telemetry.checkpoint_step,
                    "step": record.step,
                    "loss": record.loss,
                    "wall_time_seconds": record.wall_time_seconds,
                    "hyperparameters_json": json.dumps(dict(telemetry.hyperparameters), sort_keys=True),
                }
            )


def append_gameplay_eval_jsonl(path: str | Path, records: list[GameplayEvalRecord]) -> None:
    """Append gameplay evaluation records to JSONL.

    This format is machine-readable and append-safe for periodic checkpoints.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), sort_keys=True) + "\n")
