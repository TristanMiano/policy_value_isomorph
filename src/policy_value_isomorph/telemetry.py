from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class TelemetryRecord:
    """One training telemetry point.

    `step` is 1-indexed epoch/iteration count.
    `wall_time_seconds` is elapsed wall-clock since train loop start.
    """

    step: int
    loss: float
    wall_time_seconds: float


@dataclass(frozen=True)
class TrainingTelemetry:
    """Standardized telemetry schema for policy/value/Q training runs."""

    run_type: str
    game: str
    seed: int
    checkpoint_step: int
    hyperparameters: Mapping[str, float | int]
    records: tuple[TelemetryRecord, ...]
