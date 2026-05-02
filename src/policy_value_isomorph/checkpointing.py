from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def checkpoint_path(checkpoint_dir: str | Path, run_type: str, step: int) -> Path:
    return Path(checkpoint_dir) / run_type / f"step_{step:06d}.json"


def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    run_type: str,
    step: int,
    model: Any,
    metadata: dict[str, Any],
) -> Path:
    """Save model weights and training metadata to JSON.

    This is intentionally lightweight for pure-Python dataclass models.
    """

    if step <= 0:
        raise ValueError("step must be >= 1")

    out_path = checkpoint_path(checkpoint_dir, run_type, step)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if is_dataclass(model):
        model_payload = asdict(model)
    else:
        raise TypeError("model must be a dataclass instance")

    payload = {
        "run_type": run_type,
        "step": step,
        "model": model_payload,
        "metadata": metadata,
    }
    out_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return out_path
