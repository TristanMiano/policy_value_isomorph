from __future__ import annotations

import json
from dataclasses import fields
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


def list_checkpoints(checkpoint_dir: str | Path, run_type: str) -> list[Path]:
    """Return sorted checkpoint paths for a run type."""
    run_dir = Path(checkpoint_dir) / run_type
    if not run_dir.exists():
        return []
    return sorted(run_dir.glob("step_*.json"))


def latest_checkpoint_path(checkpoint_dir: str | Path, run_type: str) -> Path:
    """Return latest checkpoint path for a run type."""
    checkpoints = list_checkpoints(checkpoint_dir, run_type)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found for run_type={run_type!r}")
    return checkpoints[-1]


def load_checkpoint(path: str | Path, *, model_type: type[Any]) -> dict[str, Any]:
    """Load a checkpoint JSON and materialize the dataclass model payload."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not is_dataclass(model_type):
        raise TypeError("model_type must be a dataclass type")

    model_payload = payload.get("model")
    if not isinstance(model_payload, dict):
        raise ValueError("checkpoint payload missing model dict")

    allowed = {f.name for f in fields(model_type)}
    model_kwargs = {k: v for k, v in model_payload.items() if k in allowed}
    payload["model"] = model_type(**model_kwargs)
    return payload
