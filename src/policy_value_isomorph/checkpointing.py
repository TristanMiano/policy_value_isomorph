from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import re
import tempfile
import time
from dataclasses import fields
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

from .telemetry import TelemetryRecord
from .torch_mlp import TorchMLP, TrainingBatch, train_epoch
from .training_config import TrainingConfig


def checkpoint_path(checkpoint_dir: str | Path, run_type: str, step: int) -> Path:
    return Path(checkpoint_dir) / run_type / f"step_{step:06d}.pt"


def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    run_type: str,
    step: int,
    model: Any,
    metadata: dict[str, Any],
) -> Path:
    """Export the original JSON format for callers that store just the public weights."""

    if step <= 0:
        raise ValueError("step must be >= 1")

    out_path = checkpoint_path(checkpoint_dir, run_type, step).with_suffix(".json")
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
    paths = [p for p in run_dir.iterdir() if re.fullmatch(r"step_[0-9]+\.(json|pt)", p.name) and p.is_file()]
    return sorted(paths, key=lambda p: (int(p.stem[5:]), p.suffix == ".pt"))


def latest_checkpoint_path(checkpoint_dir: str | Path, run_type: str) -> Path:
    """Return latest checkpoint path for a run type."""
    checkpoints = list_checkpoints(checkpoint_dir, run_type)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found for run_type={run_type!r}")
    return checkpoints[-1]


def load_checkpoint(path: str | Path, *, model_type: type[Any]) -> dict[str, Any]:
    """Read native or historical weights and expose their public dataclass representation."""
    path = Path(path)
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != 2:
            raise ValueError("unsupported PyTorch checkpoint version")
        payload["model"] = {k: v.tolist() for k, v in payload["model_state"].items()}
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version", 1) != 1:
            raise ValueError("unsupported JSON checkpoint version")
    if not is_dataclass(model_type):
        raise TypeError("model_type must be a dataclass type")

    model_payload = payload.get("model")
    if not isinstance(model_payload, dict):
        raise ValueError("checkpoint payload missing model dict")

    allowed = {f.name for f in fields(model_type)}
    model_kwargs = {k: v for k, v in model_payload.items() if k in allowed}
    payload["model"] = model_type(**model_kwargs)
    return payload



def _batch_digest(batch: TrainingBatch) -> str:
    """Hash ordered inputs, labels and legal masks in their actual training precision."""
    digest = hashlib.sha256()
    for tensor in (batch.inputs, batch.targets, batch.legal):
        if tensor is not None:
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode("ascii"))
            raw = tensor.detach().cpu().contiguous().view(torch.uint8).flatten()
            for chunk in raw.split(65536):
                digest.update(bytes(chunk.tolist()))
    return digest.hexdigest()


class TrainingRun:
    """Own a fit's mutable state and publish a complete snapshot at an epoch boundary."""

    def __init__(self, kind: str, initial: Any, rng: random.Random, size: int, seed: int,
                 learning_rate: float | None, config: TrainingConfig | None, directory: str | Path | None,
                 resume_path: str | Path | None, resume_latest: bool):
        if resume_path is not None and resume_latest:
            raise ValueError("choose one resume path or latest")
        if resume_latest:
            if directory is None:
                raise ValueError("checkpoint_dir is required for resume_latest")
            resume_path = latest_checkpoint_path(directory, kind)
        saved = load_checkpoint(resume_path, model_type=type(initial)) if resume_path is not None else {}
        if saved and saved["run_type"] != kind:
            raise ValueError(f"expected a {kind} checkpoint, got {saved['run_type']}")
        meta = saved.get("metadata", {})
        self.version = saved.get("schema_version", 1)
        if self.version == 2:
            settings = dict(meta["config"])
            if config is not None:
                settings["device"] = config.device
            archived_config = TrainingConfig(**settings)
            if config is not None and config.metadata() != archived_config.metadata():
                raise ValueError("configuration differs from the saved training run")
            if learning_rate is not None and learning_rate != meta["learning_rate"]:
                raise ValueError("learning rate differs from the saved training run")
            config = archived_config
        self.config = config or TrainingConfig()
        if saved and self.version == 1 and self.config.optimizer != "sgd":
            raise ValueError("legacy JSON has no adaptive optimizer state; resume it with SGD")
        self.rate = learning_rate if learning_rate is not None else meta.get("learning_rate", 0.05)
        self.seed = meta.get("seed", seed)
        self.kind = kind
        self.directory = Path(directory) if directory is not None else None
        self.start = saved.get("step", 0)
        self.elapsed = meta.get("elapsed_seconds", 0.0)
        self.rng = rng
        self.order = meta.get("order", list(range(size)))
        if sorted(self.order) != list(range(size)):
            raise ValueError("checkpoint sample order and current dataset differ")
        if "rng_state" in meta:
            state = meta["rng_state"]
            rng.setstate(ast.literal_eval(state) if isinstance(state, str) else state)
        self.model = saved.get("model", initial)
        self.network = TorchMLP(self.model, torch.device(self.config.device), getattr(torch, self.config.dtype))
        self.optimizer = self.config.make_optimizer(self.network, self.rate)
        if self.version == 2:
            self.optimizer.load_state_dict(saved["optimizer_state"])
        self.cpu_rng = saved.get("cpu_rng")
        self.cuda_rng = saved.get("cuda_rng")
        self.digest = meta.get("batch_sha256")

    def _save(self, step: int, seconds: float) -> None:
        destination = checkpoint_path(self.directory, self.kind, step)
        destination.parent.mkdir(parents=True, exist_ok=True)
        device = self.network.w1.device
        payload = dict(schema_version=2, run_type=self.kind, step=step,
                       model_state=self.network.state_dict(), optimizer_state=self.optimizer.state_dict(),
                       cpu_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                       metadata=dict(seed=self.seed, epoch=step, learning_rate=self.rate, config=self.config.metadata(),
                                     order=self.order, rng_state=self.rng.getstate(), elapsed_seconds=seconds,
                                     batch_sha256=self.digest))
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
            staged = Path(tmp.name)
        try:
            torch.save(payload, staged)
            os.replace(staged, destination)
        finally:
            staged.unlink(missing_ok=True)

    def records(self, batch: TrainingBatch, epochs: int, interval: int) -> Iterator[TelemetryRecord]:
        """Yield just the new epochs, with elapsed training time carried across restarts."""
        if epochs < self.start:
            raise ValueError("requested final epoch precedes the checkpoint")
        if self.directory is not None or self.digest is not None:
            current = _batch_digest(batch)
            if self.digest is not None and current != self.digest:
                raise ValueError("training examples or their order differ from the checkpoint")
            self.digest = current
        started = time.perf_counter()
        with self.config.execution(self.seed):
            if self.cpu_rng is not None:
                torch.set_rng_state(self.cpu_rng)
            if self.cuda_rng is not None and self.network.w1.device.type == "cuda":
                torch.cuda.set_rng_state(self.cuda_rng, self.network.w1.device)
            for step in range(self.start + 1, epochs + 1):
                self.rng.shuffle(self.order)
                loss = train_epoch(self.network, self.optimizer, batch, self.order)
                seconds = self.elapsed + time.perf_counter() - started
                if self.directory is not None and (step == epochs or interval and step % interval == 0):
                    self._save(step, seconds)
                yield TelemetryRecord(step=step, loss=loss, wall_time_seconds=seconds)
