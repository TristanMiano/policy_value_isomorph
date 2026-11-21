import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda:0", marks=[pytest.mark.gpu, pytest.mark.skipif(
        not torch.cuda.is_available(), reason="no CUDA device in this PyTorch installation")]),
])
def test_installed_training_pipeline_and_logs(tmp_path, device):
    script = Path(__file__).resolve().parents[1] / "scripts/smoke_torch_training.py"
    env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    result = subprocess.run([sys.executable, str(script), "--output-dir", str(tmp_path), "--device", device],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    assert result.stdout.count("in 2 games") == 3
    run, = tmp_path.glob("smoke-*")
    for kind in ("policy", "value", "q"):
        dest = run / "telemetry" / "tic_tac_toe" / kind / "seed_28"
        rows = [json.loads(line) for line in (dest / "training.jsonl").read_text().splitlines()]
        with (dest / "training.csv").open(newline="") as stream:
            table = list(csv.DictReader(stream))
        assert [row["record"]["step"] for row in rows] == [1, 2]
        assert [int(row["step"]) for row in table] == [1, 2]
        assert [r["hyperparameters"]["device"] for r in rows] == [device, device]
        assert rows[1]["record"]["wall_time_seconds"] >= rows[0]["record"]["wall_time_seconds"]
        for step in (1, 2):
            file = run / "checkpoints" / kind / f"step_{step:06d}.pt"
            payload = torch.load(file, weights_only=True, map_location="cpu")
            assert payload["optimizer_state"]["state"]
            assert payload["metadata"]["epoch"] == step
