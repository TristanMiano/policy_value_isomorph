import random
from dataclasses import asdict

import pytest
import torch

from policy_value_isomorph.policy_mlp import untrained_policy_mlp
from policy_value_isomorph.q_mlp import _init_model as init_q
from policy_value_isomorph.torch_mlp import TorchMLP, TrainingBatch, train_epoch
from policy_value_isomorph.value_mlp import _init_model as init_v


@pytest.mark.parametrize("width,outputs", [(10, 9), (10, 1), (19, 1)])
def test_model_batch_and_optimizer_boundaries(width, outputs):
    if outputs == 9:
        saved = untrained_policy_mlp(4, seed=13)
    else:
        initialize = init_q if width == 19 else init_v
        saved = initialize(width, 4, random.Random(13))
    model = TorchMLP(saved, device=torch.device("cpu"))
    assert model.weights() == asdict(saved)
    rows = [[0.5] * width, [-1.0] * width]
    predictions = model.predict(rows)
    assert predictions.device.type == "cpu" and predictions.dtype == torch.float64
    assert predictions.shape == ((2, 9) if outputs == 9 else (2,))
    assert not predictions.requires_grad
    if outputs == 9:
        mask = [True, True] + [False] * 7
        batch = TrainingBatch.policy(model, rows, [0, 1], [mask, mask])
        assert batch.targets.dtype == torch.long and batch.legal.dtype == torch.bool
        with pytest.raises(ValueError, match="legal moves"):
            TrainingBatch.policy(model, rows, [0, 3], [mask, mask])
    else:
        batch = TrainingBatch.regression(model, rows, [0.25, -0.5])
        assert batch.targets.dtype == torch.float64
        with pytest.raises(ValueError, match="target per example"):
            TrainingBatch.regression(model, rows, [0.25])
    with pytest.raises(ValueError, match="input width"):
        model.predict([[0.0]])
    train_epoch(model, torch.optim.SGD(model.parameters(), lr=0.05), batch, [1, 0])
    assert not torch.equal(predictions, model.predict(rows))
    assert asdict(saved) != model.weights()
