import pytest
import torch
from autoemulate.experimental.emulators.lightgbm.lightgbm import (
    LightGBM,
)
from autoemulate.experimental.tuner import Tuner
from sklearn.datasets import make_regression


@pytest.fixture
def sample_data_y1d():
    x, y = make_regression(n_samples=20, n_features=5, n_targets=1, random_state=0)  # type: ignore
    return torch.Tensor(x), torch.Tensor(y)


def test_tune_gp(sample_data_y1d):
    x, y = sample_data_y1d
    tuner = Tuner(x, y, n_iter=5)
    scores, configs = tuner.run(LightGBM)
    assert len(scores) == 5
    assert len(configs) == 5