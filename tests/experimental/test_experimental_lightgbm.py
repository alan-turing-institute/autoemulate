import pytest
import torch
from autoemulate.experimental.emulators.lightgbm.lightgbm import (
    LightGBM,
)
from autoemulate.experimental.tuner import Tuner
from autoemulate.experimental.types import DistributionLike, OutputLike
from sklearn.datasets import make_regression


@pytest.fixture
def sample_data_y1d():
    x, y = make_regression(n_samples=20, n_features=5, n_targets=1, random_state=0)  # type: ignore
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture
def new_data_y1d():
    x, y = make_regression(n_samples=20, n_features=5, n_targets=1, random_state=1)  # type: ignore
    return torch.Tensor(x), torch.Tensor(y)


def test_predict_lightgbm(sample_data_y1d, new_data_y1d):
    x, y = sample_data_y1d
    lgbm = LightGBM(
        x,
        y,
    )
    lgbm.fit(x, y)
    x2, _ = new_data_y1d
    y_pred = lgbm.predict(x2)
    assert isinstance(y_pred, OutputLike)


def test_tune_lightgbm(sample_data_y1d):
    x, y = sample_data_y1d
    tuner = Tuner(x, y, n_iter=5)
    scores, configs = tuner.run(LightGBM)
    assert len(scores) == 5
    assert len(configs) == 5