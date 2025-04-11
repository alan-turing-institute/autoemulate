import pytest
import torch

from sklearn.datasets import make_regression


@pytest.fixture
def sample_data_y1d():
    x, y = make_regression(n_samples=20, n_features=5, n_targets=1, random_state=0)  # type: ignore
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture
def new_data_y1d():
    x, y = make_regression(n_samples=20, n_features=5, n_targets=1, random_state=1)  # type: ignore
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture
def sample_data_y2d():
    x, y = make_regression(n_samples=20, n_features=5, n_targets=2, random_state=0)  # type: ignore
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture
def new_data_y2d():
    x, y = make_regression(n_samples=20, n_features=5, n_targets=2, random_state=1)  # type: ignore
    return torch.Tensor(x), torch.Tensor(y)
