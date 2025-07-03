import numpy as np
import pytest
import torch
from autoemulate.experimental.compare import AutoEmulate
from sklearn.datasets import make_regression


@pytest.fixture(scope="module")
def sample_data_y1d():
    x, y = make_regression(n_samples=20, n_features=5, n_targets=1, random_state=0)  # type: ignore noqa: PGH003
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture(scope="module")
def new_data_y1d():
    x, y = make_regression(n_samples=20, n_features=5, n_targets=1, random_state=1)  # type: ignore noqa: PGH003
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture(scope="module")
def sample_data_y2d():
    x, y = make_regression(n_samples=20, n_features=5, n_targets=2, random_state=0)  # type: ignore noqa: PGH003
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture(scope="module")
def new_data_y2d():
    x, y = make_regression(n_samples=20, n_features=5, n_targets=2, random_state=1)  # type: ignore noqa: PGH003
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture(scope="module")
def sample_data_y2d_100_targets():
    x, y = make_regression(n_samples=20, n_features=5, n_targets=100, random_state=0)  # type: ignore noqa: PGH003
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture(scope="module")
def new_data_y2d_100_targets():
    x, y = make_regression(n_samples=20, n_features=5, n_targets=100, random_state=1)  # type: ignore noqa: PGH003
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture(scope="module")
def sample_data_y2d_1000_targets():
    x, y = make_regression(n_samples=20, n_features=5, n_targets=1000, random_state=0)  # type: ignore noqa: PGH003
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture(scope="module")
def new_data_y2d_1000_targets():
    x, y = make_regression(n_samples=20, n_features=5, n_targets=1000, random_state=1)  # type: ignore noqa: PGH003
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture
def sample_data_rbf():
    """
    At least 56 data points are required when
    `degree` is 3 (max value in hyperparameters)
    and the number of dimensions is 5
    """
    x, y = make_regression(n_samples=56, n_features=5, n_targets=2, random_state=0)  # type: ignore noqa: PGH003
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture
def new_data_rbf():
    x, y = make_regression(n_samples=56, n_features=5, n_targets=2, random_state=1)  # type: ignore noqa: PGH003
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture(scope="module")
def np_1d():
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture(scope="module")
def np_2d():
    return np.array([[1.0, 2.0], [3.0, 4.0]])


@pytest.fixture(scope="module")
def tensor_2d():
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]])


@pytest.fixture(scope="module")
def tensor_1d():
    return torch.tensor([1.0, 2.0])


@pytest.fixture(scope="module")
def sigma_full():
    # Full covariance matrix: shape (n_samples, n_features, n_features)
    return torch.eye(2).repeat(2, 1, 1)


@pytest.fixture(scope="module")
def tensor_2d_mismatch():
    # Shape (1, 2)
    return torch.tensor([[5.0, 6.0]])


@pytest.fixture(scope="module")
def tensor_2d_pair():
    # Shape (2, 2)
    return torch.tensor([[5.0, 6.0], [7.0, 8.0]])


@pytest.fixture(scope="module")
def noisy_data():
    """
    Generate a highly noisy dataset to test stochasticity effects.
    """
    rng = np.random.RandomState(0)
    x = torch.tensor(rng.normal(size=(100, 2)), dtype=torch.float32)
    y = torch.tensor(rng.normal(size=(100,)), dtype=torch.float32)
    x2 = x[:4].clone()
    return x, y, x2


@pytest.fixture(scope="module")
def ae_single_output(sample_data_y1d):
    x, y = sample_data_y1d
    em = AutoEmulate(x, y)
    em.compare()
    return em


@pytest.fixture(scope="module")
def ae_multi_output(sample_data_y2d):
    x, y = sample_data_y2d
    em = AutoEmulate(x, y)
    em.compare()
    return em
