import os

# Cap CPU threadpools before importing torch/numpy/sklearn so their threadpools
# pick up these limits at init. Without this, each test fit launches O(cores)
# threads and oversubscribed runs (e.g. on many-core nodes) spend more time in
# the scheduler than doing work. Under pytest-xdist, each worker is a separate
# process running this conftest, so we cap to 1 thread per worker to avoid
# (workers x threads) blowup; in single-process runs we allow a few threads
# for intra-test parallelism. setdefault leaves shell overrides alone.
_default_threads = "1" if "PYTEST_XDIST_WORKER" in os.environ else "4"
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, _default_threads)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from sklearn.datasets import make_regression  # noqa: E402

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

N_S = 20


@pytest.fixture
def sample_data_y1d():
    x, y = make_regression(n_samples=N_S, n_features=5, n_targets=1, random_state=0)  # type: ignore noqa: PGH003
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture
def new_data_y1d():
    x, y = make_regression(n_samples=N_S, n_features=5, n_targets=1, random_state=1)  # type: ignore noqa: PGH003
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture
def sample_data_y2d():
    x, y = make_regression(n_samples=N_S, n_features=5, n_targets=2, random_state=0)  # type: ignore noqa: PGH003
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture
def new_data_y2d():
    x, y = make_regression(n_samples=N_S, n_features=5, n_targets=2, random_state=1)  # type: ignore noqa: PGH003
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture
def sample_data_y2d_100_targets():
    x, y = make_regression(n_samples=N_S, n_features=5, n_targets=100, random_state=0)  # type: ignore  # noqa: PGH003
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture
def new_data_y2d_100_targets():
    x, y = make_regression(n_samples=N_S, n_features=5, n_targets=100, random_state=1)  # type: ignore noqa: PGH003
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture
def sample_data_y2d_1000_targets():
    x, y = make_regression(n_samples=N_S, n_features=5, n_targets=1000, random_state=0)  # type: ignore noqa: PGH003
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture
def new_data_y2d_1000_targets():
    x, y = make_regression(n_samples=N_S, n_features=5, n_targets=1000, random_state=1)  # type: ignore noqa: PGH003
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture
def sample_data_for_ae_compare():
    """
    At least 56 data points are required when
    `degree` is 3 (max value in hyperparameters)
    and the number of dimensions is 5 for rbf
    """
    x, y = make_regression(n_samples=56, n_features=5, n_targets=2, random_state=0)  # type: ignore noqa: PGH003
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture
def new_data_rbf():
    x, y = make_regression(n_samples=56, n_features=5, n_targets=2, random_state=1)  # type: ignore noqa: PGH003
    return torch.Tensor(x), torch.Tensor(y)


@pytest.fixture
def np_1d():
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def np_2d():
    return np.array([[1.0, 2.0], [3.0, 4.0]])


@pytest.fixture
def tensor_2d():
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]])


@pytest.fixture
def tensor_1d():
    return torch.tensor([1.0, 2.0])


@pytest.fixture
def sigma_full():
    # Full covariance matrix: shape (n_samples, n_features, n_features)
    return torch.eye(2).repeat(2, 1, 1)


@pytest.fixture
def tensor_2d_mismatch():
    # Shape (1, 2)
    return torch.tensor([[5.0, 6.0]])


@pytest.fixture
def tensor_2d_pair():
    # Shape (2, 2)
    return torch.tensor([[5.0, 6.0], [7.0, 8.0]])


@pytest.fixture
def noisy_data():
    """
    Generate a highly noisy dataset to test stochasticity effects.
    """
    rng = np.random.RandomState(0)
    x = torch.tensor(rng.normal(size=(100, 2)), dtype=torch.float32)
    y = torch.tensor(rng.normal(size=(100,)), dtype=torch.float32)
    x2 = x[:4].clone()
    return x, y, x2
