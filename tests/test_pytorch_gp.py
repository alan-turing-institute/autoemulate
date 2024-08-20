import gpytorch
import pytest
import torch
from sklearn.datasets import make_regression

from autoemulate.emulators.gaussian_process_torch import GaussianProcessTorch
from autoemulate.emulators.neural_networks.gp_module import GPModule


@pytest.fixture
def sample_data_y1d():
    X, y = make_regression(n_samples=10, n_features=5, n_targets=1, random_state=0)
    return X, y


# ------------------------------------------------------------
# Test GPModule
# ------------------------------------------------------------
def test_GPModule_forward_pass(sample_data_y1d):
    X, y = sample_data_y1d
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp = GPModule(X, y, likelihood)
    gp.eval()
    likelihood.eval()
    out = gp(X)
    assert out.mean.shape == (10,)
    assert out.variance.shape == (10,)


# ------------------------------------------------------------
# Test GaussianProcessTorch
# ------------------------------------------------------------
def test_GaussianProcessTorch(sample_data_y1d):
    X, y = sample_data_y1d
    print(y.shape)
    gp = GaussianProcessTorch()
    gp.fit(X, y)
    print(gp.predict(X))
