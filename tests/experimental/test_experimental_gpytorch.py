import torch
from gpytorch.likelihoods import MultitaskGaussianLikelihood

from autoemulate.experimental.config import FitConfig
from autoemulate.experimental.emulators.gpytorch_backend import GPExactRBF


def test_gp_exact_rbf():
    x = torch.rand(10, 2)
    y = torch.rand(10, 1)
    likelihood = MultitaskGaussianLikelihood(num_tasks=1)
    model = GPExactRBF(x, y, likelihood)
    model.fit(x, y, FitConfig())


# TODO: add test for GPyTorchBackend
