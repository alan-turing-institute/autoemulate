import pytest
import torch
from autoemulate.simulations.base import LatinHypercube


# Test LatinHypercube Initialisation
@pytest.fixture
def lh():
    return LatinHypercube([(0.0, 1.0), (10.0, 100.0), (1000.0, 1000.0)])


def test_init(lh):
    assert lh is not None


def test_parameters(lh):
    assert lh.get_n_parameters() == 3


@pytest.fixture
def n_samples():
    return 1_000


# Test LatinHypercube Sampling
@pytest.fixture
def lh_sample(lh, n_samples):
    return lh.sample(n_samples)


def test_sample_shape(lh_sample, n_samples):
    assert lh_sample.shape == (n_samples, 3)


def test_sample_lower_bound_par1(lh_sample):
    assert torch.all(lh_sample[:, 0] >= 0.0)


def test_sample_upper_bound_par1(lh_sample):
    assert torch.all(lh_sample[:, 0] <= 1.0)


def test_sample_lower_bound_par2(lh_sample):
    assert torch.all(lh_sample[:, 1] >= 10.0)


def test_sample_upper_bound_par2(lh_sample):
    assert torch.all(lh_sample[:, 1] <= 100.0)


def test_sample_bound_par3(lh_sample):
    assert torch.all(lh_sample[:, 2] == 1000.0)
