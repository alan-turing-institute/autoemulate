import numpy as np
import pytest
from autoemulate_design import ExperimentalDesign
from autoemulate_design import LatinHypercube


# Test LatinHypercube Initialisation
@pytest.fixture
def lh():
    return LatinHypercube([(0.0, 1.0), (10.0, 100.0)])


def test_init(lh):
    assert lh is not None


def test_parameters(lh):
    assert lh.get_n_parameters() == 2


# Test LatinHypercube Sampling
@pytest.fixture
def lh_sample(lh):
    return lh.sample(3)


def test_sample_shape(lh_sample):
    assert lh_sample.shape == (3, 2)


def test_sample_lower_bound_par1(lh_sample):
    assert np.all(lh_sample[:, 0] >= 0.0)


def test_sample_upper_bound_par1(lh_sample):
    assert np.all(lh_sample[:, 0] <= 1.0)


def test_sample_lower_bound_par2(lh_sample):
    assert np.all(lh_sample[:, 1] >= 10.0)
