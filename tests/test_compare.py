import pytest
import numpy as np
from autoemulate.experimental_design import ExperimentalDesign, LatinHypercube
from autoemulate.emulators import GaussianProcess, RandomForest
from autoemulate.compare import compare


def simple_sim(params):
    """A simple simulator."""
    x, y = params
    return x + 2 * y


@pytest.fixture(scope="module")
def compare_results():
    """Setup for tests (Arrange)"""
    lh = LatinHypercube([(0.0, 1.0), (10.0, 100.0)])
    np.random.seed(41)
    sim_in = lh.sample(10)
    sim_out = [simple_sim(p) for p in sim_in]
    return compare(sim_in, sim_out)


def test_compare_not_none(compare_results):
    assert compare_results is not None


# check that the results are a dictionary
def test_compare_dict(compare_results):
    assert isinstance(compare_results, dict)


# check that dict values are numeric
def test_compare_dict_values_numeric(compare_results):
    assert all(isinstance(v, float) for v in compare_results.values())
