import pytest
import numpy as np
import pandas as pd
from autoemulate.experimental_design import ExperimentalDesign, LatinHypercube
from autoemulate.emulators import GaussianProcess, RandomForest
from autoemulate.compare import AutoEmulate


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
    em = AutoEmulate()
    em.setup(X=sim_in, y=sim_out, cv=5)
    em.compare()
    return em.scores
