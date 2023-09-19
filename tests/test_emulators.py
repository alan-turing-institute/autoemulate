import pytest
import numpy as np
from autoemulate.experimental_design import ExperimentalDesign, LatinHypercube
from autoemulate.emulators import GaussianProcess, RandomForest


def simple_sim(params):
    """A simple simulator."""
    x, y = params
    return x + 2 * y


# fixture for simulation input and output
@pytest.fixture(scope="module")
def simulation_io():
    """Setup for tests (Arrange)"""
    lh = LatinHypercube([(0.0, 1.0), (10.0, 100.0)])
    sim_in = lh.sample(10)
    sim_out = [simple_sim(p) for p in sim_in]
    return sim_in, sim_out


# fixture for fitted GP model
@pytest.fixture(scope="module")
def gp_model(simulation_io):
    """Setup for tests (Arrange)"""
    sim_in, sim_out = simulation_io
    gp = GaussianProcess()
    gp.fit(sim_in, sim_out)
    return gp


# fixture for fitted RF model
@pytest.fixture(scope="module")
def rf_model(simulation_io):
    """Setup for tests (Arrange)"""
    sim_in, sim_out = simulation_io
    rf = RandomForest()
    rf.fit(sim_in, sim_out)
    return rf


# Test Gaussian Process
def test_gp_initialisation():
    gp = GaussianProcess()
    assert gp is not None


def test_gp_pred_exists(gp_model, simulation_io):
    sim_in, sim_out = simulation_io
    predictions = gp_model.predict(sim_in)
    assert predictions is not None


def test_gp_pred_len(gp_model, simulation_io):
    sim_in, sim_out = simulation_io
    predictions = gp_model.predict(sim_in)
    assert len(predictions.mean) == len(sim_out)


def test_gp_pred_type(gp_model, simulation_io):
    sim_in, sim_out = simulation_io
    predictions = gp_model.predict(sim_in)
    print(type(predictions.mean))
    assert isinstance(predictions.mean, np.ndarray)


# Test Random Forest
def test_rf_initialisation():
    rf = RandomForest()
    assert rf is not None


def test_rf_pred_exists(rf_model, simulation_io):
    sim_in, sim_out = simulation_io
    predictions = rf_model.predict(sim_in)
    assert predictions is not None


def test_rf_pred_len(rf_model, simulation_io):
    sim_in, sim_out = simulation_io
    predictions = rf_model.predict(sim_in)
    assert len(predictions) == len(sim_out)


def test_rf_pred_type(rf_model, simulation_io):
    sim_in, sim_out = simulation_io
    predictions = rf_model.predict(sim_in)
    assert isinstance(predictions, np.ndarray)
