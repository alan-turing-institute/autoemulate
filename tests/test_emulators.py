import numpy as np
import pytest

from autoemulate.emulators import GaussianProcessMOGP
from autoemulate.emulators import NeuralNetSk
from autoemulate.emulators import RandomForest
from autoemulate.experimental_design import LatinHypercube


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
    gp = GaussianProcessMOGP()
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


@pytest.fixture(scope="module")
def nn_sk_model(simulation_io):
    """Setup for tests (Arrange)"""
    sim_in, sim_out = simulation_io
    nn_sk = NeuralNetSk()
    nn_sk.fit(sim_in, sim_out)
    return nn_sk


# Test Gaussian Process
def test_gp_initialisation():
    gp = GaussianProcessMOGP()
    assert gp is not None


def test_gp_pred_exists(gp_model, simulation_io):
    sim_in, sim_out = simulation_io
    predictions = gp_model.predict(sim_in)
    assert predictions is not None


def test_gp_pred_len(gp_model, simulation_io):
    sim_in, sim_out = simulation_io
    predictions = gp_model.predict(sim_in)
    assert len(predictions) == len(sim_out)


def test_gp_pred_type(gp_model, simulation_io):
    sim_in, sim_out = simulation_io
    predictions = gp_model.predict(sim_in)
    assert isinstance(predictions, np.ndarray)


def test_gp_pred_with_std_len(gp_model, simulation_io):
    sim_in, sim_out = simulation_io
    predictions = gp_model.predict(sim_in, return_std=True)
    assert len(predictions) == 2


def test_gp_pred_with_std_len_of_var(gp_model, simulation_io):
    sim_in, sim_out = simulation_io
    predictions = gp_model.predict(sim_in, return_std=True)
    assert len(predictions[1]) == len(sim_out)


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


# Test Neural Network sklearn
def test_nn_sk_initialisation():
    nn_sk = NeuralNetSk()
    assert nn_sk is not None


def test_nn_sk_pred_exists(nn_sk_model, simulation_io):
    sim_in, sim_out = simulation_io
    predictions = nn_sk_model.predict(sim_in)
    assert predictions is not None


def test_nn_sk_pred_len(nn_sk_model, simulation_io):
    sim_in, sim_out = simulation_io
    predictions = nn_sk_model.predict(sim_in)
    assert len(predictions) == len(sim_out)


def test_nn_sk_pred_type(nn_sk_model, simulation_io):
    sim_in, sim_out = simulation_io
    predictions = nn_sk_model.predict(sim_in)
    assert isinstance(predictions, np.ndarray)
