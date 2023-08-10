import pytest
import numpy as np
from dtemulate.experimental_design import ExperimentalDesign, LatinHypercube
from dtemulate.emulators import GaussianProcess, RandomForest

def simple_sim(params):
    """A simple simulator."""
    x, y = params
    return x + 2*y

lh = LatinHypercube([(0., 1.), (10., 100.)])
sim_in = lh.sample(10)
sim_out = [simple_sim(p) for p in sim_in]
    
# GaussianProcess
def test_gp_initialisation():
    gp = GaussianProcess()
    assert gp.model is None
    
def test_gp_fit():
    gp = GaussianProcess()
    gp.fit(sim_in, sim_out)
    predictions = gp.predict(sim_in)
    assert predictions is not None
    assert len(predictions.mean) == len(sim_out)
    # accuracy
    assert np.allclose(predictions.mean, sim_out)
    
# RandomForest
def test_rf_initialisation():
    rf = RandomForest()
    assert rf.model is not None
    
def test_rf_fit():
    rf = RandomForest(n_estimators=100)
    rf.fit(sim_in, sim_out)
    predictions = rf.predict(sim_in)
    assert predictions is not None
    assert len(predictions) == len(sim_out)
    # accuracy
    #assert np.allclose(predictions, sim_out, atol=100) 

    