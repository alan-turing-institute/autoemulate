import pytest
import numpy as np
from dtemulate.experimental_design import ExperimentalDesign, LatinHypercube
from dtemulate.emulators import GaussianProcess

def simple_sim(params):
    """A simple simulator."""
    x, y = params
    return x + 2*y

def test_gp_initialisation():
    gp = GaussianProcess()
    assert gp.model is None
    
def test_gp_fit():
    gp = GaussianProcess()
    lh = LatinHypercube([(0., 1.), (10., 100.)])
    sim_in = lh.sample(10)
    sim_out = [simple_sim(p) for p in sim_in]
    gp.fit(sim_in, sim_out)
    predictions = gp.predict(sim_in)
    
    assert predictions is not None
    assert len(predictions.mean) == len(sim_out)
    
    # accuracy
    assert np.allclose(predictions.mean, sim_out)
    