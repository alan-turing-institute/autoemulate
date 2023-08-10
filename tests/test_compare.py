import pytest
import numpy as np
from dtemulate.experimental_design import ExperimentalDesign, LatinHypercube
from dtemulate.emulators import GaussianProcess, RandomForest
from dtemulate.compare import compare

def simple_sim(params):
    """A simple simulator."""
    x, y = params
    return x + 2*y

lh = LatinHypercube([(0., 1.), (10., 100.)])
# set random seed
np.random.seed(41)
sim_in = lh.sample(10)
sim_out = [simple_sim(p) for p in sim_in]

def test_compare():
      results = compare(sim_in, sim_out)
      assert results is not None
      assert len(results) == 2 # number of models will change
      assert 'GaussianProcess' in results
      assert 'RandomForest' in results
      assert results['GaussianProcess'] < results['RandomForest'] # RMSE should be lower for GP
      
