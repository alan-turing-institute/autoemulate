import pytest
import numpy as np
from dtemulate.experimental_design import ExperimentalDesign, LatinHypercube

def test_abstract():
    with pytest.raises(TypeError):
        ed = ExperimentalDesign([(0., 1.), (10., 100.)])

class TestLatinHypercube:
    @pytest.fixture
    def lh(self):
        return LatinHypercube([(0., 1.), (10., 100.)])
      
    def test_init(self, lh):
        assert lh.get_n_parameters() == 2
        
    def test_sample(self, lh):
        samples = lh.sample(3)
        assert samples.shape == (3, 2)
        assert np.all(samples[:,0] >= 0.)
        assert np.all(samples[:,0] <= 1.)
        assert np.all(samples[:,1] >= 10.)
        assert np.all(samples[:,1] <= 100.)
