import numpy as np
from numpy.testing import assert_allclose
import scipy.stats
import pytest
from inspect import signature
import types
from ..ExperimentalDesign import (
    ExperimentalDesign,
    MonteCarloDesign,
    LatinHypercubeDesign,
    MaxiMinLHC,
)

def test_ExperimentalDesign_init():
    "test the init method of ExperimentalDesign"
    
    ed = ExperimentalDesign(3)
    assert ed.n_parameters == 3
    assert len(ed.distributions) == 3
    for dist in ed.distributions:
        assert callable(dist)
        assert len(signature(dist).parameters) == 1
    
    ed = ExperimentalDesign(3.)
    assert ed.n_parameters == 3
    assert len(ed.distributions) == 3
    for dist in ed.distributions:
        assert callable(dist)
        assert len(signature(dist).parameters) == 1
    
    ed = ExperimentalDesign(3, (0., 1.))
    assert ed.n_parameters == 3
    assert len(ed.distributions) == 3
    for dist in ed.distributions:
        assert callable(dist)
        assert len(signature(dist).parameters) == 1
    
    ed = ExperimentalDesign(3, scipy.stats.uniform(loc = 0., scale = 1.).ppf)
    assert ed.n_parameters == 3
    assert len(ed.distributions) == 3
    for dist in ed.distributions:
        assert callable(dist)
        assert len(signature(dist).parameters) == 1

    ed = ExperimentalDesign(2, [(0., 5.), (0., 10.)])
    assert ed.n_parameters == 2
    assert len(ed.distributions) == 2
    for dist in ed.distributions:
        assert callable(dist)
        assert len(signature(dist).parameters) == 1
    
    ed = ExperimentalDesign([(0., 1.), (0., 5.), (0., 10.)])
    assert ed.n_parameters == 3
    assert len(ed.distributions) == 3
    for dist in ed.distributions:
        assert callable(dist)
        assert len(signature(dist).parameters) == 1
    
    ed = ExperimentalDesign([scipy.stats.uniform(loc = 0., scale = 1.).ppf,
                             scipy.stats.uniform(loc = 0., scale = 5.).ppf,
                             scipy.stats.uniform(loc = 0., scale = 10.).ppf])
    assert ed.n_parameters == 3
    assert len(ed.distributions) == 3
    for dist in ed.distributions:
        assert callable(dist)
        assert len(signature(dist).parameters) == 1
    
    ed = ExperimentalDesign([(0., 1.), scipy.stats.uniform(loc = 0., scale = 5.).ppf, (0., 10.)])
    assert ed.n_parameters == 3
    assert len(ed.distributions) == 3
    for dist in ed.distributions:
        assert callable(dist)
        assert len(signature(dist).parameters) == 1
    
def test_ExperimentalDesign_init_failures():
    "test situations where init for ExperimentalDesign should fail"
    
    with pytest.raises(ValueError):
        ed = ExperimentalDesign(-1)

    with pytest.raises(ValueError):
        ed = ExperimentalDesign(4, (0., 2., 2.))

    with pytest.raises(TypeError):
        ed = ExperimentalDesign(3, (0., 2., 2.))
    
    with pytest.raises(ValueError):
        ed = ExperimentalDesign(3, scipy.stats.uniform.ppf)
        
    with pytest.raises(ValueError):
        ed = ExperimentalDesign(2, [(0., 1.), (0., 5.), (0., 10.)])
        
    with pytest.raises(ValueError):
        ed = ExperimentalDesign([(1., 0.), (0., 5.), (0., 10.)])
        
    with pytest.raises(ValueError):
        ed = ExperimentalDesign(2, [(0., 1., 2), (0., 5.)])
        
    with pytest.raises(ValueError):
        ed = ExperimentalDesign([scipy.stats.uniform.ppf, (0., 5.)])
        
    
def test_ExperimentalDesign_get_n_parameters():
    "test get_n_samples method of ExperimentalDesign"
    
    ed = ExperimentalDesign(3)
    assert ed.get_n_parameters() == 3
    
def test_ExperimentalDesign_get_method():
    "test get_method method of ExperimentalDesign"
    
    ed = ExperimentalDesign(3)
    with pytest.raises(NotImplementedError):
        ed.get_method()
        
def test_ExperimentalDesign_draw_samples():
    "test _draw_samples method of ExperimentalDesign"
    
    ed = ExperimentalDesign(3)
    with pytest.raises(NotImplementedError):
        ed.sample(3)
        
def test_ExperimentalDesign_sample():
    "test sample method of ExperimentalDesign"

    ed = ExperimentalDesign(3)
    
    def tmp_draw_samples(self, n_samples):
        return np.ones((n_samples, self.get_n_parameters()))*0.5

    ed._draw_samples = types.MethodType(tmp_draw_samples, ed)
    
    sample = ed.sample(2)
    sample_expected = np.ones((2, 3))*0.5
    assert_allclose(sample, sample_expected)
    
    def tmp_draw_samples(self, n_samples):
        return np.ones((n_samples, self.get_n_parameters()))*0.25

    ed._draw_samples = types.MethodType(tmp_draw_samples, ed)
    
    sample = ed.sample(2)
    sample_expected = np.ones((2, 3))*0.25
    assert_allclose(sample, sample_expected)

def test_ExperimentalDesign_sample_failures():
    "test situation where ExperimentalDesign should raise an exception"
    
    with pytest.raises(AssertionError):
        ed = ExperimentalDesign(3)
        sample = ed.sample(-2)
    
    with pytest.raises(AssertionError):
        f = lambda x: np.nan
        ed = ExperimentalDesign(2, f)
        
        def tmp_draw_samples(self, n_samples):
            return np.ones((n_samples, self.get_n_parameters()))*0.5

        ed._draw_samples = types.MethodType(tmp_draw_samples, ed)
        sample = ed.sample(2)

def test_ExperimentalDesign_str():
    "test string method of ExperimentalDesign"
    
    ed = ExperimentalDesign(3)
    assert str(ed) == "Experimental Design with 3 parameters"

def test_MonteCarloDesign_init():
    "test that init method correctly sets method for Monte Carlo"
    
    ed = MonteCarloDesign(3)
    assert ed.get_method() == "Monte Carlo"
    assert str(ed) == "Monte Carlo Experimental Design with 3 parameters"
    
def test_MonteCarloDesign_draw_samples():
    "test low level method for drawing samples from a monte carlo design"
    
    ed = MonteCarloDesign(3)
    np.random.seed(73859)
    sample = ed._draw_samples(4)
    sample_expected = np.array([[0.4587231420553423, 0.8712779046673758, 0.9592574641762792],
                                [0.6863958521075596, 0.1642774978932529, 0.5102796498909351],
                                [0.0945231904820333, 0.7578914914795801, 0.4338919363034113],
                                [0.8612216997446789, 0.0194411385272029, 0.0765283763808666]])
    assert_allclose(sample, sample_expected)
    
def test_MonteCarloDesign_sample():
    "test method to samples from a Monte Carlo design"
    
    ed = MonteCarloDesign(3)
    np.random.seed(73859)
    sample = ed.sample(4)
    sample_expected = np.array([[0.4587231420553423, 0.8712779046673758, 0.9592574641762792],
                                [0.6863958521075596, 0.1642774978932529, 0.5102796498909351],
                                [0.0945231904820333, 0.7578914914795801, 0.4338919363034113],
                                [0.8612216997446789, 0.0194411385272029, 0.0765283763808666]])
    assert_allclose(sample, sample_expected)
    
    ed = MonteCarloDesign([scipy.stats.uniform(loc = 0., scale = 5.).ppf, scipy.stats.uniform(loc = 0., scale = 2.).ppf])
    sample = ed.sample(3)
    sample_expected = np.array([[1.4976951596996646, 1.9820310379261286],
                                [4.841151376332333 , 0.9447917510829495],
                                [3.8191224939802453, 0.6532873534667354]])
    assert_allclose(sample, sample_expected)
        
def test_LatinHypercubeDesign_init():
    "test that init method correctly sets method for Latin Hypercube"
    
    ed = LatinHypercubeDesign(3)
    assert ed.get_method() == "Latin Hypercube"
    assert str(ed) == "Latin Hypercube Experimental Design with 3 parameters"
    
def test_LatinHypercubeDesign_draw_samples():
    "test low level method for drawing samples from a latin hypercube design"
    
    ed = LatinHypercubeDesign(2)
    np.random.seed(73859)
    sample = ed._draw_samples(5)
    sample_expected = np.array([[0.3057514377091521, 0.2505993516661572],
                                [0.5739619489876799, 0.0113841212300836],
                                [0.7776329187470997, 0.5688057959369438],
                                [0.8885708869643932, 0.7610737581544206],
                                [0.0628594487771623, 0.8735926821354483]])
    assert_allclose(sample, sample_expected)
    
def test_LatinHypercubeDesign_sample():
    "test method to samples from a latin hypercube design"
    
    ed = LatinHypercubeDesign(2)
    np.random.seed(73859)
    sample = ed.sample(5)
    sample_expected = np.array([[0.3057514377091521, 0.2505993516661572],
                                [0.5739619489876799, 0.0113841212300836],
                                [0.7776329187470997, 0.5688057959369438],
                                [0.8885708869643932, 0.7610737581544206],
                                [0.0628594487771623, 0.8735926821354483]])
    assert_allclose(sample, sample_expected)
    
    ed = LatinHypercubeDesign([scipy.stats.uniform(loc = 0., scale = 5.).ppf, scipy.stats.uniform(loc = 0., scale = 2.).ppf])
    sample = ed.sample(3)
    sample_expected = np.array([[3.0319074209362724, 1.4378127211566698],
                                [3.8539110146362914, 0.186886121478102 ],
                                [0.3690344832370781, 1.1036076298948434]])
    assert_allclose(sample, sample_expected)

def test_MaxiMinLHC(monkeypatch):
    "test the MaxiMin version of an LHC"

    class MockLHCSample:
        def __init__(self):
            self.i = 0

        def __call__(self, n_samples):
            if self.i == 0:
                self.i = 1
                return np.array([[0.3], [0.7]])
            elif self.i == 1:
                self.i = 2
                return np.array([[0.25], [0.75]])
            else:
                return np.array([[0.0], [1.0]])

    def reset_lhc_draw_samples():
        monkeypatch.setattr(
            "mogp_emulator.ExperimentalDesign.LatinHypercubeDesign._draw_samples",
            MockLHCSample(),
        )

    reset_lhc_draw_samples()

    ed = MaxiMinLHC(1)
    sample = ed._draw_samples(2, n_tries=1)
    assert_allclose(sample, np.array([[0.3], [0.7]]))

    reset_lhc_draw_samples()

    ed = MaxiMinLHC(1)
    sample = ed._draw_samples(2, n_tries=2)
    assert_allclose(sample, np.array([[0.25], [0.75]]))

    reset_lhc_draw_samples()

    ed = MaxiMinLHC(1)
    sample = ed._draw_samples(2, n_tries=3)
    assert_allclose(sample, np.array([[0.0], [1.0]]))

    reset_lhc_draw_samples()

    ed = MaxiMinLHC(1)
    sample = ed._draw_samples(2)
    assert_allclose(sample, np.array([[0.0], [1.0]]))

    with pytest.raises(AssertionError):
        ed = MaxiMinLHC(1)
        sample = ed._draw_samples(-1)

    with pytest.raises(AssertionError):
        ed = MaxiMinLHC(1)
        sample = ed._draw_samples(1, n_tries=-1)

    ed = MaxiMinLHC(1)
    sample = ed._draw_samples(1, metric="cityblock")
