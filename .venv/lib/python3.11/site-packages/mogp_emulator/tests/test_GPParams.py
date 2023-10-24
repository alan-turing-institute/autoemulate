import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..GPParams import GPParams

def test_GPParams_init():
    "Test init method of GPParams"
    
    gpp = GPParams()
    
    assert gpp.n_mean == 0
    assert gpp.n_corr == 1
    assert gpp.nugget_type == "fit"
    assert gpp.nugget is None
    assert gpp._cov is None
    assert gpp.mean.shape == (0,)
    assert gpp._data is None

    gpp = GPParams(n_mean=2, n_corr=3, nugget="pivot")
    
    assert gpp.n_mean == 2
    assert gpp.n_corr == 3
    assert gpp._cov is None
    assert gpp.nugget_type == "pivot"
    assert gpp.nugget is None
    assert gpp._data is None
    assert gpp.mean is None
    
    gpp = GPParams(n_mean=2, n_corr=3, nugget=1.e-5)
    
    assert_allclose(gpp.nugget, 1.e-5)
    assert gpp.nugget_type == "fixed"

def test_GPParams_init_failures():
    "Test failures of GPParams init method"
    
    with pytest.raises(AssertionError):
        GPParams(n_mean=-1)

    with pytest.raises(AssertionError):
        GPParams(n_corr=-1)
        
    with pytest.raises(ValueError):
        GPParams(nugget="blah")
        
    with pytest.raises(ValueError):
        GPParams(nugget=-1.)
    
    with pytest.raises(TypeError):
        GPParams(nugget=[])

def test_GPParams_n_params():
    "Test the n_params property of GPParams"
    
    gpp = GPParams()
    
    assert gpp.n_params == 3
    
    gpp = GPParams(n_mean=2, n_corr=3, nugget="pivot")
    
    assert gpp.n_params == 4
    
    gpp = GPParams(n_mean=2, n_corr=3, nugget=1.e-5)
    
    assert gpp.n_params == 4

def test_GPParams_mean():
    "Test the mean functionality of GPParams"
    
    gpp = GPParams()
    
    assert gpp.n_mean == 0
    assert len(gpp.mean) == 0
    
    gpp.mean = None
    
    assert len(gpp.mean) == 0
    
    with pytest.raises(AssertionError):
        gpp.mean = [1.]
    
    gpp = GPParams(n_mean=2)
    
    assert gpp.n_mean == 2
    assert gpp.mean is None
    
    gpp.mean = np.array([2., 3.])
    
    assert_allclose(gpp.mean, np.array([2., 3.]))

    with pytest.raises(AssertionError):
        gpp.mean = []
        
    gpp = GPParams(n_mean=1)
    
    assert gpp.mean is None
    
    gpp.mean = 1.
    
    assert_allclose(gpp.mean, np.ones(1))

def test_GPParams_corr():
    "Test the correlation functionality of GPParams"
    
    gpp = GPParams()
    
    assert gpp.n_corr == 1
    assert gpp.corr is None
    assert gpp.corr_raw is None
    
    with pytest.raises(ValueError):
        gpp.corr = 2.
    
    gpp._data = np.zeros(gpp.n_params)
    assert_allclose(gpp.corr, np.ones(1))
    assert_allclose(gpp.corr_raw, np.zeros(1))

    gpp.corr = 2.
    
    assert_allclose(gpp.corr, np.array([2.]))
    assert_allclose(gpp.corr_raw, -2.*np.log(2.))
    assert_allclose(gpp._data[0], -2.*np.log(2.))

    gpp.corr = np.array([3.])
    
    assert_allclose(gpp.corr, np.array([3.]))
    assert_allclose(gpp.corr_raw, -2.*np.log(3.))
    assert_allclose(gpp._data[0], -2.*np.log(3.))

    with pytest.raises(AssertionError):
        gpp.corr = np.array([2., 3.])
        
    with pytest.raises(AssertionError):
        gpp.corr = -1.
        
    gpp = GPParams(n_corr=3)
    gpp.set_data(np.array([2., 3., 2., 0., 0.]))
    
    assert gpp.n_corr == 3
    assert_allclose(gpp.corr, np.exp(-0.5*np.array([2., 3., 2.])))
    assert_allclose(gpp.corr_raw, np.array([2., 3., 2.]))
    assert_allclose(gpp._data[:3], np.array([2., 3., 2.]))

    gpp.corr = np.ones(3)
    
    assert_allclose(gpp.corr, np.ones(3))
    assert_allclose(gpp.corr_raw, np.zeros(3))
    assert_allclose(gpp._data[:3], np.zeros(3))

    with pytest.raises(AssertionError):
        gpp.corr = np.array([2., 3.])
        
    with pytest.raises(AssertionError):
        gpp.corr = np.array([-1., 3., 3.])

def test_GPParams_cov():
    "Test the covariance functionality of GPParams"
    
    gpp = GPParams()
    
    assert gpp.cov is None
    
    with pytest.raises(ValueError):
        gpp.cov = 2.
    
    gpp._data = np.zeros(gpp.n_params)
    assert_allclose(gpp.cov, np.ones(1))
    assert_allclose(gpp._data[-2], 0.)

    gpp.cov = 2.
    
    assert_allclose(gpp.cov, np.array([2.]))
    assert_allclose(gpp._data[-2], np.log(2.))

    gpp.cov = np.array([3.])
    
    assert_allclose(gpp.cov, np.array([3.]))
    assert_allclose(gpp._data[-2], np.log(3.))

    with pytest.raises(AssertionError):
        gpp.cov = np.array([2., 3.])
        
    with pytest.raises(AssertionError):
        gpp.cov = -1.

def test_GPParams_nugget():
    "Test the covariance functionality of GPParams"
    
    gpp = GPParams()
    
    assert gpp.nugget is None
    assert gpp.nugget_type == "fit"
    
    with pytest.raises(ValueError):
        gpp.nugget = 2.
    
    gpp.set_data(np.zeros(gpp.n_params))
    assert_allclose(gpp.nugget, np.ones(1))

    gpp.nugget = 2.
    
    assert_allclose(gpp.nugget, np.array([2.]))
    assert_allclose(gpp._data[-1], np.log(2.))

    gpp.nugget = np.array([3.])
    
    assert_allclose(gpp.nugget, np.array([3.]))
    assert_allclose(gpp._data[-1], np.log(3.))

    with pytest.raises(AssertionError):
        gpp.nugget = np.array([2., 3.])
        
    with pytest.raises(AssertionError):
        gpp.nugget = -1.
        
    gpp = GPParams(nugget="pivot")
    
    assert gpp.nugget is None
    assert gpp.nugget_type == "pivot"
    
    with pytest.raises(ValueError):
        gpp.nugget = 1.
        
    gpp = GPParams(nugget=1.)
    
    assert_allclose(gpp.nugget, 1.)
    assert gpp.nugget_type == "fixed"
    
    with pytest.raises(ValueError):
        gpp.nugget = 2.
        
    gpp = GPParams(nugget="adaptive")
    
    assert gpp.nugget is None
    assert gpp.nugget_type == "adaptive"
    
    gpp.nugget = 1.
    
    assert_allclose(gpp.nugget, 1.)
    
    gpp.nugget = None
    
    assert gpp.nugget is None

def test_GPParams_data():
    "Test the data getter and setter methods"
    
    gpp = GPParams()
    
    assert gpp.get_data() is None
    
    gpp._data = np.zeros(gpp.n_params)
    
    assert_allclose(gpp.get_data(), np.zeros(3))
    
    gpp.set_data(np.ones(3))
    
    assert_allclose(gpp._data, np.ones(3))
    
    with pytest.raises(AssertionError):
        gpp.set_data(np.ones(4))
        
    gpp = GPParams(n_mean=2, n_corr=3, nugget="adaptive")
    
    gpp.set_data(np.ones(4))
    
    assert_allclose(gpp._data, np.ones(4))
    
    gpp.set_data(None)
    
    assert gpp._data is None
    
    gpp.mean = np.array([2., 3.])
    
    gpp.set_data(None)
    
    assert gpp.mean is None
    assert gpp.cov is None
    assert gpp.nugget is None

def test_GPParams_same_shape():
    
    gpp = GPParams()
    
    assert gpp.same_shape(GPParams())
    assert not gpp.same_shape(GPParams(n_mean=1))
    assert not gpp.same_shape(GPParams(n_corr=2))
    assert not gpp.same_shape(GPParams(nugget=1.))
    assert gpp.same_shape(np.zeros(3))
    assert not gpp.same_shape(np.zeros(4))

    gpp = GPParams(n_mean=2, n_corr=3, nugget="adaptive")
    
    assert gpp.same_shape(GPParams(n_mean=2, n_corr=3, nugget="adaptive"))
    assert not gpp.same_shape(GPParams(n_mean=1, n_corr=3, nugget="adaptive"))
    assert not gpp.same_shape(GPParams(n_mean=2, n_corr=2, nugget="adaptive"))
    assert not gpp.same_shape(GPParams(n_mean=2, n_corr=3, nugget="fit"))
    assert gpp.same_shape(np.zeros(4))
    assert not gpp.same_shape(np.zeros(7))
    
    with pytest.raises(ValueError):
        gpp.same_shape([])

def test_GPParams_str():
    "Test the str method of GPParams doesn't raise an error"
    
    str(GPParams())
