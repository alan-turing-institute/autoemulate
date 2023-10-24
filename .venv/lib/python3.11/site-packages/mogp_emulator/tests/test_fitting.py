import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..GaussianProcess import GaussianProcess
from ..GaussianProcessGPU import GaussianProcessGPU
from ..LibGPGPU import gpu_usable
from ..MultiOutputGP import MultiOutputGP
from ..fitting import fit_GP_MAP, _fit_single_GP_MAP, _fit_MOGP_MAP

GPU_NOT_FOUND_MSG = "A compatible GPU could not be found or the GPU library (libgpgpu) could not be loaded"

def minimize_mock(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
    return {"x": np.array([1.6, -2.1, -0.8]),
            "fun": 0.}

def test_fit_GP_MAP(monkeypatch):
    "test the fit_GP_MAP function"

    monkeypatch.setattr("mogp_emulator.fitting.minimize", minimize_mock)

    # test correct basic functioning

    x = np.linspace(0., 1.)
    y = x**2

    gp = GaussianProcess(x, y, nugget="fit")

    theta_exp = np.array([ 1.6, -2.1 , -0.8])
    logpost_exp = gp.logposterior(theta_exp)

    gp = fit_GP_MAP(gp)

    assert isinstance(gp, GaussianProcess)
    assert_allclose(gp.theta.get_data(), theta_exp)
    assert_allclose(gp.current_logpost, logpost_exp)

    # same test, but pass args and kwargs rather than gp

    gp = fit_GP_MAP(x, y, mean="0", nugget="fit", method="L-BFGS-B")
    assert isinstance(gp, GaussianProcess)
    assert_allclose(gp.theta.get_data(), theta_exp)
    assert_allclose(gp.current_logpost, logpost_exp)


@pytest.mark.skipif(not gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_fit_GP_MAP_GPU():
    "test the fit_GP_MAP function.  Provide expected theta as theta0 to avoid numerical errors"

    # test correct basic functioning

    x = np.linspace(0., 1.)
    y = x**2

    gp = GaussianProcessGPU(x, y, nugget="fit")

    theta_exp = np.array([ 1.6, -2.1 , -0.8])
    logpost_exp = gp.logposterior(theta_exp)

    gp = fit_GP_MAP(gp, theta0=theta_exp)

    # remove value checks as machine precision means they might fail
    # instead test basic functionality
    assert isinstance(gp, GaussianProcessGPU)
    assert gp.theta.data_has_been_set()
    assert gp.theta.get_data().shape ==  theta_exp.shape

def test_fit_GP_MAP_failures():
    "test failures of fit_GP_MAP"

    x = np.linspace(0., 1.)
    y = x**2

    gp = GaussianProcess(x, y, nugget="fit")

    # minimization fails

    with pytest.raises(RuntimeError):
        fit_GP_MAP(gp, n_tries=1, theta0=-10000.*np.ones(3))

    gp = GaussianProcess(x, y, nugget=0.)

    with pytest.raises(RuntimeError):
        fit_GP_MAP(gp, theta0=np.array([0., 0.]), n_tries=1)

    with pytest.raises(RuntimeError):
        fit_GP_MAP(gp, theta0 = np.array([800., 0.]), n_tries=1)

    # bad inputs

    with pytest.raises(TypeError):
        fit_GP_MAP(x)

    with pytest.raises(TypeError):
        fit_GP_MAP()

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, n_tries=-1)

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, theta0=np.ones(1))

@pytest.mark.skipif(not gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_fit_GP_MAP_GPU_failures():
    "test failures of fit_GP_MAP using GaussianProcessGPU"

    x = np.linspace(0., 1.)
    y = x**2

    gp = GaussianProcessGPU(x, y)

    # minimization fails

    with pytest.raises(RuntimeError):
        fit_GP_MAP(gp, n_tries=1, theta0=-1000000.*np.ones(3))

    gp = GaussianProcessGPU(x, y, nugget=0.)

    with pytest.raises(RuntimeError):
        fit_GP_MAP(gp, n_tries=1)

    with pytest.raises(RuntimeError):
        fit_GP_MAP(gp, theta0 = np.array([800., 0., 0.]), n_tries=1)

    # bad inputs

    with pytest.raises(TypeError):
        fit_GP_MAP(x)

    with pytest.raises(TypeError):
        fit_GP_MAP()

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, n_tries=-1)

    with pytest.raises(RuntimeError):
        fit_GP_MAP(gp, theta0=np.ones(1))

def test_fit_GP_MAP_MOGP():
    "test the fit_GP_MAP function with multiple outputs"

    x = np.linspace(0., 1.)
    y = np.zeros((2, 50))
    y[0] = x**2
    y[1] = 2. + x**3

    gp = MultiOutputGP(x, y, nugget="fit")

    np.random.seed(4335)

    gp = fit_GP_MAP(gp)

    assert isinstance(gp, MultiOutputGP)

    np.random.seed(4335)

    # same test, but pass args and kwargs rather than gp

    gp = fit_GP_MAP(x, y, mean="0", nugget="fit", method="L-BFGS-B", processes=1)
    assert isinstance(gp, MultiOutputGP)

    # pass processes argument

    gp = fit_GP_MAP(x, y, mean="0", nugget="fit", method="L-BFGS-B", processes=1)
    assert isinstance(gp, MultiOutputGP)

    # pass various theta0 arguments

    gp = fit_GP_MAP(x, y, nugget="fit", theta0=np.zeros(3))

    gp = fit_GP_MAP(x, y, nugget="fit", theta0=np.zeros((2, 3)))

    gp = fit_GP_MAP(x, y, nugget="fit", theta0=[None, np.zeros(3)])

    # test re-fitting

    gp = MultiOutputGP(x, y, nugget="fit")
    gp.emulators[0].fit(np.ones(3))

    gp = fit_GP_MAP(gp, theta0=np.zeros(3), n_tries=1, refit=False)
    assert_allclose(gp.emulators[0].theta.get_data(), np.ones(3))
    assert not gp.emulators[1].theta is None

    gp = fit_GP_MAP(gp, theta0=np.zeros(3), n_tries=1, refit=True)
    assert not np.allclose(gp.emulators[0].theta.get_data(), np.ones(3))

def test_fit_GP_MAP_MOGP_failures():
    "test situations where mogp fitting should fail"

    x = np.linspace(0., 1.)
    y = np.zeros((2, 50))
    y[0] = x**2
    y[1] = 2. + x**3

    gp = MultiOutputGP(x, y, nugget="fit")

    # minimization fails

    gp = fit_GP_MAP(gp, n_tries=1, theta0=-10000.*np.ones(3))
    assert gp.get_indices_not_fit() == [0, 1]

    with pytest.raises(RuntimeError):
        gp = fit_GP_MAP(gp, n_tries=1, theta0=-10000.*np.ones(3),
                        skip_failures=False, refit=True)

    gp = MultiOutputGP(x, y, nugget=0.)

    gp = fit_GP_MAP(gp, theta0=np.array([0., 0.]), n_tries=1)
    assert gp.get_indices_not_fit() == [0, 1]

    with pytest.raises(RuntimeError):
        gp = fit_GP_MAP(gp, theta0=np.array([0., 0.]), n_tries=1,
                        skip_failures=False, refit=True)

    gp = fit_GP_MAP(gp, theta0 = np.array([800., 0.]), n_tries=1)
    assert gp.get_indices_not_fit() == [0, 1]

    with pytest.raises(RuntimeError):
        gp = fit_GP_MAP(gp, theta0 = np.array([800., 0.]), n_tries=1,
                        skip_failures=False, refit=True)

    # bad inputs

    with pytest.raises(TypeError):
        fit_GP_MAP(x)

    with pytest.raises(TypeError):
        fit_GP_MAP()

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, n_tries=-1)

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, theta0=np.ones(1))

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, theta0=np.zeros((3, 3)))

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, theta0=np.zeros((2, 1)))

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, theta0=np.zeros((1, 1, 1)))

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, theta0=[None, None, None])

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, theta0=[np.zeros(1), np.zeros(2)], processes=1)

def test_fit_single_GP_MAP(monkeypatch):
    "test the method to run the minimization algorithm on a GP class"

    monkeypatch.setattr("mogp_emulator.fitting.minimize", minimize_mock)

    x = np.linspace(0., 1.)
    y = x**2

    gp = GaussianProcess(x, y, nugget="fit")

    theta_exp = np.array([ 1.6, -2.1 , -0.8])
    logpost_exp = gp.logposterior(theta_exp)

    gp = _fit_single_GP_MAP(gp)

    assert isinstance(gp, GaussianProcess)
    assert_allclose(gp.theta.get_data(), theta_exp)
    assert_allclose(gp.current_logpost, logpost_exp)

def test_fit_single_GP_MAP_failures():
    "test situation where fitting one emulator should fail"

    x = np.linspace(0., 1.)
    y = x**2

    gp = GaussianProcess(x, y, nugget="fit")

    # minimization fails

    gp = _fit_single_GP_MAP(gp, n_tries=1, theta0=-10000.*np.ones(3))
    assert gp.theta.get_data() is None

    gp = GaussianProcess(x, y, nugget=0.)

    gp = _fit_single_GP_MAP(gp, theta0 = np.array([0., 0.]), n_tries=1)
    assert gp.theta.get_data() is None

    gp =_fit_single_GP_MAP(gp, theta0 = np.array([800., 0.]), n_tries=1)
    assert gp.theta.get_data() is None

    # bad inputs

    with pytest.raises(AssertionError):
        _fit_single_GP_MAP(x)

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, n_tries=-1)

    with pytest.raises(AssertionError):
        fit_GP_MAP(gp, theta0=np.ones(1))

def test_fit_MOGP_MAP():
    "test the fit_GP_MAP function with multiple outputs"

    x = np.linspace(0., 1.)
    y = np.zeros((2, 50))
    y[0] = x**2
    y[1] = 2. + x**3

    gp = MultiOutputGP(x, y, nugget="fit")

    np.random.seed(4335)

    gp = _fit_MOGP_MAP(gp, nugget="fit")

    assert isinstance(gp, MultiOutputGP)

    # pass processes argument

    np.random.seed(4335)

    gp = _fit_MOGP_MAP(gp, processes=1)
    assert isinstance(gp, MultiOutputGP)

    # pass various theta0 arguments

    gp = _fit_MOGP_MAP(gp, theta0=np.zeros(3))

    gp = _fit_MOGP_MAP(gp, theta0=np.zeros((2, 3)))

    gp = _fit_MOGP_MAP(gp, theta0=[None, np.zeros(3)])

    # test re-fitting

    gp = MultiOutputGP(x, y, nugget="fit")
    gp.emulators[0].fit(np.ones(3))

    gp = _fit_MOGP_MAP(gp, theta0=np.zeros(3), n_tries=1, refit=False)
    assert_allclose(gp.emulators[0].theta.get_data(), np.ones(3))
    assert not gp.emulators[1].theta.get_data() is None

    gp = _fit_MOGP_MAP(gp, theta0=np.zeros(3), n_tries=1, refit=True)
    assert not np.allclose(gp.emulators[0].theta.get_data(), np.ones(3))

def test_fit_MOGP_MAP_failures():
    "test situations where fitting should fail"

    x = np.linspace(0., 1.)
    y = np.zeros((2, 50))
    y[0] = x**2
    y[1] = 2. + x**3

    gp = MultiOutputGP(x, y, nugget="fit")

    # minimization fails

    gp = _fit_MOGP_MAP(gp, n_tries=1, theta0=-10000.*np.ones(3))
    assert gp.get_indices_not_fit() == [0, 1]

    gp = MultiOutputGP(x, y, nugget=0.)

    gp = _fit_MOGP_MAP(gp, theta0=np.array([0., 0.]), n_tries=1)
    assert gp.get_indices_not_fit() == [0, 1]

    gp = _fit_MOGP_MAP(gp, theta0 = np.array([800., 0.]), n_tries=1)
    assert gp.get_indices_not_fit() == [0, 1]

    # bad inputs

    with pytest.raises(AssertionError):
        _fit_MOGP_MAP(x)

    with pytest.raises(AssertionError):
        _fit_MOGP_MAP(gp, n_tries=-1)

    with pytest.raises(AssertionError):
        _fit_MOGP_MAP(gp, theta0=np.ones(1))

    with pytest.raises(AssertionError):
        _fit_MOGP_MAP(gp, theta0=np.zeros((3, 3)))

    with pytest.raises(AssertionError):
        _fit_MOGP_MAP(gp, theta0=np.zeros((2, 1)))

    with pytest.raises(AssertionError):
        _fit_MOGP_MAP(gp, theta0=np.zeros((1, 1, 1)))

    with pytest.raises(AssertionError):
        _fit_MOGP_MAP(gp, theta0=[None, None, None])

    with pytest.raises(AssertionError):
        _fit_MOGP_MAP(gp, theta0=[np.zeros(1), None], processes=1)

    with pytest.raises(AssertionError):
        _fit_MOGP_MAP(gp, processes=-1)
