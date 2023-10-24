import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..GaussianProcess import GaussianProcess, PredictResult
from ..MultiOutputGP import MultiOutputGP
from ..MultiOutputGP_GPU import MultiOutputGP_GPU
from ..LibGPGPU import gpu_usable
from ..MeanFunction import ConstantMean, LinearMean, MeanFunction
from ..Kernel import Matern52, SquaredExponential
from ..Priors import GPPriors
from scipy import linalg

GPU_NOT_FOUND_MSG = "A compatible GPU could not be found or the GPU library (libgpgpu) could not be loaded"

@pytest.fixture
def x():
    return np.array([[1., 2., 3.], [4., 5., 6.]])

@pytest.fixture
def y():
    return np.array([[2., 4.], [3., 5.]])

def test_MultiOutputGP_init(x, y):
    "Test function for correct functioning of the init method of GaussianProcess"

    gp = MultiOutputGP(x, y)
    assert_allclose(x, gp.inputs)
    assert_allclose(y, gp.targets)
    assert gp.D == 3
    assert gp.n == 2
    assert gp.n_emulators == 2
    assert gp.n_params == [4, 4]
    
    gp = MultiOutputGP(x, y, mean="1")
    gp = MultiOutputGP(x, y, mean=["1", "x[0]"])
    
    gp = MultiOutputGP(x, y, kernel="SquaredExponential")
    gp = MultiOutputGP(x, y, kernel=["SquaredExponential", "Matern52"])
    
    gp = MultiOutputGP(x, y, priors=GPPriors(n_corr=3, nugget_type="adaptive"))
    gp = MultiOutputGP(x, y, priors={"n_corr":3, "nugget_type":"adaptive"})
    gp = MultiOutputGP(x, y, priors=[None, GPPriors(n_corr=3, nugget_type="adaptive")])
    
    gp = MultiOutputGP(x, y, nugget=0.)
    gp = MultiOutputGP(x, y, nugget=[0., "adaptive"])


@pytest.mark.skipif(not gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_MultiOutputGP_GPU_init(x, y):
    "Test function for correct functioning of the init method of GaussianProcessGPU"

    gp = MultiOutputGP_GPU(x, y)
    assert gp.D == 3
    assert gp.n == 2
    assert gp.n_emulators == 2

@pytest.fixture
def dx():
    return 1.e-6

def test_MultiOutputGP_predict(x, y, dx):
    "test the predict method of GaussianProcess"

    gp = MultiOutputGP(x, y, nugget=0.)
    thetas = [np.ones((n,)) for n in gp.n_params]

    gp.fit(thetas)

    x_test = np.array([[2., 3., 4.]])

    mu, var, deriv = gp.predict(x_test)

    for i in range(2):

        K = np.exp(thetas[i][-1])*gp.emulators[i].kernel.kernel_f(x, x, thetas[i][:-1])
        Ktest = np.exp(thetas[i][-1])*gp.emulators[i].kernel.kernel_f(x_test, x, thetas[i][:-1])

        mu_expect = np.dot(Ktest, gp.emulators[i].Kinv_t)
        var_expect = np.exp(thetas[i][-1]) - np.diag(np.dot(Ktest, np.linalg.solve(K, Ktest.T)))

        assert_allclose(mu[i], mu_expect)
        assert_allclose(var[i], var_expect)
        
    gp.reset_fit_status()

    nugget = 1.
    gp = MultiOutputGP(x, y, nugget=nugget)
    theta0 = np.ones(gp.emulators[0].n_params)
    theta1 = np.ones(gp.emulators[1].n_params)

    gp.fit_emulator(0, theta0)
    gp.fit_emulator(1, theta1)

    x_test = np.array([[2., 3., 4.]])

    mu, var, deriv = gp.predict(x_test)

    for i, theta in zip(range(2), [theta0, theta1]):

        K = np.exp(theta[-1])*gp.emulators[i].kernel.kernel_f(x, x, theta[:-1]) + np.eye(gp.emulators[i].n)*nugget
        Ktest = np.exp(theta[-1])*gp.emulators[i].kernel.kernel_f(x_test, x, theta[:-1])

        var_expect = np.exp(theta[-1]) + nugget - np.diag(np.dot(Ktest, np.linalg.solve(K, Ktest.T)))

        assert_allclose(var[i], var_expect)

    mu, var, deriv = gp.predict(x_test, include_nugget=False)

    for i in range(2):

        K = np.exp(theta[-1])*gp.emulators[i].kernel.kernel_f(x, x, theta[:-1]) + np.eye(gp.emulators[i].n)*nugget
        Ktest = np.exp(theta[-1])*gp.emulators[i].kernel.kernel_f(x_test, x, theta[:-1])

        var_expect = np.exp(theta[-1]) - np.diag(np.dot(Ktest, np.linalg.solve(K, Ktest.T)))

        assert_allclose(var[i], var_expect)

    # test behavior if not all emulators are fit

    gp.emulators[1].theta = None

    mu, var, deriv = gp.predict(x_test, allow_not_fit=True)
    assert np.all(np.isnan(mu[1]))
    assert np.all(np.isnan(var[1]))

    mu, var, deriv = gp.predict(x_test, unc=False, deriv=False,
                                allow_not_fit=True)
    assert np.all(np.isnan(mu[1]))
    assert var is None
    assert deriv is None

    with pytest.raises(ValueError):
        gp.predict(x_test)


@pytest.mark.skipif(not gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_MultiOutputGP_GPU_predict(x, y, dx):
    "test the predict method of GaussianProcess"

    gp = MultiOutputGP_GPU(x, y, nugget=0.)
    theta = np.ones(gp.n_params[0])

    gp.fit_emulator(0,theta)
    gp.fit_emulator(1,theta)

    x_test = np.array([[2., 3., 4.]])

    mu, var, deriv = gp.predict(x_test)

    for i in range(2):

        K = SquaredExponential().kernel_f(x, x, theta[:-1])
        Ktest = SquaredExponential().kernel_f(x_test, x, theta[:-1])

        var_expect = np.exp(theta[-2]) - np.diag(np.dot(Ktest, np.linalg.solve(K, Ktest.T)))

        D = gp.D

        assert_allclose(var[i], var_expect, atol=1e-3)

    nugget = 1.
    gp = MultiOutputGP_GPU(x, y, nugget=nugget)
    theta = np.ones(gp.n_params[0])

    gp.fit_emulator(0,theta)
    gp.fit_emulator(1,theta)

    x_test = np.array([[2., 3., 4.]])

    mu, var, deriv = gp.predict(x_test, include_nugget=False)

    for i in range(2):

        K = SquaredExponential().kernel_f(x, x, theta[:-1]) + np.eye(gp.n)*nugget
        Ktest = SquaredExponential().kernel_f(x_test, x, theta[:-1])

        var_expect = np.exp(theta[-2]) - np.diag(np.dot(Ktest, np.linalg.solve(K, Ktest.T)))

        assert_allclose(var[i], var_expect, atol=1e-3)

    gp.reset_fit_status()

    mu, var, deriv = gp.predict(x_test, allow_not_fit=True)
    assert np.all(np.isnan(mu[1]))
    assert np.all(np.isnan(var[1]))
    assert np.all(np.isnan(deriv[1]))

    mu, var, deriv = gp.predict(x_test, unc=False, deriv=False, allow_not_fit=True)
    assert np.all(np.isnan(mu[1]))
    assert np.all(np.isnan(var[1]))
    assert np.all(np.isnan(deriv[1]))

    with pytest.raises(ValueError):
        gp.predict(x_test, allow_not_fit=False)


def test_MultiOutputGP_check(x, y):
    """test the methods of MultiOutputGP that extracts GPs that have or
    have not been fit (or their indices)
    """
    gp = MultiOutputGP(x, y, nugget=0.)
    theta = np.ones(gp.emulators[0].n_params)

    assert gp.get_indices_fit() == []
    assert gp.get_indices_not_fit() == [0, 1]
    assert len(gp.get_emulators_fit()) == 0
    assert len(gp.get_emulators_not_fit()) == 2

    gp.emulators[0].fit(theta)

    assert gp.get_indices_fit() == [0]
    assert gp.get_indices_not_fit() == [1]
    assert len(gp.get_emulators_fit()) == 1
    assert len(gp.get_emulators_not_fit()) == 1

    gp.emulators[1].fit(theta)

    assert gp.get_indices_fit() == [0, 1]
    assert gp.get_indices_not_fit() == []
    assert len(gp.get_emulators_fit()) == 2
    assert len(gp.get_emulators_not_fit()) == 0


@pytest.mark.skipif(not gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_MultiOutputGP_GPU_check(x, y):
    """test the methods of MultiOutputGP that extracts GPs that have or
    have not been fit (or their indices)
    """

    gp = MultiOutputGP_GPU(x, y, nugget=0.)
    theta = np.ones(gp.n_params[0])

    assert gp.get_indices_fit() == []
    assert gp.get_indices_not_fit() == [0, 1]

    gp.fit_emulator(0,theta)
    assert gp.get_indices_fit() == [0]
    assert gp.get_indices_not_fit() == [1]

    gp.fit_emulator(1,theta) 
    assert gp.get_indices_fit() == [0, 1]
    assert gp.get_indices_not_fit() == []
