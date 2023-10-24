from tempfile import TemporaryFile
import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..LibGPGPU import gpu_usable
from ..GaussianProcess import GaussianProcess, PredictResult
from ..GaussianProcessGPU import GaussianProcessGPU
from ..Kernel import SquaredExponential, Matern52
from ..GPParams import GPParams
from ..Priors import ( GPPriors, NormalPrior, LogNormalPrior, GammaPrior,
                       InvGammaPrior, WeakPrior, MeanPriors )
from scipy import linalg

GPU_NOT_FOUND_MSG = "A compatible GPU could not be found or the GPU library (libgpgpu) could not be loaded"

@pytest.fixture
def x():
    return np.array([[1., 2., 3.], [4., 5., 6.]])

@pytest.fixture
def y():
    return np.array([2., 4.])

def test_GaussianProcess_init(x, y):
    "Test function for correct functioning of the init method of GaussianProcess"

    gp = GaussianProcess(x, y)
    assert_allclose(x, gp.inputs)
    assert_allclose(y, gp.targets)
    assert gp.D == 3
    assert gp.n == 2
    assert gp.nugget == None

    gp = GaussianProcess(y, y)
    assert gp.inputs.shape == (2, 1)

    gp = GaussianProcess(x, y, nugget=1.e-12)
    assert_allclose(gp.nugget, 1.e-12)

    gp = GaussianProcess(x, y, mean="1", kernel=Matern52(), nugget="fit")
    assert gp._dm.shape == (2, 1)

    gp = GaussianProcess(x, y, kernel="SquaredExponential")
    assert isinstance(gp.kernel, SquaredExponential)

    gp = GaussianProcess(x, y, kernel="Matern52")
    assert isinstance(gp.kernel, Matern52)

    gp = GaussianProcess(x, y, mean="0")
    assert gp._dm.shape == (2, 0)

    gp = GaussianProcess(x, y, mean="-1")
    assert gp._dm.shape == (2, 0)

    gp = GaussianProcess(x, y, mean="x[0]")
    assert gp._dm.shape == (2, 2)

    gp = GaussianProcess(x, y, mean="y ~ x[0]")
    assert gp._dm.shape == (2, 2)


@pytest.mark.skipif(not gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_GaussianProcessGPU_init(x, y):
    "Test function for correct functioning of the init method of GaussianProcess"

    gp = GaussianProcessGPU(x, y)
    assert_allclose(x, gp.inputs)
    assert_allclose(y, gp.targets)
    assert gp.D == 3
    assert gp.n == 2
    assert gp.nugget == 0.
    assert gp.nugget_type == "adaptive"
    gp = GaussianProcessGPU(y, y)
    assert gp.inputs.shape == (2, 1)

    gp = GaussianProcessGPU(x, y, nugget=1.e-12)
    assert_allclose(gp.nugget, 1.e-12)
    from ..LibGPGPU import kernel_type
    gp = GaussianProcessGPU(x, y, kernel="SquaredExponential")
    assert isinstance(gp.kernel_type, kernel_type)
    assert gp.kernel_type is kernel_type.SquaredExponential
    assert isinstance(gp.kernel, SquaredExponential)
    # test with mean function
    gp = GaussianProcessGPU(x, y, mean="x[0]")
    assert gp.mean.get_n_params() == 2
    # mean function with LHS
    gp = GaussianProcessGPU(x, y, mean="y ~ x[0]")
    assert gp.mean.get_n_params() == 2


def test_GP_init_failures(x, y):
    "Tests that GaussianProcess fails correctly with bad inputs"

    with pytest.raises(AssertionError):
        gp = GaussianProcess(np.ones((2, 2, 2)), y)

    with pytest.raises(AssertionError):
        gp = GaussianProcess(x, x)

    with pytest.raises(AssertionError):
        gp = GaussianProcess(np.ones((2, 3)), np.ones(3))

    with pytest.raises(ValueError):
        gp = GaussianProcess(x, y, mean=1.)

    with pytest.raises(ValueError):
        gp = GaussianProcess(x, y, mean="x[6]")

    with pytest.raises(ValueError):
        gp = GaussianProcess(x, y, kernel="blah")

    with pytest.raises(ValueError):
        gp = GaussianProcess(x, y, kernel=1)

    with pytest.raises(ValueError):
        gp = GaussianProcess(x, y, nugget="a")


@pytest.mark.skipif(not gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_GPGPU_init_failures(x, y):
    "Tests that GaussianProcessGPU fails correctly with bad inputs"

    with pytest.raises(AssertionError):
        gp = GaussianProcessGPU(np.ones((2, 2, 2)), y)

    with pytest.raises(AssertionError):
        gp = GaussianProcessGPU(x, x)

    with pytest.raises(AssertionError):
        gp = GaussianProcessGPU(np.ones((2, 3)), np.ones(3))

    with pytest.raises(ValueError):
        gp = GaussianProcessGPU(x, y, mean=1)

    with pytest.raises(ValueError):
        gp = GaussianProcessGPU(x, y, kernel="blah")

    with pytest.raises(ValueError):
        gp = GaussianProcessGPU(x, y, kernel=1)

    with pytest.raises(ValueError):
        gp = GaussianProcessGPU(x, y, nugget="a")

def test_GaussianProcess_n_params(x, y):
    "test the get_n_params method of GaussianProcess"

    gp = GaussianProcess(x, y)
    assert gp.n_params == x.shape[1] + 1

    gp = GaussianProcess(x, y, mean="x[0]")
    assert gp.n_params == x.shape[1] + 1

    gp = GaussianProcess(x, y, kernel="UniformSqExp")
    assert gp.n_params == 2

    gp = GaussianProcess(x, y, nugget="pivot")
    assert gp.n_params == x.shape[1] + 1

@pytest.mark.skipif(not gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_GaussianProcessGPU_n_params(x, y):
    "test the get_n_params method of GaussianProcessGPU"

    gp = GaussianProcessGPU(x, y)
    assert gp.n_params == x.shape[1] + 1

def test_GaussianProcess_nugget(x, y):
    "Tests the get_nugget method of GaussianProcess"

    gp = GaussianProcess(x, y)
    assert gp.nugget is None
    assert gp.nugget_type == "adaptive"

    gp = GaussianProcess(x, y, nugget="fit")
    assert gp.nugget is None
    assert gp.nugget_type == "fit"

    gp = GaussianProcess(x, y, nugget=1.e-4)
    assert_allclose(gp.nugget, 1.e-4)
    assert gp.nugget_type == "fixed"

    gp = GaussianProcess(x, y, nugget=0.)
    assert_allclose(gp.nugget, 0.)
    assert gp.nugget_type == "fixed"

    gp = GaussianProcess(x, y, nugget="pivot")
    assert gp.nugget is None
    assert gp.nugget_type == "pivot"

    gp = GaussianProcess(x, y)

    # nugget and nugget_type have no setter methods, check error is raises
    with pytest.raises(AttributeError):
        gp.nugget = 1.

    with pytest.raises(AttributeError):
        gp.nugget_type = "fit"

@pytest.mark.skipif(not gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_GaussianProcessGPU_nugget(x, y):
    "Tests the get_nugget method of GaussianProcessGPU"

    gp = GaussianProcessGPU(x, y)
    assert gp.nugget == 0.
    assert gp.nugget_type == "adaptive"

    gp.nugget = "fit"
    assert gp.nugget == 1.
    assert gp.nugget_type == "fit"

    gp.nugget = 1.
    assert_allclose(gp.nugget, 1.)
    assert gp.nugget_type == "fixed"

    gp.nugget = 0
    assert_allclose(gp.nugget, 0.)
    assert gp.nugget_type == "fixed"

    with pytest.raises(TypeError):
        gp.nugget = [1]

    with pytest.raises(ValueError):
        gp.nugget = "blah"

    with pytest.raises(ValueError):
        gp.nugget = -1.

@pytest.mark.parametrize("mean,nugget,sn", [(None, 0., 1.), (None, "adaptive", 0.),
                                           (None, "pivot", 0.),
                                           ("x[0]", "fit", np.log(1.e-6))])
def test_GaussianProcess_theta(x, y, mean, nugget, sn):
    "test the theta property of GaussianProcess (effectively the same as fit)"

    if isinstance(nugget, float):
        nugget_type = "fixed"
    else:
        nugget_type = nugget

    gp = GaussianProcess(x, y, mean=mean, nugget=nugget, priors=GPPriors(n_corr=3, nugget_type=nugget_type))

    with pytest.raises(AssertionError):
        gp.theta = np.ones(gp.n_params + 1)

    theta = np.ones(gp.n_params)
    if nugget == "fit":
        theta[-1] = sn

    gp.theta = theta

    if nugget == "adaptive" or nugget == 0.:
        assert gp.nugget == 0.
        noise = 0.
    elif nugget == "pivot":
        assert gp.nugget is None
        noise = 0.
    else:
        assert_allclose(gp.nugget, np.exp(sn))
        noise = np.exp(sn)*np.eye(x.shape[0])
    K = np.exp(theta[gp.D])*gp.kernel.kernel_f(x, x, theta[:gp.D]) + noise

    L_expect = np.linalg.cholesky(K)
    Kinv_t_expect = np.linalg.solve(K, y)
    A = np.dot(gp._dm.T, np.linalg.solve(K, gp._dm))
    LA_expect = np.linalg.cholesky(A)

    logpost_expect = 0.5*(np.log(np.linalg.det(K)) +
                          np.log(np.linalg.det(A)) +
                          np.dot(y, Kinv_t_expect) -
                          np.linalg.multi_dot([Kinv_t_expect, gp._dm,
                                               np.linalg.solve(A, np.dot(gp._dm.T, Kinv_t_expect))]) +
                          (gp.n - gp.n_mean)*np.log(2.*np.pi))

    mean_expect = np.linalg.solve(A, np.dot(gp._dm.T, Kinv_t_expect))

    assert_allclose(L_expect, gp.Kinv.L)
    assert_allclose(Kinv_t_expect, gp.Kinv_t)
    assert_allclose(LA_expect, gp.Ainv.L)
    assert_allclose(mean_expect, gp.theta.mean)
    assert_allclose(logpost_expect, gp.current_logpost)

def test_GaussianProcess_theta_scipy(x, y):
    "test the theta property of GaussianProcess by comparing with scipy implementation"

    mean = "x[0]"
    nugget = "fit"
    nugget_type = "fit"
    sn = np.log(1.e-6)

    gp = GaussianProcess(x, y, mean=mean, nugget=nugget,
                         priors=GPPriors(mean=MeanPriors(mean=np.ones(2), cov=np.eye(2)),
                                         n_corr=3, nugget_type=nugget_type))

    with pytest.raises(AssertionError):
        gp.theta = np.ones(gp.n_params + 1)

    theta = np.ones(gp.n_params)
    if nugget == "fit":
        theta[-1] = sn

    gp.theta = theta

    noise = np.exp(sn)*np.eye(x.shape[0])
    K = np.exp(theta[gp.D])*gp.kernel.kernel_f(x, x, theta[:gp.D])
    K += noise
    K += np.dot(gp._dm, np.dot(gp.priors.mean.cov, gp._dm.T))

    m = np.dot(gp._dm, gp.priors.mean.mean)

    from scipy.stats import multivariate_normal

    logpost_expect = -multivariate_normal.logpdf(y, mean=m, cov=K)

    assert_allclose(logpost_expect, gp.current_logpost)

def test_GaussianProcess_theta_GPParams(x, y):
    "test that we can set parameters using a GPParams object"

    gp = GaussianProcess(x, y)

    gpp = GPParams(n_corr=3, nugget="adaptive")
    gpp.set_data(np.ones(4))

    gp.theta = gpp

    assert_allclose(gp.theta.get_data(), np.ones(4))
    assert_allclose(gp.theta.nugget, 0.)

    with pytest.raises(AssertionError):
        gp.theta = GPParams(n_corr=3, nugget="fit")

    with pytest.raises(AssertionError):
        gp.theta = GPParams(n_corr=1, nugget="adaptive")

@pytest.mark.skipif(not gpu_usable(), reason=GPU_NOT_FOUND_MSG)
@pytest.mark.parametrize("mean,nugget,sn", [(None, 0., 1.), (None, "adaptive", 0.),
                                           ("x[0]", "fit", np.log(1.e-6))])
def test_GaussianProcessGPU_theta(x, y, mean, nugget, sn):
    "test the theta property of GaussianProcess (effectively the same as fit)"

    if isinstance(nugget, float):
        nugget_type = "fixed"
    else:
        nugget_type = nugget

    gp = GaussianProcessGPU(x, y, mean=mean, nugget=nugget, priors=GPPriors(n_corr=3, nugget_type=nugget_type))

    with pytest.raises(RuntimeError):
        gp.theta = np.ones(gp.n_params + 1)

    theta = np.ones(gp.n_params)
    if nugget == "fit":
        theta[-1] = sn

    gp.theta = theta

    if nugget == "adaptive" or nugget == 0.:
        assert gp.nugget == 0.
        noise = 0.
    else:
        assert_allclose(gp.nugget, np.exp(sn))
        noise = np.exp(sn)*np.eye(x.shape[0])

 #   Q = np.exp(theta[switch + gp.D])*gp.kernel.kernel_f(x, x, theta[switch:(switch + gp.D)]) + noise
#    ym = y - np.dot(gp._dm, theta[:switch]) TBD for GPU

    Q = gp.kernel.kernel_f(x, x, theta[:gp.D]) + noise

    L_expect = np.linalg.cholesky(Q)
#    invQt_expect = np.linalg.solve(Q, ym)  TBD for GPU
#    logpost_expect = 0.5*(np.log(np.linalg.det(Q)) +
#                          np.dot(ym, invQt_expect) +
#                          gp.n*np.log(2.*np.pi))

 #   assert_allclose(L_expect, gp.L)
 #   assert_allclose(invQt_expect, gp.Kinv_t)
 #   assert_allclose(logpost_expect, gp.current_logpost)

@pytest.mark.skipif(not gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_GaussianProcess_theta_GPParams(x, y):
    "test that we can set parameters using a GPParams object"

    gp = GaussianProcess(x, y)

    gpp = GPParams(n_corr=3, nugget="adaptive")
    gpp.set_data(np.ones(4))

    gp.theta = gpp

    assert_allclose(gp.theta.get_data(), np.ones(4))
    assert_allclose(gp.theta.nugget, 0.)

    with pytest.raises(AssertionError):
        gp.theta = GPParams(n_corr=3, nugget="fit")

    with pytest.raises(AssertionError):
        gp.theta = GPParams(n_corr=1, nugget="adaptive")

def test_GaussianProcess_theta_pivot():
    "test that pivoting works as expected"

    # input arrays are re-ordered such that pivoting should re-order the second to match the first

    x1 = np.array([1., 4., 2.])
    x2 = np.array([1., 2., 4.])
    y1 = np.array([1., 1., 2.])
    y2 = np.array([1., 2., 1.])

    gp1 = GaussianProcess(x1, y1, nugget=0.)
    gp2 = GaussianProcess(x2, y2, nugget="pivot")

    gp1.theta = np.zeros(2)
    gp2.theta = np.zeros(2)

    assert_allclose(gp1.Kinv.L, gp2.Kinv.L)
    assert_allclose(gp1.Kinv_t, gp2.Kinv_t[gp2.Kinv.P])
    assert np.array_equal(gp2.Kinv.P, [0, 2, 1])

def test_GaussianProcess_priors_property(x, y):
    "test that priors are set properly"

    gp = GaussianProcess(x, y)

    assert isinstance(gp.priors.mean, MeanPriors)
    assert gp.priors.mean.mean is None
    assert gp.priors.mean.cov is None
    for p in gp.priors.corr:
        assert isinstance(p, WeakPrior)
    assert isinstance(gp.priors.cov, WeakPrior)
    assert gp.priors.nugget is None

    gp = GaussianProcess(x, y, priors=None)

    assert isinstance(gp.priors.mean, MeanPriors)
    assert gp.priors.mean.mean is None
    assert gp.priors.mean.cov is None
    for p in gp.priors.corr:
        assert isinstance(p, WeakPrior)
    assert isinstance(gp.priors.cov, WeakPrior)
    assert gp.priors.nugget is None

    priors = GPPriors(n_corr=3, nugget_type="adaptive")
    gp = GaussianProcess(x, y, priors=priors)

    assert isinstance(gp.priors.mean, MeanPriors)
    assert gp.priors.mean.mean is None
    assert gp.priors.mean.cov is None
    for p in gp.priors.corr:
        assert isinstance(p, WeakPrior)
    assert isinstance(gp.priors.cov, WeakPrior)
    assert gp.priors.nugget is None

    priors = {"mean": None, "corr": [LogNormalPrior(2., 2.), WeakPrior(), WeakPrior()], "cov": GammaPrior(3., 1.),
              "nugget_type": "adaptive"}
    gp = GaussianProcess(x, y, priors=priors)

    assert isinstance(gp.priors.mean, MeanPriors)
    assert gp.priors.mean.mean is None
    assert gp.priors.mean.cov is None
    assert isinstance(gp.priors.corr[0], LogNormalPrior)
    assert isinstance(gp.priors.corr[1], WeakPrior)
    assert isinstance(gp.priors.corr[2], WeakPrior)
    assert isinstance(gp.priors.cov, GammaPrior)
    assert gp.priors.nugget is None

    priors = {"mean": None, "corr": [LogNormalPrior(2., 2.), WeakPrior(), WeakPrior()], "cov": GammaPrior(3., 1.),
              "nugget_type": "adaptive", "nugget": InvGammaPrior(3., 3.)}
    gp = GaussianProcess(x, y, priors=priors)

    assert gp.priors.nugget is None

    with pytest.raises(AssertionError):
        priors = GPPriors(n_corr=3, nugget_type="fit")
        gp = GaussianProcess(x, y, priors=priors)

    with pytest.raises(AssertionError):
        priors = GPPriors(n_corr=4, nugget_type="adaptive")
        gp = GaussianProcess(x, y, priors=priors)

    with pytest.raises(TypeError):
        GaussianProcess(x, y, priors=1.)

    x = np.array([[1., 1.], [2., 2.], [4., 4.]])
    y = np.array([2., 4., 6.])

    gp = GaussianProcess(x, y, mean="1", priors={"mean": (np.array([1.]), np.array([1.])), "n_corr": 2,
                                                 "nugget_type": "adaptive"})

    assert_allclose(gp.priors.mean.mean, [1.])
    assert_allclose(gp.priors.mean.cov, [1.])

    gp = GaussianProcess(x, y, mean="1", priors={"mean": None, "n_corr": 2, "nugget_type": "adaptive"})

    with pytest.raises(AssertionError):
        gp = GaussianProcess(x, y, mean="1", priors={"mean": (np.array([1., 2.]), np.array([1., 2.])), "n_corr": 2,
                                                     "nugget_type": "adaptive"})

@pytest.mark.parametrize("mean,nugget,sn", [(None, 0., 1.), (None, "adaptive", 0.),
                                            (None, "pivot", 0.),
                                            ("x[0]", "fit", np.log(1.e-6))])
def test_GaussianProcess_fit_logposterior(x, y, mean, nugget, sn):
    "test the fit and logposterior methods of GaussianProcess"

    if isinstance(nugget, float):
        nugget_type = "fixed"
    else:
        nugget_type = nugget

    gp = GaussianProcess(x, y, mean=mean, nugget=nugget, priors=GPPriors(n_corr=3, nugget_type=nugget_type))

    with pytest.raises(AssertionError):
        gp.theta = np.ones(gp.n_params + 1)

    theta = np.ones(gp.n_params)
    if nugget == "fit":
        theta[-1] = sn

    gp.theta = theta

    if nugget == "adaptive" or nugget == 0.:
        assert gp.nugget == 0.
        noise = 0.
    elif nugget == "pivot":
        assert gp.nugget is None
        noise = 0.
    else:
        assert_allclose(gp.nugget, np.exp(sn))
        noise = np.exp(sn)*np.eye(x.shape[0])
    K = np.exp(theta[gp.D])*gp.kernel.kernel_f(x, x, theta[:gp.D]) + noise


    L_expect = np.linalg.cholesky(K)
    Kinv_t_expect = np.linalg.solve(K, y)
    A = np.dot(gp._dm.T, np.linalg.solve(K, gp._dm))
    LA_expect = np.linalg.cholesky(A)

    logpost_expect = 0.5*(np.log(np.linalg.det(K)) +
                          np.log(np.linalg.det(A)) +
                          np.dot(y, Kinv_t_expect) -
                          np.linalg.multi_dot([Kinv_t_expect, gp._dm,
                                               np.linalg.solve(A, np.dot(gp._dm.T, Kinv_t_expect))]) +
                          (gp.n - gp.n_mean)*np.log(2.*np.pi))

    mean_expect = np.linalg.solve(A, np.dot(gp._dm.T, Kinv_t_expect))

    assert_allclose(L_expect, gp.Kinv.L)
    assert_allclose(Kinv_t_expect, gp.Kinv_t)
    assert_allclose(LA_expect, gp.Ainv.L)
    assert_allclose(mean_expect, gp.theta.mean)
    assert_allclose(logpost_expect, gp.current_logpost)
    assert_allclose(logpost_expect, gp.logposterior(theta))

def test_GaussianProcess_logposterior(x, y):
    "test logposterior method of GaussianProcess"

    # logposterior already tested, but check that parameters are re-fit if changed

    gp = GaussianProcess(x, y, nugget = 0., priors=GPPriors(n_corr=3, nugget_type="fixed"))

    theta = np.ones(gp.n_params)
    gp.fit(theta)

    theta = np.zeros(gp.n_params)

    K = np.exp(theta[-2])*gp.kernel.kernel_f(x, x, theta[:-1])

    L_expect = np.linalg.cholesky(K)
    Kinv_t_expect = np.linalg.solve(K, y)
    logpost_expect = 0.5*(np.log(np.linalg.det(K)) +
                          np.dot(y, Kinv_t_expect) +
                          (gp.n - gp.n_mean)*np.log(2.*np.pi))

    assert_allclose(logpost_expect, gp.logposterior(theta))
    assert_allclose(gp.Kinv.L, L_expect)
    assert_allclose(Kinv_t_expect, gp.Kinv_t)
    assert_allclose(gp.Ainv.L, np.zeros((0,0)))
    assert_allclose(logpost_expect, gp.current_logpost)

    # check we can set theta back to none correctly

    gp.theta = None
    assert gp.theta.get_data() is None
    assert gp.Kinv is None
    assert gp.Kinv_t is None
    assert gp.Ainv is None
    assert gp.current_logpost is None

@pytest.mark.skipif(not gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_GaussianProcessGPU_logposterior(x, y):
    "test logposterior method of GaussianProcessGPU"

    # logposterior already tested, but check that parameters are re-fit if changed
    gp = GaussianProcessGPU(x, y, nugget = 0., priors=GPPriors(n_corr=3, nugget_type="fixed"))

    theta = np.ones(gp.n_params)
    gp.fit(theta)

    theta = np.zeros(gp.n_params)


    K = np.exp(theta[-2])*gp.kernel.kernel_f(x, x, theta[:-1])

#    Q = gp.kernel.kernel_f(x, x, theta[:gp.D])

    L_expect = np.linalg.cholesky(K)
    Kinv_t_expect = np.linalg.solve(K, y)
    logpost_expect = 0.5*(np.log(np.linalg.det(K)) +
                          np.dot(y, Kinv_t_expect) +
                          gp.n*np.log(2.*np.pi))

    assert_allclose(logpost_expect, gp.logposterior(theta))
    assert_allclose(gp.L, L_expect)
    assert_allclose(Kinv_t_expect, gp.Kinv_t)
    assert_allclose(logpost_expect, gp.current_logpost)

    # check we can set theta back to none correctly

    gp.theta = None
    assert_allclose(gp.theta.get_data(), np.zeros(gp.n_params)) #GPU implementation resets to zero
    assert gp.Kinv_t is None
    assert gp.current_logpost is None


@pytest.fixture
def dx():
    return 1.e-6

@pytest.mark.parametrize("mean,nugget,sn", [(None, 1.e-6, 1.),
                                            (None, "pivot", 1.),
                                            ("x[0]", "fit", np.log(1.e-6))])
def test_GaussianProcess_logpost_deriv(dx, mean, nugget, sn):
    "test logposterior derivatives for GaussianProcess via finite differences"

    x, y = np.meshgrid(np.linspace(0., 4., 11), np.linspace(0., 4., 11))
    inputs = np.zeros((11*11, 2))
    inputs[:,0] = x.flatten()
    inputs[:,1] = y.flatten()
    targets = np.exp(-0.5*((x-2.)**2 + (y - 3.)**2)).flatten()

    if isinstance(nugget, float):
        nugget_type = "fixed"
    else:
        nugget_type = nugget

    gp = GaussianProcess(inputs, targets, mean=mean, nugget=nugget, priors=GPPriors(n_corr=2, nugget_type=nugget_type))

    n = gp.n_params
    theta = np.zeros(n)
    theta[:2] = -np.ones(2)
    theta[2] = -2.
    if nugget == "fit":
        theta[-1] = sn

    deriv = np.zeros(n)

    for i in range(n):
        dx_array = np.zeros(n)
        dx_array[i] = dx
        gp.logposterior(theta)
        gp.logposterior(theta - dx_array)
        deriv[i] = (gp.logposterior(theta) - gp.logposterior(theta - dx_array))/dx

    assert_allclose(deriv, gp.logpost_deriv(theta), atol=1.e-4, rtol=1.e-4)

@pytest.mark.skipif(not gpu_usable(), reason=GPU_NOT_FOUND_MSG)
@pytest.mark.parametrize("nugget,sn", [(0., 1.), ("adaptive", 1.),
                                            ("fit", np.log(1.e-6))])
def test_GaussianProcessGPU_logpost_deriv(x, y, dx, nugget, sn):
    "test logposterior derivatives for GaussianProcessGPU via finite differences"

    gp = GaussianProcessGPU(x, y, nugget=nugget)

    n = gp.n_params
    theta = np.zeros(n)
    theta[:2] = -np.ones(2)
    theta[2] = -2.
    if gp.nugget_type == "fit":
        theta[-1] = sn

    deriv = np.zeros(n)

    for i in range(n):
        dx_array = np.zeros(n)
        dx_array[i] = dx
        deriv[i] = (gp.logposterior(theta) - gp.logposterior(theta - dx_array))/dx

    assert_allclose(deriv, gp.logpost_deriv(theta), atol=1.e-4, rtol=1.e-4)

@pytest.mark.parametrize("mean,nugget,sn", [(None, 0., 1.),
                                            (None, "pivot", 1.), ("x[0]", "fit", np.log(1.e-6))])
def test_GaussianProcess_logpost_hessian(x, y, dx, mean, nugget, sn):
    "test the hessian method of GaussianProcess with finite differences"

    # zero mean, no nugget

    if isinstance(nugget, float):
        nugget_type = "fixed"
    else:
        nugget_type = nugget

    gp = GaussianProcess(x, y, mean=mean, nugget=nugget, priors=GPPriors(n_corr=3, nugget_type=nugget_type))

    n = gp.n_params
    theta = np.ones(n)
    if nugget == "fit":
        theta[-1] = sn

    with pytest.raises(NotImplementedError):
        gp.logpost_hessian(theta)

    # hess = np.zeros((n, n))
    #
    # for i in range(n):
    #     for j in range(n):
    #         dx_array = np.zeros(n)
    #         dx_array[j] = dx
    #         hess[i, j] = (gp.logpost_deriv(theta)[i] - gp.logpost_deriv(theta - dx_array)[i])/dx
    #
    # assert_allclose(hess, gp.logpost_hessian(theta), rtol=1.e-5, atol=1.e-7)

def test_GaussianProcess_default_priors(dx):
    "test that the default priors work as expected"

    x = np.array([[1., 1.], [2., 2.], [4., 4.]])
    y = np.array([2., 4., 6.])

    gp = GaussianProcess(x, y, nugget=0.)

    theta = np.zeros(gp.n_params)

    gp.fit(theta)

    K = gp.get_K_matrix()

    L_expect = np.linalg.cholesky(K)
    Kinv_t_expect = np.linalg.solve(K, y)
    logpost_expect = 0.5*(np.log(np.linalg.det(K)) +
                          np.dot(y, Kinv_t_expect) +
                          gp.n*np.log(2.*np.pi))

    dist = InvGammaPrior.default_prior_corr(np.array([1., 2., 4.]))

    logpost_expect -= 2.*dist.logp(1.)

    assert_allclose(L_expect, gp.Kinv.L)
    assert_allclose(Kinv_t_expect, gp.Kinv_t)
    assert_allclose(logpost_expect, gp.current_logpost)
    assert_allclose(logpost_expect, gp.logposterior(theta))

    n = gp.n_params
    deriv = np.zeros(n)

    for i in range(n):
        dx_array = np.zeros(n)
        dx_array[i] = dx
        deriv[i] = (gp.logposterior(theta) - gp.logposterior(theta - dx_array))/dx

    assert_allclose(deriv, gp.logpost_deriv(theta), atol=1.e-5, rtol=1.e-5)

    # hess = np.zeros((n, n))
    #
    # for i in range(n):
    #     for j in range(n):
    #         dx_array = np.zeros(n)
    #         dx_array[j] = dx
    #         hess[i, j] = (gp.logpost_deriv(theta)[i] - gp.logpost_deriv(theta - dx_array)[i])/dx
    #
    # assert_allclose(hess, gp.logpost_hessian(theta), rtol=1.e-5, atol=1.e-5)

@pytest.mark.parametrize("priors,nugget,sn", [({"mean": MeanPriors(mean=[1.], cov=[[1.]]),
                                                "corr": [ LogNormalPrior(0.9, 0.5), WeakPrior(), LogNormalPrior(0.5, 2.)],
                                                "cov": InvGammaPrior(2., 1.), "nugget_type": "fixed"}, 0., 0.),
                                           ( {"corr": [ WeakPrior(), LogNormalPrior(1.2, 0.2), WeakPrior()],
                                              "cov": GammaPrior(2., 1.), "nugget": InvGammaPrior(2., 1.e-6),
                                              "nugget_type": "fit"}, "fit", np.log(1.e-6))])
def test_GaussianProcess_priors(x, y, dx, priors, nugget, sn):
    "test that prior distributions are properly accounted for in posterior"

    gp = GaussianProcess(x, y, mean="1", priors=priors, nugget=nugget)

    theta = np.ones(gp.n_params)
    if nugget == "fit":
        theta[-1] = sn

    gp.fit(theta)

    if nugget == 0.:
        noise = 0.
    else:
        assert_allclose(gp.nugget, np.exp(sn))
        noise = np.exp(sn)*np.eye(x.shape[0])
    K = gp.get_K_matrix() + noise

    n_fact = gp.n
    if gp.priors.mean.has_weak_priors:
        n_fact -= gp.n_mean

    L_expect = np.linalg.cholesky(K)
    Kinv_t_expect = np.linalg.solve(K, y - gp.priors.mean.dm_dot_b(gp._dm))
    A = np.dot(gp._dm.T, np.linalg.solve(K, gp._dm)) + gp.priors.mean.inv_cov()
    LA_expect = np.linalg.cholesky(A)

    logpost_expect = 0.5*(np.log(np.linalg.det(K)) +
                          np.log(np.linalg.det(A)) +
                          gp.priors.mean.logdet_cov() +
                          np.dot(y - gp.priors.mean.dm_dot_b(gp._dm), Kinv_t_expect) -
                          np.linalg.multi_dot([Kinv_t_expect, gp._dm,
                                               np.linalg.solve(A, np.dot(gp._dm.T, Kinv_t_expect))]) +
                          n_fact*np.log(2.*np.pi))

    mean_expect = np.linalg.solve(A, np.dot(gp._dm.T, Kinv_t_expect) + gp.priors.mean.inv_cov_b())

    theta_transformed = np.zeros(gp.n_params)
    theta_transformed[0:3] = np.exp(-0.5*theta[0:3])
    theta_transformed[3:] = np.exp(theta[3:])
    for p, t in zip(gp.priors.corr, theta_transformed[0:3]):
        logpost_expect -= p.logp(t)
    logpost_expect -= gp.priors.cov.logp(theta_transformed[3])
    if nugget == "fit":
        logpost_expect -= gp.priors.nugget.logp(theta_transformed[-1])

    assert_allclose(L_expect, gp.Kinv.L)
    assert_allclose(Kinv_t_expect, gp.Kinv_t)
    assert_allclose(LA_expect, gp.Ainv.L)
    assert_allclose(mean_expect, gp.theta.mean)
    assert_allclose(logpost_expect, gp.current_logpost)
    assert_allclose(logpost_expect, gp.logposterior(theta))

    n = gp.n_params
    deriv = np.zeros(n)

    for i in range(n):
        dx_array = np.zeros(n)
        dx_array[i] = dx
        deriv[i] = (gp.logposterior(theta) - gp.logposterior(theta - dx_array))/dx

    assert_allclose(deriv, gp.logpost_deriv(theta), atol=1.e-5, rtol=1.e-5)

    # hess = np.zeros((n, n))
    #
    # for i in range(n):
    #     for j in range(n):
    #         dx_array = np.zeros(n)
    #         dx_array[j] = dx
    #         hess[i, j] = (gp.logpost_deriv(theta)[i] - gp.logpost_deriv(theta - dx_array)[i])/dx
    #
    # assert_allclose(hess, gp.logpost_hessian(theta), rtol=1.e-5, atol=1.e-5)

def test_GaussianProcess_predict(x, y):
    "test the predict method of GaussianProcess"

    # zero mean

    gp = GaussianProcess(x, y, nugget=0.)
    theta = np.ones(gp.n_params)

    gp.fit(theta)

    x_test = np.array([[2., 3., 4.]])

    mu, var, deriv = gp.predict(x_test)

    K = np.exp(theta[-1])*gp.kernel.kernel_f(x, x, theta[:-1])
    Ktest = np.exp(theta[-1])*gp.kernel.kernel_f(x_test, x, theta[:-1])

    mu_expect = np.dot(Ktest, gp.Kinv_t_mean)
    var_expect = np.exp(theta[-1]) - np.diag(np.dot(Ktest, np.linalg.solve(K, Ktest.T)))

    assert_allclose(mu, mu_expect)
    assert_allclose(var, var_expect)

    # check that reshaping works as expected

    x_test = np.array([2., 3., 4.])

    mu, var, deriv = gp.predict(x_test)

    assert_allclose(mu, mu_expect)
    assert_allclose(var, var_expect)

    # check that with 1D input data can handle 1D prediction data correctly

    gp = GaussianProcess(y, y, nugget=0.)

    gp.fit(np.ones(gp.n_params))

    n_predict = 51
    mu, var, deriv = gp.predict(np.linspace(0., 1., n_predict))

    assert mu.shape == (n_predict,)
    assert var.shape == (n_predict,)

    # nonzero mean function

    gp = GaussianProcess(x, y, mean="x[0]", nugget=0.)

    theta = np.ones(gp.n_params)

    gp.fit(theta)

    x_test = np.array([[2., 3., 4.]])

    mu, var, deriv = gp.predict(x_test)

    dm_test = gp.get_design_matrix(x_test)

    m = np.dot(dm_test, gp.theta.mean)
    K = np.exp(theta[-1])*gp.kernel.kernel_f(x, x, theta[:-1])
    Ktest = np.exp(theta[-1])*gp.kernel.kernel_f(x_test, x, theta[:-1])
    R = dm_test.T - np.dot(gp._dm.T, np.linalg.solve(K, Ktest.T))

    mu_expect = m + np.dot(Ktest, gp.Kinv_t_mean)
    var_expect = np.exp(theta[-1]) - np.diag(np.dot(Ktest, np.linalg.solve(K, Ktest.T)))
    var_expect += np.diag(np.dot(R.T, np.linalg.solve(np.dot(gp._dm.T, np.linalg.solve(K, gp._dm)),
                                                      R)))

    assert_allclose(mu, mu_expect)
    assert_allclose(var, var_expect)

    # check that a formula with LHS doesn't trip up predictions

    gp = GaussianProcess(x, y, mean="y ~ x[0]", nugget=0.)

    theta = np.ones(gp.n_params)

    gp.fit(theta)

    x_test = np.array([[2., 3., 4.]])

    mu, var, deriv = gp.predict(x_test)

    # check that predictions at inputs are close to mean

    mu, var, deriv = gp.predict(x)

    assert_allclose(mu, y)

    # nonzero mean priors

    # gp = GaussianProcess(x, y, mean="x[0]",
    #                      priors=GPPriors(mean=MeanPriors(mean=[0., 0.], cov=np.eye(2)),
    #                                      n_corr = 3, nugget_type="fixed"),
    #                      nugget=0.)
    #
    # theta = np.ones(gp.n_params)
    #
    # gp.fit(theta)
    #
    # x_test = np.array([[2., 3., 4.]])
    #
    # mu, var, deriv = gp.predict(x_test)
    #
    # dm_test = gp.get_design_matrix(x_test)
    #
    # m = np.dot(dm_test, gp.theta.mean)
    # K = np.exp(theta[-1])*gp.kernel.kernel_f(x, x, theta[:-1])
    # Ktest = np.exp(theta[-1])*gp.kernel.kernel_f(x_test, x, theta[:-1])
    # R = dm_test.T - np.dot(gp._dm.T, np.linalg.solve(K, Ktest.T))
    #
    # mu_expect = m + np.dot(Ktest, gp.Kinv_t)
    # var_expect += np.diag(np.dot(R.T, np.linalg.solve(np.eye(2) + np.dot(gp._dm.T, np.linalg.solve(K, gp._dm)),
    #                                                   R)))
    #
    # assert_allclose(mu, mu_expect)
    # assert_allclose(var, var_expect)

    # check unc and deriv flags work

    _, var, deriv = gp.predict(x_test, unc=False, deriv=False)

    assert var is None
    assert deriv is None

    # check that the returned PredictResult works correctly

    pr = gp.predict(x_test)

    assert_allclose(pr.mean, mu_expect)
    assert_allclose(pr.unc, var_expect)
    assert pr.deriv is None

    assert_allclose(pr['mean'], mu_expect)
    assert_allclose(pr['unc'], var_expect)
    assert pr["deriv"] is None

    assert_allclose(pr[0], mu_expect)
    assert_allclose(pr[1], var_expect)
    assert pr[2] is None

    # check that calling gp is equivalent to predicting

    assert_allclose(gp(x_test), mu_expect)

@pytest.mark.skipif(not gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_GaussianProcessGPU_predict(x, y, dx):
    "test the predict method of GaussianProcessGPU vs an equivalent GaussianProcess"

    # zero mean
    gp = GaussianProcessGPU(x, y, nugget=0.)
    theta = np.ones(gp.n_params)
    gp.fit(theta)

    x_test = np.array([[2., 3., 4.]])

    mu, var, deriv = gp.predict(x_test)

    # GP (CPU implementation) to compare it to
    gpcpu = GaussianProcess(x, y, nugget=0.)
    gpcpu.fit(theta)
    mu_expect, var_expect, _ = gpcpu.predict(x_test)

    D = gp.D
    deriv_expect = np.zeros((1, D))
    for i in range(D):
        dx_array = np.zeros(D)
        dx_array[i] = dx
        deriv_expect[0, i] = (gp.predict(x_test)[0] - gp.predict(x_test - dx_array)[0])/dx

    assert_allclose(mu, mu_expect)
    assert_allclose(var, var_expect)

    # check that reshaping works as expected

    x_test = np.array([2., 3., 4.])

    mu, var, deriv = gp.predict(x_test)

    assert_allclose(mu, mu_expect)
    assert_allclose(var, var_expect)

    # check that with 1D input data can handle 1D prediction data correctly

    gp = GaussianProcessGPU(y, y, nugget=0.)

    gp.fit(np.ones(gp.n_params))

    n_predict = 51
    mu, var, deriv = gp.predict(np.linspace(0., 1., n_predict))

    assert mu.shape == (n_predict,)
    assert var.shape == (n_predict,)
    assert deriv.shape == (n_predict, 1)

    # check unc and deriv flags work

    _, var, deriv = gp.predict(x_test, unc=False, deriv=False)

    assert var is None
    assert deriv is None

    # check that the returned PredictResult works correctly
    gp = GaussianProcessGPU(x, y, nugget=0.)
    theta = np.ones(gp.n_params)
    gp.fit(theta)
    x_test = np.array([[2., 3., 4.]])

    pr = gp.predict(x_test)

    assert_allclose(pr.mean, mu_expect)
    assert_allclose(pr.unc, var_expect)
    assert_allclose(pr.deriv, deriv_expect, atol=1.e-7, rtol=1.e-7)

    assert_allclose(pr['mean'], mu_expect)
    assert_allclose(pr['unc'], var_expect)
    assert_allclose(pr['deriv'], deriv_expect, atol=1.e-7, rtol=1.e-7)

    assert_allclose(pr[0], mu_expect)
    assert_allclose(pr[1], var_expect)
    assert_allclose(pr[2], deriv_expect, atol=1.e-7, rtol=1.e-7)

    # check that calling gp is equivalent to predicting

    assert_allclose(gp(x_test), mu_expect)

def test_GaussianProcess_predict_nugget(x, y):
    "test that the nugget works correctly when making predictions"

    nugget = 1.e0

    gp = GaussianProcess(x, y, nugget=nugget)
    theta = np.ones(gp.n_params)

    gp.fit(theta)

    preds = gp.predict(x)

    K = np.exp(theta[-1])*gp.kernel.kernel_f(x, x, theta[:-1])

    var_expect = np.exp(theta[-1]) + nugget - np.diag(np.dot(K, np.linalg.solve(K + np.eye(gp.n)*nugget, K)))

    assert_allclose(preds.unc, var_expect, atol=1.e-7)

    preds = gp.predict(x, include_nugget=False)

    var_expect = np.exp(theta[-1]) - np.diag(np.dot(K, np.linalg.solve(K + np.eye(gp.n)*nugget, K)))

    assert_allclose(preds.unc, var_expect, atol=1.e-7)


@pytest.mark.skipif(not gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_GaussianProcessGPU_predict_nugget(x, y):
    "test that the nugget works correctly when making predictions"

    nugget = 1.e0

    gp = GaussianProcessGPU(x, y, nugget=nugget)
    theta = np.ones(gp.n_params)
    gp.fit(theta)
    preds = gp.predict(x)

    gpcpu = GaussianProcess(x, y, nugget=nugget)
    gpcpu.fit(theta)
    preds_expect = gpcpu.predict(x)

    assert_allclose(preds.unc, preds_expect.unc, atol=1.e-7)

    preds = gp.predict(x, include_nugget=False)
    preds_expect = gpcpu.predict(x, include_nugget=False)

    assert_allclose(preds.unc, preds_expect.unc, atol=1.e-7)

def test_GaussianProcess_predict_pivot():
    "test that pivoting gives same predictions as standard version"

    # input arrays are re-ordered such that pivoting should re-order the second to match the first

    x1 = np.array([1., 4., 2.])
    x2 = np.array([1., 2., 4.])
    y1 = np.array([1., 1., 2.])
    y2 = np.array([1., 2., 1.])

    gp1 = GaussianProcess(x1, y1, nugget=0.)
    gp2 = GaussianProcess(x2, y2, nugget="pivot")

    gp1.theta = np.zeros(2)
    gp2.theta = np.zeros(2)

    xpred = np.linspace(0., 5.)

    mean1, var1, deriv1 = gp1.predict(xpred)
    mean2, var2, deriv2 = gp2.predict(xpred)

    assert_allclose(mean1, mean2)
    assert_allclose(var1, var2)

def test_GaussianProcess_predict_variance():
    "confirm that caching factorized matrix produces stable variance predictions"

    x = np.linspace(0., 5., 21)
    y = x**2
    x = np.reshape(x, (-1, 1))
    nugget = 1.e-8

    gp = GaussianProcess(x, y, nugget=nugget)

    theta = np.array([-7.352408190715323, 15.041447753599755])
    gp.fit(theta)

    testing = np.reshape(np.linspace(0., 5., 101), (-1, 1))

    _, var, _ = gp.predict(testing)

    assert_allclose(np.zeros(101), var, atol = 1.e-3)

def test_GaussianProcess_predict_full_cov(x, y):
    "Test the predictions with full covariance"

    # zero mean

    gp = GaussianProcess(x, y, nugget=0.0)
    theta = np.ones(gp.n_params)

    gp.fit(theta)

    x_test = np.array([[2.0, 3.0, 4.0]])

    mu, var, deriv = gp.predict(x_test, full_cov=True)

    K = np.exp(theta[-1]) * gp.kernel.kernel_f(x, x, theta[:-1])
    Kpredict = np.exp(theta[-1]) * gp.kernel.kernel_f(x_test, x_test, theta[:-1])
    Ktest = np.exp(theta[-1]) * gp.kernel.kernel_f(x_test, x, theta[:-1])

    mu_expect = np.dot(Ktest, gp.Kinv_t_mean)
    var_expect = Kpredict - np.dot(Ktest, np.linalg.solve(K, Ktest.T))

    assert_allclose(mu, mu_expect)
    assert_allclose(var, var_expect)

    # nonzero mean function

    gp = GaussianProcess(x, y, mean="x[0]", nugget=0.0)

    theta = np.ones(gp.n_params)

    gp.fit(theta)

    x_test = np.array([[2.0, 3.0, 4.0]])

    mu, var, deriv = gp.predict(x_test, full_cov=True)

    dm_test = gp.get_design_matrix(x_test)

    m = np.dot(dm_test, gp.theta.mean)
    K = np.exp(theta[-1]) * gp.kernel.kernel_f(x, x, theta[:-1])
    Kpredict = np.exp(theta[-1]) * gp.kernel.kernel_f(x_test, x_test, theta[:-1])
    Ktest = np.exp(theta[-1]) * gp.kernel.kernel_f(x_test, x, theta[:-1])
    R = dm_test.T - np.dot(gp._dm.T, np.linalg.solve(K, Ktest.T))

    mu_expect = m + np.dot(Ktest, gp.Kinv_t_mean)
    var_expect = Kpredict - np.diag(np.dot(Ktest, np.linalg.solve(K, Ktest.T)))
    var_expect += np.dot(
        R.T, np.linalg.solve(np.dot(gp._dm.T, np.linalg.solve(K, gp._dm)), R)
    )

    assert_allclose(mu, mu_expect)
    assert_allclose(var, var_expect)

def test_GaussianProcess_predict_failures(x, y):
    "test situations where predict method of GaussianProcess should fail"

    gp = GaussianProcess(x, y)

    with pytest.raises(ValueError):
        gp.predict(np.array([2., 3., 4.]))

    theta = np.ones(gp.n_params)
    gp.fit(theta)

    with pytest.raises(AssertionError):
        gp.predict(np.ones((2, 2, 2)))

    with pytest.raises(AssertionError):
        gp.predict(np.array([[2., 4.]]))

@pytest.mark.skipif(not gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_GaussianProcessGPU_predict_failures(x, y):
    "test situations where predict method of GaussianProcessGPU should fail"

    gp = GaussianProcessGPU(x, y)

    with pytest.raises(ValueError):
        gp.predict(np.array([2., 3., 4.]))

    theta = np.ones(gp.n_params)
    gp.fit(theta)

    with pytest.raises(AssertionError):
        gp.predict(np.ones((2, 2, 2)))

    with pytest.raises(AssertionError):
        gp.predict(np.array([[2., 4.]]))

def test_GaussianProcess_str(x, y):
    "Test function for string method"

    gp = GaussianProcess(x, y)
    assert (str(gp) == "Gaussian Process with {} training examples and {} input variables".format(x.shape[0], x.shape[1]))


@pytest.mark.skipif(not gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_GaussianProcessGPU_str(x, y):
    "Test function for string method"

    gp = GaussianProcessGPU(x, y)
    assert (str(gp) == "Gaussian Process with {} training examples and {} input variables".format(x.shape[0], x.shape[1]))
