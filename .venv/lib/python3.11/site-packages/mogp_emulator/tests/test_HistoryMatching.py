import numpy as np
from numpy.testing import assert_allclose
import pytest
from ..HistoryMatching import HistoryMatching
from ..fitting import fit_GP_MAP
from ..GaussianProcess import GaussianProcess, PredictResult
from ..MultiOutputGP import MultiOutputGP
from ..LibGPGPU import gpu_usable
from ..GaussianProcessGPU import GaussianProcessGPU

GPU_NOT_FOUND_MSG = "A compatible GPU could not be found or the GPU library (libgpgpu) could not be loaded"

def get_y_simulated_1D(x):
    n_points = len(x)
    f = np.zeros(n_points)
    for i in range(n_points):
        f[i] = np.sin(2.*np.pi*x[i] / 50.)
    return f

def test_sanity_checks():
    "test basic functioning of HistoryMatching"

    # Create a gaussian process
    x_training = np.array([
        [0.],
        [10.],
        [20.],
        [30.],
        [43.],
        [50.]
    ])

    y_training = get_y_simulated_1D(x_training)

    np.random.seed(47)
    gp = fit_GP_MAP(x_training, y_training)

    # Define observation and implausibility threshold
    obs = [-0.8, 0.0004]

    # Coords to predict
    n_rand = 2000
    x_predict_min = -3
    x_predict_max = 53
    x_predict = np.random.rand(n_rand)
    x_predict = np.sort(x_predict, axis=0)
    x_predict *= (x_predict_max - x_predict_min)
    x_predict += x_predict_min
    x_predict = x_predict[:, None]

    coords = x_predict

    expectations = gp.predict(coords)

    # Create History Matching Instance
    print("---TEST INPUTS---")
    print("No Args")
    hm = HistoryMatching()
    hm.status()

    print("Obs Only a - list")
    hm = HistoryMatching(obs=obs)
    hm.status()

    print("Obs only b - single-element list")
    hm = HistoryMatching(obs=[3.])
    hm.status()

    print("Obs only c - single-value")
    hm = HistoryMatching(obs=3.)
    hm.status()

    print("gp Only")
    hm = HistoryMatching(gp=gp)
    hm.status()

    print("Coords only a - 2d ndarray")
    hm = HistoryMatching(coords=coords)
    hm.status()

    print("Coords only b - 1d ndarray")
    hm = HistoryMatching(coords=np.random.rand(n_rand))
    hm.status()

    print("Coords only c - list")
    hm = HistoryMatching(coords=[a for a in range(n_rand)])
    hm.status()

    print("Expectation only")
    hm = HistoryMatching(expectations=expectations)
    hm.status()

    print("Threshold Only")
    hm = HistoryMatching(threshold=3.)
    hm.status()


    print("---TEST ASSIGNMENT---")
    print("Assign gp")
    hm = HistoryMatching(obs)
    hm.status()
    hm.set_gp(gp)
    hm.status()

    print("Assign Obs")
    hm = HistoryMatching(gp)
    hm.status()
    hm.set_obs(obs)
    hm.status()

    print("Assign Coords")
    hm = HistoryMatching()
    hm.status()
    hm.set_coords(coords)
    hm.status()

    print("Assign Expectations")
    hm = HistoryMatching()
    hm.status()
    hm.set_expectations(expectations)
    hm.status()

    print("Assign Threshold")
    hm = HistoryMatching()
    hm.status()
    hm.set_threshold(3.)
    hm.status()


    print("---TEST IMPLAUSABILIY---")
    print("implausibility test a - no vars")
    hm = HistoryMatching(obs=obs, gp=gp, coords=coords)
    I = hm.get_implausibility()

    print("implausibility test b - single value")
    hm = HistoryMatching(obs=obs, gp=gp, coords=coords)
    I = hm.get_implausibility(7.)


@pytest.mark.skipif(not gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_sanity_checks_GPU():
    "test basic functioning of HistoryMatching"

    # Create a gaussian process
    x_training = np.array([
        [0.],
        [10.],
        [20.],
        [30.],
        [43.],
        [50.]
    ])

    y_training = get_y_simulated_1D(x_training)

    np.random.seed(47)
    gp = GaussianProcessGPU(x_training, y_training)
    gp = fit_GP_MAP(gp)

    # Define observation and implausibility threshold
    obs = [-0.8, 0.0004]

    # Coords to predict
    n_rand = 2000
    x_predict_min = -3
    x_predict_max = 53
    x_predict = np.random.rand(n_rand)
    x_predict = np.sort(x_predict, axis=0)
    x_predict *= (x_predict_max - x_predict_min)
    x_predict += x_predict_min
    x_predict = x_predict[:, None]

    coords = x_predict

    expectations = gp.predict(coords)

    # Create History Matching Instance
    print("---TEST INPUTS---")
    print("No Args")
    hm = HistoryMatching()
    hm.status()

    print("Obs Only a - list")
    hm = HistoryMatching(obs=obs)
    hm.status()

    print("Obs only b - single-element list")
    hm = HistoryMatching(obs=[3.])
    hm.status()

    print("Obs only c - single-value")
    hm = HistoryMatching(obs=3.)
    hm.status()

    print("gp Only")
    hm = HistoryMatching(gp=gp)
    hm.status()

    print("Coords only a - 2d ndarray")
    hm = HistoryMatching(coords=coords)
    hm.status()

    print("Coords only b - 1d ndarray")
    hm = HistoryMatching(coords=np.random.rand(n_rand))
    hm.status()

    print("Coords only c - list")
    hm = HistoryMatching(coords=[a for a in range(n_rand)])
    hm.status()

    print("Expectation only")
    hm = HistoryMatching(expectations=expectations)
    hm.status()

    print("Threshold Only")
    hm = HistoryMatching(threshold=3.)
    hm.status()


    print("---TEST ASSIGNMENT---")
    print("Assign gp")
    hm = HistoryMatching(obs)
    hm.status()
    hm.set_gp(gp)
    hm.status()

    print("Assign Obs")
    hm = HistoryMatching(gp)
    hm.status()
    hm.set_obs(obs)
    hm.status()

    print("Assign Coords")
    hm = HistoryMatching()
    hm.status()
    hm.set_coords(coords)
    hm.status()

    print("Assign Expectations")
    hm = HistoryMatching()
    hm.status()
    hm.set_expectations(expectations)
    hm.status()

    print("Assign Threshold")
    hm = HistoryMatching()
    hm.status()
    hm.set_threshold(3.)
    hm.status()


    print("---TEST IMPLAUSABILIY---")
    print("implausibility test a - no vars")
    hm = HistoryMatching(obs=obs, gp=gp, coords=coords)
    I = hm.get_implausibility()

    print("implausibility test b - single value")
    hm = HistoryMatching(obs=obs, gp=gp, coords=coords)
    I = hm.get_implausibility(7.)


def test_HistoryMatching_init():
    "test the init method of HistoryMatching"

    # basic functionality

    hm = HistoryMatching()

    assert hm.gp == None
    assert hm.obs == None
    assert hm.coords == None
    assert hm.expectations == None
    assert hm.ndim == None
    assert hm.ncoords == None
    assert_allclose(hm.threshold, 3.)
    assert hm.I == None
    assert hm.NROY == None
    assert hm.RO == None

    # with gp/coords, obs, and threshold

    gp = GaussianProcess(np.reshape(np.linspace(0., 1.), (-1, 1)), np.linspace(0., 1.))
    coords = np.array([[0.2], [0.4]])
    hm = HistoryMatching(gp=gp, obs=1., coords=coords, expectations=None, threshold=5.)

    assert hm.gp == gp
    assert_allclose(hm.obs, [[1.], [0.]])
    assert_allclose(hm.coords, coords)
    assert hm.expectations is None
    assert hm.ndim == 1
    assert hm.ncoords == len(coords)
    assert_allclose(hm.threshold, 5.)
    assert hm.I == None
    assert hm.NROY == None
    assert hm.RO == None

    # with obs, expectations, coords, and threshold

    expectations = PredictResult(mean=np.array([1.]), unc=np.array([0.2]), deriv=np.array([[0.1]]))
    hm = HistoryMatching(gp=None, obs=[1., 0.1], coords=None, expectations=expectations,
                         threshold=5.)

    assert hm.gp == None
    assert_allclose(hm.obs, [[1.], [0.1]])
    assert hm.coords == None
    for a in range(3):
        assert_allclose(hm.expectations[a], expectations[a])
    assert hm.ndim == None
    assert hm.ncoords == len(expectations[0])
    assert_allclose(hm.threshold, 5.)
    assert hm.I == None
    assert hm.NROY == None
    assert hm.RO == None


def test_HistoryMatching_select_expectations():
    "test the select_expectations method of HistoryMatching"

    # correct functionality

    expectations = PredictResult(mean=np.array([2., 10.]), unc=np.array([0., 0.]),
                                 deriv=None)
    hm = HistoryMatching(obs=[1., 1.], expectations=expectations)

    expectations_new = hm._select_expectations()

    for a in range(2):
        assert_allclose(expectations[a], expectations_new[a])

    np.random.seed(57483)
    gp = fit_GP_MAP(np.reshape(np.linspace(0., 1.), (-1, 1)), np.linspace(0., 1.))
    coords = np.array([[0.1], [1.]])
    obs = [1., 0.01]
    expectations = gp.predict(coords)

    hm = HistoryMatching(gp=gp, obs=obs, coords=coords)

    expectations_new = hm._select_expectations()

    for a in range(2):
        assert_allclose(expectations[a], expectations_new[a])

    # ncoords somehow not set

    hm.ncoords = None
    with pytest.raises(ValueError):
        hm._select_expectations()

    # both coords and expectations set

    hm = HistoryMatching(gp=gp, obs=obs, coords=coords, expectations=expectations)

    with pytest.raises(ValueError):
        hm._select_expectations()

    # if no expectations provided, fails

    hm = HistoryMatching(obs=obs)

    with pytest.raises(ValueError):
        hm._select_expectations()

def test_HistoryMatching_get_implausibility():
    "test the get_implausibility method of HistoryMatching"

    # correct functionality

    expectations = PredictResult(mean=np.array([2., 10.]), unc=np.array([0., 0.]), deriv=np.array([[1., 2.]]))
    hm = HistoryMatching(obs=[1., 1.], expectations=expectations)
    I = hm.get_implausibility()

    assert_allclose(I, [1., 9.])
    assert_allclose(hm.I, [1., 9.])

    I = hm.get_implausibility(1.)

    assert_allclose(I, [1./np.sqrt(2.), 9./np.sqrt(2.)])
    assert_allclose(hm.I, [1./np.sqrt(2.), 9./np.sqrt(2.)])

    np.random.seed(57483)
    gp = fit_GP_MAP(np.reshape(np.linspace(0., 1.), (-1, 1)), np.linspace(0., 1.))
    coords = np.array([[0.1], [1.]])
    obs = [1., 0.01]
    mean, unc, _ = gp.predict(coords)
    I_exp = np.abs(mean - obs[0])/np.sqrt(unc + obs[1])

    hm = HistoryMatching(gp=gp, obs=obs, coords=coords)
    I = hm.get_implausibility()

    assert_allclose(I, I_exp)
    assert_allclose(hm.I, I_exp)

    # no observations

    hm = HistoryMatching(expectations=expectations)

    with pytest.raises(ValueError):
        hm.get_implausibility()

    # negative variance for model discrepancy

    hm = HistoryMatching(obs=[1., 1.], expectations=expectations)

    with pytest.raises(AssertionError):
        hm.get_implausibility(-1.)

def test_HistoryMatching_get_implausibility_mogp():
    "Test implausibility with multiple outputs"
    
    # correct functionality

    expectations = PredictResult(mean=np.array([[2., 10.], [4., 6.]]), unc=np.array([[0.5, 0.5], [0.5, 0.5]]),
                                 deriv=None)
    hm = HistoryMatching(obs=[[1., 5.], 0.5], expectations=expectations)
    I = hm.get_implausibility()

    assert_allclose(I, [1., 1.])
    assert_allclose(hm.I, [1., 1.])

    I = hm.get_implausibility(1.)

    assert_allclose(I, [1./np.sqrt(2.), 1./np.sqrt(2.)])
    assert_allclose(hm.I, [1./np.sqrt(2.), 1./np.sqrt(2.)])
    
    I = hm.get_implausibility(np.array([1., 1.]))

    assert_allclose(I, [1./np.sqrt(2.), 1./np.sqrt(2.)])
    assert_allclose(hm.I, [1./np.sqrt(2.), 1./np.sqrt(2.)])

    np.random.seed(57483)
    gp = fit_GP_MAP(np.reshape(np.linspace(0., 1.), (-1, 1)), [np.linspace(0., 1.), np.linspace(0., 1.)**2])
    coords = np.array([[0.1], [1.]])
    obs = [np.array([1., 5.]), np.array([0.01, 0.01])]
    mean, unc, _ = gp.predict(coords)
    I_exp = np.abs(mean - obs[0][:, np.newaxis])/np.sqrt(unc + obs[1][:,np.newaxis])
    I_exp2 = [np.max(x) for x in I_exp.T]

    hm = HistoryMatching(gp=gp, obs=obs, coords=coords)
    I = hm.get_implausibility(rank=0)

    assert_allclose(I, I_exp2)
    assert_allclose(hm.I, I_exp2)
    
    np.random.seed(57483)
    gp = fit_GP_MAP(np.reshape(np.linspace(0., 1.), (-1, 1)), [np.linspace(0., 1.),
                                                               np.linspace(0., 1.)**2,
                                                               np.linspace(0., 1.)**3])
    coords = np.array([[0.1], [1.]])
    obs = [np.array([1., 5., 4.]), np.array([0.01, 0.01, 0.01])]
    mean, unc, _ = gp.predict(coords)
    I_exp = np.abs(mean - obs[0][:, np.newaxis])/np.sqrt(unc + obs[1][:,np.newaxis])
    I_exp2 = [np.min(x) for x in I_exp.T]

    hm = HistoryMatching(gp=gp, obs=obs, coords=coords)
    I = hm.get_implausibility(rank=2)

    assert_allclose(I, I_exp2)
    assert_allclose(hm.I, I_exp2)

def test_HistoryMatching_get_NROY():
    "test the get_NROY method of HistoryMatching"

    # correct functionality

    hm = HistoryMatching(obs=[1., 1.], expectations=PredictResult(mean=np.array([2., 10.]),
                                                                  unc=np.array([0., 0.]),
                                                                  deriv=np.array([[1., 2.]])))
    hm.get_implausibility()

    NROY = hm.get_NROY()

    assert NROY == [0]

    hm = HistoryMatching(obs=[1., 0.], expectations=PredictResult(mean=np.array([2., 10.]),
                                                                  unc=np.array([0., 0.]),
                                                                  deriv=np.array([[1., 2.]])))

    NROY = hm.get_NROY(1.)

    assert NROY == [0]

def test_HistoryMatching_get_RO():
    "test the get_RO method of HistoryMatching"

    # correct functionality

    hm = HistoryMatching(obs=[1., 1.], expectations=PredictResult(mean=np.array([2., 10.]),
                                                                  unc=np.array([0., 0.]),
                                                                  deriv=np.array([[1., 2.]])))
    hm.get_implausibility()

    RO = hm.get_RO()

    hm = HistoryMatching(obs=[1., 0.], expectations=PredictResult(mean=np.array([2., 10.]),
                                                                  unc=np.array([0., 0.]),
                                                                  deriv=np.array([[1., 2.]])))

    RO = hm.get_RO(1.)

    assert RO == [1]

def test_HistoryMatching_set_gp():
    'test the set_gp method of HistoryMatching'

    # correct functionality

    gp = GaussianProcess(np.reshape(np.linspace(0., 1.), (-1, 1)), np.linspace(0., 1.))

    hm = HistoryMatching()
    hm.set_gp(gp)

    assert hm.gp == gp
    
    gp = MultiOutputGP(np.reshape(np.linspace(0., 1.), (-1, 1)), [np.linspace(0., 1.),
                                                                  np.linspace(0., 1.)**2])
    
    hm = HistoryMatching()
    hm.set_gp(gp)

    assert hm.gp == gp

    # bad type for GP

    with pytest.raises(TypeError):
        hm.set_gp(gp=1.)

@pytest.mark.skipif(not gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_HistoryMatching_set_gpGPU():
    'test the set_gp method of HistoryMatching'

    # correct functionality

    gp = GaussianProcessGPU(np.reshape(np.linspace(0., 1.), (-1, 1)), np.linspace(0., 1.))

    hm = HistoryMatching()
    hm.set_gp(gp)

    assert hm.gp == gp

    # bad type for GP

    with pytest.raises(TypeError):
        hm.set_gp(gp=1.)


def test_HistoryMatching_set_obs():
    'test the set_obs method of HistoryMatching'

    # correct functionality

    obs = [1., 0.1]

    hm = HistoryMatching()
    hm.set_obs(obs)

    assert_allclose(hm.obs, np.reshape(obs, (-1, 1)))

    obs = 1.
    hm.set_obs(obs)

    assert_allclose(hm.obs, [[obs], [0.]])
    
    obs = [np.array([1., 2.]), 2.]

    hm = HistoryMatching()
    hm.set_obs(obs)

    assert_allclose(hm.obs[0], obs[0])
    assert_allclose(hm.obs[1], obs[1])
    
    obs = [np.array([1., 2.]), np.array([2., 4.])]

    hm = HistoryMatching()
    hm.set_obs(obs)

    assert_allclose(hm.obs[0], obs[0])
    assert_allclose(hm.obs[1], obs[1])

    # wrong number of values for obs

    with pytest.raises(ValueError):
        hm.set_obs([1., 2., 3.])

    # negative variance

    with pytest.raises(AssertionError):
        hm.set_obs([1., -1.])

    # bad values for obs

    with pytest.raises(TypeError):
        hm.set_obs("abc")
        
    # array with wrong first dimension size
    
    with pytest.raises(AssertionError):
        hm.set_obs(np.ones((3, 2)))

def test_HistoryMatching_set_coords():
    "test the set_coords method of HistoryMatching"

    # correct functionality

    coords = np.array([[1.]])

    hm = HistoryMatching()
    hm.set_coords(coords)

    assert_allclose(hm.coords, coords)

    coords = np.array([1.])

    hm.set_coords(coords)

    assert_allclose(hm.coords, np.reshape(coords, (1, 1)))

    # confirm can reset coords

    hm.set_coords(None)

    assert hm.coords == None

def test_HistoryMatching_set_expectations():
    "test the set_expectations method of HistoryMatching"

    # correct functionality

    expectations = PredictResult(mean=np.array([0.]), unc=np.array([0.1]), deriv=np.array([[2.]]))

    hm = HistoryMatching()
    hm.set_expectations(expectations)

    for a in range(3):
        assert_allclose(hm.expectations[a], expectations[a])

    # confirm ability to reset expectations

    hm.set_expectations(None)
    assert hm.expectations == None

    # verify derivatives can be None

    expectations = PredictResult(mean=np.array([0.]), unc=np.array([0.1]), deriv=None)

    hm = HistoryMatching()
    hm.set_expectations(expectations)

    for a in range(2):
        assert_allclose(hm.expectations[a], expectations[a])
    assert hm.expectations[2] is None

    # negative variance

    expectations = PredictResult(mean=np.array([0.]), unc=np.array([-0.1]), deriv=np.array([[2.]]))

    with pytest.raises(AssertionError):
        hm.set_expectations(expectations)

    # bad values for expectations

    expectations = PredictResult(mean=np.array([0.1]), unc=1., deriv=1.)
    hm = HistoryMatching()

    with pytest.raises(TypeError):
        hm.set_expectations(expectations)

    expectations = PredictResult(mean=np.array([0.1]), unc=np.array([1., 2.]), deriv=1.)

    with pytest.raises(TypeError):
        hm.set_expectations(expectations)

    expectations = PredictResult(mean=np.array([1.]), unc=np.array([1., 2.]), deriv=None)

    with pytest.raises(ValueError):
        hm.set_expectations(expectations)


def test_HistoryMatching_set_threshold():
    "test the set_threshold method of HistoryMatching"

    # correct functionality

    hm = HistoryMatching()

    hm.set_threshold(1.)

    assert_allclose(hm.threshold, 1.)

    # threshold cannot be negative

    with pytest.raises(AssertionError):
        hm.set_threshold(-1.)

def test_HistoryMatching_update():
    "test the update method of HistoryMatching"

    # correct functionality

    hm = HistoryMatching()

    hm.coords = np.array([[1.]])

    hm.update()

    assert hm.ncoords == 1
    assert hm.ndim == 1

    hm = HistoryMatching()

    hm.expectations = PredictResult(mean=np.array([0.]), unc=np.array([0.1]), deriv=np.array([[2.]]))
    hm.update()

    assert hm.ncoords == 1

def test_HistoryMatching_str():
    "test the str method of HistoryMatching"

    # correct functionality

    hm = HistoryMatching()

    str_exp = ("History Matching tools created with:\n" +
               "Gaussian Process: None\n" +
               "Observations: None\n" +
               "Coords: None\n" +
               "Expectations: None\n" +
               "No. of Input Dimensions: None\n" +
               "No. of Descrete Expectation Values: None\n" +
               "I_threshold: {}\n" +
               "I: None\n" +
               "NROY: None\n" +
               "RO: None\n").format(3.)

    assert str(hm) == str_exp
