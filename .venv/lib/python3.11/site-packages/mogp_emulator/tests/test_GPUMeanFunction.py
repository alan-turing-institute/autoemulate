import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..MeanFunction import MeanFunction
from .. import LibGPGPU

from ..GaussianProcessGPU import parse_meanfunc_formula

GPU_NOT_FOUND_MSG = "A compatible GPU could not be found or the GPU library (libgpgpu) could not be loaded"

@pytest.fixture
def x():
    return np.array([[1., 2., 3.], [4., 5., 6.]])

@pytest.fixture
def params():
    return np.array([1., 2., 4.])

@pytest.fixture
def zeroparams():
    return np.zeros(0)

@pytest.fixture
def oneparams():
    return np.ones(1)

@pytest.fixture
def dx():
    return 1.e-6

@pytest.mark.skipif(not LibGPGPU.gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_ZeroMean(x, params):
    "test zero mean function works as expected"
    mf = LibGPGPU.ZeroMeanFunc()
    assert isinstance(mf, LibGPGPU.BaseMeanFunc)
    assert mf.get_n_params() == 0
    assert_allclose(mf.mean_f(x,np.array([])), np.zeros(2))
    assert_allclose(mf.mean_deriv(x,np.array([])), np.zeros((2,1)))
    assert_allclose(mf.mean_inputderiv(x,np.array([])), np.zeros((3,2)))


@pytest.mark.skipif(not LibGPGPU.gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_FixedMean(x, params):
    "test fixed mean function works as expected"
    mf = LibGPGPU.FixedMeanFunc(22.2)
    assert isinstance(mf, LibGPGPU.BaseMeanFunc)
    assert mf.get_n_params() == 0
    assert_allclose(mf.mean_f(x,np.array([])), np.array([22.2,22.2]))
    assert_allclose(mf.mean_deriv(x,np.array([])), np.zeros((2,1)))
    assert_allclose(mf.mean_inputderiv(x,np.array([])), np.zeros((3,2)))


@pytest.mark.skipif(not LibGPGPU.gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_ConstMean(x, params):
    "test const mean function works as expected"
    mf = LibGPGPU.ConstMeanFunc()
    assert isinstance(mf, LibGPGPU.BaseMeanFunc)
    assert mf.get_n_params() == 1
    param = np.array([7.])
    assert_allclose(mf.mean_f(x, param), np.array([7.,7.]))
    assert_allclose(mf.mean_deriv(x, param), np.ones((2,1)))
    assert_allclose(mf.mean_inputderiv(x, param), np.zeros((3,2)))


@pytest.mark.skipif(not LibGPGPU.gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_PolyMean(x, params):
    "test polynomial mean function works as expected"

    # constructor takes list of lists, where inner list is a pair [index, power]
    mf = LibGPGPU.PolyMeanFunc([[0,1],[1,1],[2,1]]) # linear in all three dimensions
    assert isinstance(mf, LibGPGPU.BaseMeanFunc)
    assert mf.get_n_params() == 4
    param = np.array([1.,2.,3.,4.])
    expected = []
    for xi in x:
        expected.append(param[0]+param[1]*xi[0]+param[2]*xi[1]+param[3]*xi[2])
    assert_allclose(mf.mean_f(x, param), np.array(expected))
    deriv_result = mf.mean_deriv(x, param)
    assert_allclose(deriv_result[0], np.ones(2))
    for i in range(1, 4):
        assert_allclose(deriv_result[i], np.array([x[0][i-1],x[1][i-1]]))
    inputderiv_result = mf.mean_inputderiv(x, param)
    for i in range(3):
        assert_allclose(inputderiv_result[i], np.array((param[i+1],param[i+1])))


@pytest.mark.skipif(not LibGPGPU.gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_WrongNumParams(x, params):
    " test we get an informative RuntimeError if we provide the wrong number of parameters "
    zmf = LibGPGPU.ZeroMeanFunc()
    with pytest.raises(RuntimeError, match=r"Expected params list of length 0"):
        zmf.mean_f(x,params)
    fmf = LibGPGPU.FixedMeanFunc(5.)
    with pytest.raises(RuntimeError, match=r"Expected params list of length 0"):
        fmf.mean_f(x,params)
    cmf = LibGPGPU.ConstMeanFunc()
    with pytest.raises(RuntimeError, match=r"Expected params list of length 1"):
        cmf.mean_f(x,params)
    pmf = LibGPGPU.PolyMeanFunc([[0,1],[1,2],[1,3]])
    with pytest.raises(RuntimeError, match=r"Expected params list of length 4"):
        pmf.mean_f(x,params)


@pytest.mark.skipif(not LibGPGPU.gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_PolyBadDim(x, params):
    " test we get an informative RuntimeError if we have a polynomial term in a non-existent dimension "
    mf = LibGPGPU.PolyMeanFunc([[0,1],[4,3]])  # dimension index 4 doesn't exist
    with pytest.raises(RuntimeError, match=r"Dimension index must be less than 3"):
        mf.mean_f(x,params)


@pytest.mark.skipif(not LibGPGPU.gpu_usable(), reason=GPU_NOT_FOUND_MSG)
def test_parse_formula(x):
    mf = parse_meanfunc_formula("1.")
    assert isinstance(mf, LibGPGPU.FixedMeanFunc)
    mf = parse_meanfunc_formula("c")
    assert isinstance(mf, LibGPGPU.ConstMeanFunc)
    mf = parse_meanfunc_formula("c+c*x[0]")
    assert isinstance(mf, LibGPGPU.PolyMeanFunc)
    assert mf.get_n_params() == 2
    mf = parse_meanfunc_formula("c+c*x[0]+c*x[0]^2+c*x[1]")
    assert isinstance(mf, LibGPGPU.PolyMeanFunc)
    assert mf.get_n_params() == 4
    # try without the initial const term (should be the same
    mf = parse_meanfunc_formula("c*x[0]+c*x[0]^2+c*x[1]")
    assert isinstance(mf, LibGPGPU.PolyMeanFunc)
    assert mf.get_n_params() == 4
    # try with a cross term x[0]*x[1]
    with pytest.raises(NotImplementedError,
                       match=r"Cross terms, e.g. x\[0\]\*x\[1\] not implemented"):
        mf = parse_meanfunc_formula("c+x[0]*x[1]")
