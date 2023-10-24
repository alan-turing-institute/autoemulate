import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..linalg.cholesky import ChoInv, ChoInvPivot, cholesky_factor, fixed_cholesky
from ..linalg.cholesky import jit_cholesky, _check_cholesky_inputs, pivot_cholesky, _pivot_transpose
from ..linalg.linalg_utils import calc_Ainv, calc_A_deriv, calc_mean_params, calc_R, logdet_deriv
from ..Kernel import SquaredExponential
from ..Priors import MeanPriors
from scipy import linalg

@pytest.fixture
def A():
    return np.array([[2., 1., 0.2], [1., 2., 0.4], [0.2, 0.4, 2.]])
    
@pytest.fixture
def b():
    return np.array([2., 3., 1.])

def test_ChoInv(A, b):
    "test the ChoInv class"
    
    L = linalg.cholesky(A, lower=True)
    Ainv = ChoInv(L)
    
    assert_allclose(Ainv.L, L)
    
    x = np.linalg.solve(A, b)
    
    assert_allclose(Ainv.solve(b), x)
    
    x = np.linalg.solve(L, b)
    
    assert_allclose(Ainv.solve_L(b), x)
    
    assert_allclose(np.log(np.linalg.det(A)), Ainv.logdet())
    
    assert Ainv.solve(np.zeros((3,0))).shape == (3,0)
    
    Ainv = ChoInv(np.zeros((0,0)))
    
    assert Ainv.solve(np.ones(3)).shape == (3,)
    
    Ainv = ChoInv(2.*np.ones((1,1)))
    
    assert_allclose(Ainv.solve(np.ones((1, 3, 1))), 0.25*np.ones((1,3,1)))

def test_ChoInvPivot(A, b):
    "test the cho_solve routine using pivoting"

    L = np.linalg.cholesky(A)

    x = linalg.cho_solve((L, True), b)

    L_pivot, P = pivot_cholesky(A)
    
    Ainv = ChoInvPivot(L_pivot, P)

    x_pivot = Ainv.solve(b)

    assert_allclose(x, x_pivot)
    
    x = np.linalg.solve(Ainv.L, b[Ainv.P])
    
    x_pivot = Ainv.solve_L(b)
    
    assert_allclose(x, x_pivot)
    
    prod = np.dot(b, np.linalg.solve(A, b))
    prod_pivot = np.dot(x_pivot, x_pivot)
    
    assert_allclose(prod, prod_pivot)

    with pytest.raises(AssertionError):
        ChoInvPivot(L_pivot, np.array([0, 2, 1, 1], dtype=np.int32)).solve(b)

    with pytest.raises(ValueError):
        ChoInvPivot(L_pivot, np.array([0, 0, 1], dtype=np.int32)).solve(b)

def test_check_cholesky_inputs():
    "Test function that checks inputs to cholesky decomposition routines"
    
    A = np.array([[2., 1.], [1., 2.]])
    B = _check_cholesky_inputs(A)
    
    assert_allclose(A, B)

    A = np.array([[1., 2.], [1., 2.]])
    with pytest.raises(AssertionError):
        _check_cholesky_inputs(A)
    
    A = np.array([1., 2.])
    with pytest.raises(AssertionError):
        _check_cholesky_inputs(A)
        
    A = np.array([[1., 2., 3.], [4., 5., 6.]])
    with pytest.raises(AssertionError):
        _check_cholesky_inputs(A)
    
    input_matrix = np.array([[-1., 2., 2.], [2., 3., 2.], [2., 2., -3.]])
    with pytest.raises(linalg.LinAlgError):
        _check_cholesky_inputs(input_matrix)

def test_fixed_cholesky(A):
    "Test the cholesky routine with fixed nugget"
    
    L_expected = np.array([[2., 0., 0.], [6., 1., 0.], [-8., 5., 3.]])
    input_matrix = np.array([[4., 12., -16.], [12., 37., -43.], [-16., -43., 98.]])
    
    L_actual = fixed_cholesky(input_matrix)
    assert_allclose(L_actual, L_expected)
    
    L_actual, nugget = cholesky_factor(input_matrix, 0., "fixed")
    assert_allclose(L_actual.L, L_expected)
    assert nugget == 0.
    
    L_actual, nugget = cholesky_factor(input_matrix, 0., "fit")
    assert_allclose(L_actual.L, L_expected)
    assert nugget == 0.
    
    L_expected = np.array([[1.0000004999998751e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],
                         [9.9999950000037496e-01, 1.4142132088085626e-03, 0.0000000000000000e+00],
                         [6.7379436301144941e-03, 4.7644444411381860e-06, 9.9997779980004420e-01]])
    input_matrix = np.array([[1. + 1.e-6        , 1.                , 0.0067379469990855 ],
                             [1.                , 1. + 1.e-6        , 0.0067379469990855 ],
                             [0.0067379469990855, 0.0067379469990855, 1. + 1.e-6         ]])
    L_actual = fixed_cholesky(input_matrix)
    assert_allclose(L_expected, L_actual)

def test_jit_cholesky():
    "Tests the stabilized Cholesky decomposition routine"
    
    L_expected = np.array([[2., 0., 0.], [6., 1., 0.], [-8., 5., 3.]])
    input_matrix = np.array([[4., 12., -16.], [12., 37., -43.], [-16., -43., 98.]])
    L_actual, jitter = jit_cholesky(input_matrix)
    assert_allclose(L_expected, L_actual)
    assert_allclose(jitter, 0.)
    
    L_expected = np.array([[1.0000004999998751e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],
                         [9.9999950000037496e-01, 1.4142132088085626e-03, 0.0000000000000000e+00],
                         [6.7379436301144941e-03, 4.7644444411381860e-06, 9.9997779980004420e-01]])
    input_matrix = np.array([[1.                , 1.                , 0.0067379469990855],
                             [1.                , 1.                , 0.0067379469990855],
                             [0.0067379469990855, 0.0067379469990855, 1.                ]])
    L_actual, jitter = jit_cholesky(input_matrix)
    assert_allclose(L_expected, L_actual)
    assert_allclose(jitter, 1.e-6)
    
    L_actual, jitter = cholesky_factor(input_matrix, 0., "adaptive")
    assert_allclose(L_expected, L_actual.L)
    assert_allclose(jitter, 1.e-6)
    
    input_matrix = np.array([[1.e-6, 1., 0.], [1., 1., 1.], [0., 1., 1.e-10]])
    with pytest.raises(linalg.LinAlgError):
        jit_cholesky(input_matrix)

def test_pivot_cholesky():
    "Tests  pivoted cholesky decomposition routine"
    
    input_matrix = np.array([[4., 12., -16.], [12., 37., -43.], [-16., -43., 98.]])
    input_matrix_copy = np.copy(input_matrix)
    L_expected = np.array([[ 9.899494936611665 ,  0.                ,  0.                ],
                           [-4.3436559415745055,  4.258245303082538 ,  0.                ],
                           [-1.616244071283537 ,  1.1693999481734827,  0.1423336335961131]])
    Piv_expected = np.array([2, 1, 0], dtype = np.int32)
    
    L_actual, Piv_actual = pivot_cholesky(input_matrix)
    
    assert_allclose(L_actual, L_expected)
    assert np.array_equal(Piv_expected, Piv_actual)
    assert_allclose(input_matrix, input_matrix_copy)
    
    input_matrix = np.array([[1., 1., 1.e-6], [1., 1., 1.e-6], [1.e-6, 1.e-6, 1.]])
    input_matrix_copy = np.copy(input_matrix)
    L_expected = np.array([[1.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00],
                           [9.9999999999999995e-07, 9.9999999999949996e-01, 0.0000000000000000e+00],
                           [1.0000000000000000e+00, 0.0000000000000000e+00, 3.3333333333316667e-01]])
    Piv_expected = np.array([0, 2, 1], dtype=np.int32)               
    
    L_actual, Piv_actual = pivot_cholesky(input_matrix)
    
    assert_allclose(L_actual, L_expected)
    assert np.array_equal(Piv_expected, Piv_actual)
    assert_allclose(input_matrix, input_matrix_copy)
    
    L_actual, nugget = cholesky_factor(input_matrix, np.array([]), "pivot")
    assert_allclose(L_actual.L, L_expected)
    assert np.array_equal(Piv_expected, L_actual.P)
    assert len(nugget) == 0
    
def test_pivot_transpose():
    "Test function to invert pivot matrix"
    
    P = np.array([0, 2, 1], dtype = np.int32)
    
    Piv = _pivot_transpose(P)
    
    np.array_equal(Piv, P)
    
    P = np.array([1, 2, 1], dtype = np.int32)
    
    with pytest.raises(ValueError):
        _pivot_transpose(P)

    P = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int32)

    with pytest.raises(AssertionError):
        _pivot_transpose(P)

@pytest.fixture
def dm():
    return np.array([[1., 1.], [1., 2.], [1., 4.]])
    
@pytest.fixture
def Kinv(A):
    return ChoInv(np.linalg.cholesky(A))

def test_calc_Ainv(A, dm, Kinv):
    "test the function to compute inverse of A"
    
    # Zero mean, weak mean covariance
    
    dm_z = np.zeros((3, 0))
    B = MeanPriors()
    
    result = calc_Ainv(Kinv, dm_z, B)
    
    assert result.L.shape == (0,0)
    
    # nonzero mean, weak mean covariance
    
    result = calc_Ainv(Kinv, dm, B)
    result_expected = np.linalg.cholesky(np.dot(dm.T, np.dot(np.linalg.inv(A), dm)))
    
    assert_allclose(result.L, result_expected)
    
    # nonzero mean, mean covariance
    
    B = MeanPriors(mean=[2., 1.], cov=np.eye(2))
    
    result = calc_Ainv(Kinv, dm, B)
    result_expected = np.linalg.cholesky(np.dot(dm.T, np.dot(np.linalg.inv(A), dm)) + np.eye(2))
    
    assert_allclose(result.L, result_expected)
    
    with pytest.raises(AssertionError):
        calc_Ainv(1., dm, B)
        
    with pytest.raises(AssertionError):
        calc_Ainv(Kinv, dm, 1.)

def test_calc_A_deriv(dm):
    "test calculating the derivative of Ainv"
    
    x = np.array([[1.], [2.], [4.]])
    
    K = SquaredExponential().kernel_f(x, x, [0.])
    dKdtheta = SquaredExponential().kernel_deriv(x, x, [0.])
    Kinv = ChoInv(np.linalg.cholesky(K))
    A = np.dot(dm.T, Kinv.solve(dm))
    
    deriv_expect = calc_A_deriv(Kinv, dm, dKdtheta)
    
    dx = 1.e-6
    K2 = SquaredExponential().kernel_f(x, x, [-dx])
    Kinv_2 = ChoInv(np.linalg.cholesky(K2))
    A2 = np.dot(dm.T, Kinv_2.solve(dm))
    deriv_fd = np.zeros((1, 2, 2))
    deriv_fd[0] = (A - A2)/dx
                
    assert_allclose(deriv_expect, deriv_fd, atol=1.e-6, rtol=1.e-6)
    
    # zero mean
    
    dm_z = np.zeros((3, 0))
    A = np.dot(dm_z.T, Kinv.solve(dm_z))
    
    deriv_expect = calc_A_deriv(Kinv, dm_z, dKdtheta)
    
    dx = 1.e-6
    K2 = SquaredExponential().kernel_f(x, x, [-dx])
    Kinv_2 = ChoInv(np.linalg.cholesky(K2))
    A2 = np.dot(dm_z.T, Kinv_2.solve(dm_z))
    deriv_fd = np.zeros((1, 0, 0))
    deriv_fd[0] = (A - A2)/dx
                
    assert_allclose(deriv_expect, deriv_fd, atol=1.e-6, rtol=1.e-6)
    
    with pytest.raises(AssertionError):
        calc_A_deriv(1., dm, dKdtheta)
        
    with pytest.raises(AssertionError):
        calc_A_deriv(Kinv, dm, np.ones(3))
        
    with pytest.raises(AssertionError):
        calc_A_deriv(Kinv, dm, np.ones((1, 2, 3)))
        
    with pytest.raises(AssertionError):
        calc_A_deriv(Kinv, dm, np.ones((1, 2, 2)))

def test_calc_mean_params(A, dm, Kinv):
    "test the calc_mean_params function"
    
    # weak mean priors
    
    Kinv_t = Kinv.solve(np.array([1., 2., 4.]))
    B = MeanPriors()
    
    Ainv = calc_Ainv(Kinv, dm, B)
    
    beta_actual = calc_mean_params(Ainv, Kinv_t, dm, B)
    beta_expected = Ainv.solve(np.dot(dm.T, Kinv_t))
    
    assert_allclose(beta_actual, beta_expected)
    
    # mean priors
    
    B = MeanPriors(mean=[2., 1.], cov=np.eye(2))
    
    beta_actual = calc_mean_params(Ainv, Kinv_t, dm, B)
    beta_expected = Ainv.solve(np.dot(dm.T, Kinv_t) + np.array([2., 1.]))
    
    assert_allclose(beta_actual, beta_expected)
    
    with pytest.raises(AssertionError):
        calc_mean_params(1., Kinv_t, dm, B)
        
    with pytest.raises(AssertionError):
        calc_mean_params(Ainv, Kinv_t, dm, 1.)

def test_calc_R(A, dm, Kinv):
    "test the calc_R function"
    
    dmtest = np.array([[1., 3.], [1., 5.]])
    Ktest = np.array([[0.2, 0.6, 0.8], [0.1, 0.2, 1.]]).T
    Kinv_Ktest = Kinv.solve(Ktest)

    R_expected = dmtest.T - np.dot(dm.T, np.linalg.solve(A, Ktest))
    
    R_actual = calc_R(Kinv_Ktest, dm, dmtest)
    
    assert_allclose(R_actual, R_expected)
        
    # zero mean function
    
    dm_z = np.zeros((3, 0))
    dm_z_test = np.zeros((2, 0))
    
    R_actual = calc_R(Kinv_Ktest, dm_z, dm_z_test)
    R_expected = np.zeros((0,2))
    
    assert_allclose(R_actual, R_expected)

def test_logdet_deriv():
    "compute the derivative of the log determinant of a matrix"
    
    x = np.array([[1.], [2.], [4.]])
    
    K = SquaredExponential().kernel_f(x, x, [0.])
    dKdtheta = SquaredExponential().kernel_deriv(x, x, [0.])
    Kinv = ChoInv(np.linalg.cholesky(K))
    
    deriv_expect = logdet_deriv(Kinv, dKdtheta)
    
    dx = 1.e-6
    deriv_fd = np.zeros(1)
    deriv_fd[0] = (np.log(np.linalg.det(SquaredExponential().kernel_f(x, x, [0.]))) -
                   np.log(np.linalg.det(SquaredExponential().kernel_f(x, x, [-dx]))))/dx
                
    assert_allclose(deriv_expect, deriv_fd, atol=1.e-6, rtol=1.e-6)
    
    x = np.array([[1., 4.], [2., 2.], [4., 1.]])
    
    K = SquaredExponential().kernel_f(x, x, [0., 0.])
    dKdtheta = SquaredExponential().kernel_deriv(x, x, [0., 0.])
    Kinv = ChoInv(np.linalg.cholesky(K))
    
    deriv_expect = logdet_deriv(Kinv, dKdtheta)
    
    dx = 1.e-6
    deriv_fd = np.zeros(2)
    deriv_fd[0] = (np.log(np.linalg.det(SquaredExponential().kernel_f(x, x, [0., 0.]))) -
                   np.log(np.linalg.det(SquaredExponential().kernel_f(x, x, [-dx, 0.]))))/dx
    deriv_fd[1] = (np.log(np.linalg.det(SquaredExponential().kernel_f(x, x, [0., 0.]))) -
                   np.log(np.linalg.det(SquaredExponential().kernel_f(x, x, [0., -dx]))))/dx

    assert_allclose(deriv_expect, deriv_fd, atol=1.e-6, rtol=1.e-6)
    
    with pytest.raises(AssertionError):
        logdet_deriv(1., dKdtheta)
        
    with pytest.raises(AssertionError):
        logdet_deriv(Kinv, np.ones(3))
        
    with pytest.raises(AssertionError):
        logdet_deriv(Kinv, np.ones((1, 2, 3)))
        
    with pytest.raises(AssertionError):
        logdet_deriv(Kinv, np.ones((1, 2, 2)))
