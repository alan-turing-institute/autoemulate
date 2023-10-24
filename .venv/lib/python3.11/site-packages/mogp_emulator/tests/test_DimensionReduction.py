import numpy as np
import pytest
from scipy import linalg
from .. import gKDR
from ..DimensionReduction import median_dist, gram_matrix_sqexp, gram_matrix
from .. import fitting

##### Some simple functions useful for training
def fn(x):
    """A linear function for testing the dimension reduction"""
    return 10*(x[0] + x[1]) + (x[1] - x[0])

def fn2(x):
    """A linear function for testing the dimension reduction"""
    return 10*(x[0] + x[1]) + (x[1] - x[0]) + x[2] + 0.1*x[3]

def fn3(x):
    return x[0]


##### The tests

def test_DimensionReduction_basic():
    """Basic check that we can create gKDR with the expected arguments"""
    Y = np.array([[1],[2.1],[3.2]])
    X = np.array([[1,2,3],[4,5.1,6],[7.1,8,9.1]])
    K_in = 2
    dr = gKDR(X, Y, K=K_in, SGX=2, SGY=2, EPS=1E-5)
    assert(dr.K == K_in)


def test_DimensionReduction_tune_parameters():
    """Check that we can tune the dimension reduction to discover the
    correct structural dimension in a simple case"""

    np.random.seed(100)
    X = np.random.random((20,20))
    Y = np.apply_along_axis(fn3, 1, X)

    dr, loss = gKDR.tune_parameters(X, Y, fitting.fit_GP_MAP,
                                    cXs=[5.0], cYs=[5.0], maxK = 3)

    # These are somewhat conservative bounds: the random seed makes
    # the resulting K and loss deterministic.  For other seeds, these
    # bounds are unlikely to be exceeded.
    assert(dr.K <= 2)
    assert(loss > 0.0 and loss < 0.2)


def test_DimensionReduction_GP():
    """Test that a GP based on reduced inputs behaves well."""

    # Make some test points on a grid.  Any deterministic set of
    # points would work well for this test.

    X = np.mgrid[0:10,0:10].T.reshape(-1,2)/10.0
    Y = np.apply_along_axis(fn, 1, X)
    dr = gKDR(X,Y,1)

    np.random.seed(10)
    gp = fitting.fit_GP_MAP(X, Y)

    gp_red = fitting.fit_GP_MAP(dr(X), Y)

    ## some points offset w.r.t the initial grid
    Xnew = (np.mgrid[0:9,0:9].T.reshape(-1,2) + 0.5)/10.0

    Yexpect = np.apply_along_axis(fn, 1, Xnew)
    Ynew = gp.predict(Xnew)[0] # value prediction
    Ynew_red = gp_red.predict(dr(Xnew))[0] # value prediction

    # check that the fit was reasonable in both cases
    assert(np.max(np.abs(Ynew - Yexpect)) <= 0.02)
    assert(np.max(np.abs(Ynew_red - Yexpect)) <= 0.02)


def test_DimensionReduction_B():
    """Test that a dimension reduction gives the same result as a
    pre-computed result from the Fukumizu matlab code (see
    [Fukumizu1]_)."""

    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    Y = np.array([0.1, 1.0, 3.0, 3.6])
    dr = gKDR(X,Y,2,SGX=1.0,SGY=2.0)

    B_expected = np.array([[-0.2653073259794961, -0.9641638982982144],
                           [-0.9641638982982144,  0.2653073259794961]])

    for i in range(B_expected.shape[1]):
        r = dr.B[:,i]/B_expected[:,i]
        assert(np.allclose(r, 1.0) or np.allclose(r, -1.0))

def test_DimensionReduction_median_dist():
    """Some basic checks of `mean_dist`
    """
    X1 = np.array([[0.0], [1.0], [2.0]])
    assert(np.allclose(median_dist(X1), 1))

    X2 = np.array([[0.0], [1.0], [2.0], [3.0]])
    assert(np.allclose(median_dist(X2), 1.5))

def test_DimensionReduction_gram_matrix():
    """Some basic checks of `gram_matrix` and `gram_matrix_sqexp`
    """
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    def k_dot(x0, x1):
        return np.dot(x0, x1)

    def k_sqexp(x0, x1):
        d = x0 - x1
        return np.exp(-0.5 * np.dot(d, d))

    G_dot = gram_matrix(X, k_dot)
    assert(np.allclose(G_dot, np.array([[0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 1.0],
                                        [0.0, 0.0, 1.0, 1.0],
                                        [0.0, 1.0, 1.0, 2.0]])))

    G_sqexp1 = gram_matrix_sqexp(X, 1.0)
    G_sqexp2 = gram_matrix(X, k_sqexp)
    G_sqexp_expected = np.exp(np.array([[ 0.0, -0.5, -0.5, -1.0],
                                        [-0.5,  0.0, -1.0, -0.5],
                                        [-0.5, -1.0,  0.0, -0.5],
                                        [-1.0, -0.5, -0.5,  0.0]]))

    assert(np.allclose(G_sqexp1, G_sqexp_expected))
    assert(np.allclose(G_sqexp2, G_sqexp_expected))


def test_DimensionReduction_large():
    """Check that gKDR runs on a 'large' problem

    The implementation of gKDR in mogp v0.7 and earlier did not scale
    well with problem size.  This test is of a larger case than could
    reasonably be handled by that implementation, but should cause no
    issue for the current one.  The check is simply that the function
    completes without a crash or error.
    """
    X = np.eye(200, 3200)
    Y = np.arange(200)

    dr = gKDR(X, Y, 2)
