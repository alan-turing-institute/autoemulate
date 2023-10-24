import numpy as np
from mogp_emulator.linalg.cholesky import ChoInv, fixed_cholesky
from mogp_emulator.Priors import MeanPriors

def calc_Ainv(Kinv, dm, B):
    """
    Compute inverse of A matrix
    
    Computes the :math:`A` matrix
    (:math:`A = H^TK^{-1}H + B^{-1}`) and inverts it
    using Cholesky decomposition. :math:`A` has
    dimensions :math:`M\timesM` where :math:`M` is the
    number of mean paramaters. If the GP has no mean
    function, this will return a ``ChoInv`` object that
    returns zero whenever its ``solve`` method is called.
    
    :param Kinv: Inverted covariance matrix (as a ``ChoInv``
                 object)
    :type Kinv: ChoInv
    :param dm: Design matrix, must be a numpy array (or a
               ``patsy.DesignMatrix`` object) with shape
               ``(n, M)``
    :type dm: ndarray or DesignMatrix
    :param B: MeanPriors object with the appropriate size
              (either the ``mean`` and ``cov`` kwargs must have
              length ``M`` or be ``None`` to indicate weak
              prior information on the mean parameters)
    :type B: MeanPriors
    :returns: Inverted A matrix
    :rtype: ChoInv
    """
    
    assert isinstance(Kinv, ChoInv)
    assert isinstance(B, MeanPriors)
    
    A = np.dot(dm.T, Kinv.solve(dm)) + B.inv_cov()
    
    L = fixed_cholesky(A)
    
    return ChoInv(L)
    
def calc_A_deriv(Kinv, dm, dKdtheta):
    """
    Compute derivative of :math:`A` with respect to the raw
    hyperparameters
    
    Computing the gradient of the log posterior requires computing
    the derivative of :math:`A` with respect to the raw
    hyperparameters. This can be computed from the inverse of the
    covariance matrix, the design matrix, and the derivative of
    the covariance matrix with respect to the hyperparameters
    as
    
    :math:`\frac{\partial A}{\partial \theta} =
    -H^TK^{-1}\frac{\partial K}{\partial \theta}K^{-1}H`
    
    Note that the way these arrays are stored requires some
    transposition of the axes to get the solves and dot product
    dimensions to line up correctly.
    
    :param Kinv: Inverted covariance matrix (as a ``ChoInv``
                 object)
    :type Kinv: ChoInv
    :param dm: Design matrix, must be a numpy array (or a
               ``patsy.DesignMatrix`` object) with shape
               ``(n, M)``
    :type dm: ndarray or DesignMatrix
    :param dKdtheta: Derivative of the covariance matrix with
                     respect to the hyperparameters. Must be
                     ``ndarray`` with shape ``(n_data, n, n)``.
    :type dKdtheta: ndarray
    :returns: Derivative of A with respect to the hyperparameters
              as a ``ndarray`` with shape ``(n_data, n_mean, n_mean)``
              where the first dimension indicates the hyperparameters
              and the following dimensions are the elements of the
              derivative of A.
    :rtype: ndarray
    """

    assert isinstance(Kinv, ChoInv)
    assert dKdtheta.ndim == 3
    assert dKdtheta.shape[1] == dKdtheta.shape[2]
    assert dKdtheta.shape[1] == Kinv.L.shape[0]
    
    return -np.transpose(
            np.dot(
                dm.T,
                np.transpose(
                    Kinv.solve(np.transpose(np.dot(dKdtheta, Kinv.solve(dm)), (1, 0, 2))),
                    (1, 0, 2),
                ),
            ),
            (1, 0, 2),
        )

def calc_mean_params(Ainv, Kinv_t, dm, B):
    """
    Compute analytical mean solution
    
    Computes the analytical mean :math:`\hat{\beta}` given the
    factorized matrix for :math:`A=(H^TK^{-1}H)`, the solution
    to the inverse covariance matrix multiplied by the target
    array (:math:`y` here), the design matrix :math:`H`, and
    the covariance of the mean priors :math:`B`.
    
    :math:`\hat{\beta} = (H^TK^{-1}H)^{-1}(K^-1y + B^{-1})`
    
    :param Ainv: Factorized :math:`A` matrix
    :type Ainv: ChoInv
    :param Kinv_t: Inverse covariance matrix multiplied by
                   the targets (numpy array of length `n`)
    :type Kinv_t: ndarray
    :param dm: Design matrix, must be a numpy array (or a
               ``patsy.DesignMatrix`` object) with shape
               ``(n, M)``
    :type dm: ndarray or DesignMatrix
    :param B: MeanPriors object with the appropriate size
              (either the ``mean`` and ``cov`` kwargs must have
              length ``M`` or be ``None`` to indicate weak
              prior information on the mean parameters)
    :type B: MeanPriors
    :returns: numpy array holding analytical mean solution
              (array of length `n_mean`)
    :rtype: ndarray
    """
    
    assert isinstance(Ainv, ChoInv)
    assert isinstance(B, MeanPriors)
    
    return Ainv.solve(np.dot(dm.T, Kinv_t) + B.inv_cov_b())
    
def calc_R(Kinv_Ktest, dm, dmtest):
    """
    Compute R matrix
    
    The analytical mean solution requires adding an
    additional component to the predictive variance.
    Computing this requires computing the :math:`R`
    matrix, :math:`H^*^T - H^TK^{-1}K^*`, where
    :math:`H` is the design matrix for the training
    points, :math:`K` is the covariance matrix
    for the traning points, :math:`H^*` is the
    design matrix for the test points, and 
    :math:`K^*` is the covariance matrix
    for the test points.
    
    :param Kinv_Ktest: Product of the inverse covariance
                       matrix with the testing covariance
                       matrix. This is already computed
                       in the standard predictions, so
                       is assumed to be cached and passed
                       here as an argument. Shape is
                       ``(n, n_testing)``
    :type Kinv_Ktest: ndarray
    :param dm: Design matrix, must be a numpy array (or a
               ``patsy.DesignMatrix`` object) with shape
               ``(n, M)``
    :type dm: ndarray or DesignMatrix
    :param dmtest: Design matrix of the test data, must
                   be a numpy array (or a
                   ``patsy.DesignMatrix`` object) with
                   shape ``(n_testing, M)``
    :type dm: ndarray or DesignMatrix
    :returns: R matrix (shape ``(M, n_testing)``
    :rtype: ndarray
    """
    
    return dmtest.T - np.dot(dm.T, Kinv_Ktest)
    
def logdet_deriv(Kinv, dKdtheta):
    """
    Compute the derivative of the log-determinant of a matrix
    
    The derivative of the log-determinant of a matrix can
    be easily computed from the inverted matrix and the
    derivative of the matrix via
    
    :math:`{\rm tr}(K^{-1}\frac{\partial K}{\partial \theta})`
    
    :param Kinv: Inverted covariance matrix (as a ``ChoInv``
                 object)
    :type Kinv: ChoInv
    :param dKdtheta: Derivative of the covariance matrix with
                     respect to the hyperparameters. Must be
                     ``ndarray`` with shape ``(n_data, n, n)``.
    :type dKdtheta: ndarray
    :returns: derivative of the log determinant with respect
              to the same variables as are present in the
              ``dKdtheta`` argument.
    :rtype: float
    """
    
    assert isinstance(Kinv, ChoInv)
    assert dKdtheta.ndim == 3
    assert dKdtheta.shape[1] == dKdtheta.shape[2]
    assert dKdtheta.shape[1] == Kinv.L.shape[0]
    
    return np.trace(Kinv.solve(np.transpose(dKdtheta, (1, 2, 0))))