"""This module provides classes and utilities for performing dimension
reduction.  There is a single class :class:`mogp_emulator.gKDR` which
implements the method of Fukumizu and Leng [FL13]_, and which can be
used jointly with Gaussian process emulation as in [LG17]_.

Example: ::

  >>> from mogp_emulator import gKDR
  >>> import numpy as np
  >>> X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
  >>> Y = np.array([0.0, 1.0, 5.0, 6.0])
  >>> xnew = np.array([0.5, 0.5])
  >>> dr = gKDR(X, Y, 1)
  >>> dr(xnew)
  array([0.60092477])

In this example, the reduction was performed from a two- to a
one-dimensional input space.  The value returned by ``dr(xnew)`` is
the input coordinate `xnew` transformed to the reduced space.

The following example illustrates how to perform Gaussian process
regression on the reduced input space:

::

  >>> import numpy as np
  >>> from mogp_emulator import gKDR, GaussianProcess, fit_GP_MAP

  ### generate some training data (from the function f)

  >>> def f(x):
  ...     return np.sqrt(x[0] + np.sin(0.1 * x[1]))

  >>> X = np.mgrid[0:10,0:10].T.reshape(-1,2)/10.0
  >>> print(X)
  [[0.  0. ]
   [0.1 0. ]
   [0.2 0. ]
   [0.3 0. ]
   [0.4 0. ]
   [0.5 0. ]
   [0.6 0. ]
   [0.7 0. ]
   [0.8 0. ]
   [0.9 0. ]
   [0.  0.1]
   [0.1 0.1]
   ...
   [0.8 0.9]
   [0.9 0.9]]

  >>> Y = np.apply_along_axis(f, 1, X)

  ### reduced input space
  >>> dr = gKDR(X, Y, K=1)

  ### train a Gaussian Process with reduced inputs
  >>> gp = GaussianProcess(dr(X), Y)
  >>> gp = fit_GP_MAP(gp)

  ### make a prediction (given an input in the reduced space)
  >>> Xnew = np.array([0.12, 0.37])
  >>> gp.predict(dr(Xnew))
  (array([0.398083]), ...)

  >>> f(Xnew)
  0.396221

"""

import sys
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import cho_solve, cho_factor
from mogp_emulator.utils import k_fold_cross_validation, integer_bisect

def gram_matrix(X, k):
    """Computes the Gram matrix of `X`

    :type X: ndarray

    :param X: Two-dimensional numpy array, where rows are feature
              vectors

    :param k: The covariance function

    :returns: The gram matrix of `X` under the kernel `k`, that is,
              :math:`G_{ij} = k(X_i, X_j)`
    """
    ## note: do not use squareform(pdist(X, k)) here, since it assumes
    ## that dist(x,x) == 0, which might not be the case for an arbitrary k.
    return cdist(X, X, k)

def gram_matrix_sqexp(X, sigma2):
    r"""Computes the Gram matrix of `X` under the squared expontial kernel.
    Equivalent to, but more efficient than, calling ``gram_matrix(X,
    k_sqexp)``

    :type X: ndarray

    :param X: Two-dimensional numpy array, where rows are feature
              vectors

    :param sigma2: The variance parameter of the squared exponential kernel

    :returns: The gram matrix of `X` under the squared exponential
              kernel `k_sqexp` with variance parameter `sigma2` (:math:`=\sigma^2`), that
              is, :math:`G_{ij} = k_{sqexp}(X_i, X_j; \sigma^2)`

    """
    return np.exp(-0.5 * squareform(pdist(X, 'sqeuclidean')) / sigma2)


def median_dist(X):
    """Return the median of the pairwise (Euclidean) distances between
    each row of X
    """
    return np.median(pdist(X))


class gKDR(object):

    """Dimension reduction by the gKDR method.

    See [Fukumizu1]_, [FL13]_ and [LG17]_.

    An instance of this class is callable, with the ``__call__``
    method taking an input coordinate and mapping it to a reduced
    coordinate.
    """

    def __init__(self, X, Y, K=None, X_scale = 1.0, Y_scale = 1.0, EPS=1E-8, SGX=None, SGY=None):
        """Create a gKDR object

        Given some `M`-dimensional inputs (explanatory variables) `X`,
        and corresponding one-dimensional outputs (responses) `Y`, use
        the gKDR method to produce a reduced version of the input
        space with `K` dimensions.

        :type X: ndarray, of shape (N, M)
        :param X: `N` rows of `M` dimensional input vectors

        :type Y: ndarray, of shape (N,)
        :param Y: `N` response values

        :type K: integer
        :param K: The number of reduced dimensions to use (`0 <= K <= M`).

        :type EPS: float
        :param EPS: The regularization parameter, default `1e-08`; `EPS >= 0`

        :type X_scale: float
        :param X_scale: Optional, default `1.0`.  If SGX is None (the default), scale the
                        automatically determined value for SGX by X_scale.  Otherwise ignored.

        :type Y_scale: float
        :param Y_scale: Optional, default `1.0`.  If SGY is None (the default), scale the
                        automatically determined value for SGY by Y_scale.  Otherwise ignored.

        :type SGX: float | NoneType
        :param SGX: Optional, default `None`. The kernel parameter representing the
                    scale of variation on the input space.  If `None`, then the median distance
                    between pairs of input points (`X`) is used (as computed by
                    :func:`mogp_emulator.DimensionReduction.median_dist`).  If a float is
                    passed, then this must be positive.

        :type SGY: float | NoneType
        :param SGY: Optional, default `None`. The kernel parameter representing the
                    scale of variation on the output space.  If `None`, then the median distance
                    between pairs of output values (`Y`) is used (as computed by
                    :func:`mogp_emulator.DimensionReduction.median_dist`).  If a float is
                    passed, then this must be positive.
        """

        ## Note: see the Matlab implementation ...

        N, M = np.shape(X)

        ## default K: use the entire input space
        if K is None:
            K = M

        assert(K >= 0 and K <= M)
        assert(EPS >= 0)
        assert(SGX is None or SGX > 0.0)
        assert(SGY is None or SGY > 0.0)

        Y = np.reshape(Y, (N,1))

        if SGX is None:
            SGX = X_scale * median_dist(X)
        if SGY is None:
            SGY = Y_scale * median_dist(Y)

        I = np.eye(N)

        SGX2 = max(SGX*SGX, sys.float_info.min)
        SGY2 = max(SGY*SGY, sys.float_info.min)

        Kx = gram_matrix_sqexp(X, SGX2)
        Ky = gram_matrix_sqexp(Y, SGY2)

        regularized_Kx = Kx + N*EPS*I

        # Returns a tuple `(factor, lower?)`
        # we pass this directly to cho_solve
        cho_data = cho_factor(regularized_Kx, lower=True)

        tmp = cho_solve(cho_data, Ky)
        F = cho_solve(cho_data, tmp.T).T

        Dx = np.reshape(np.tile(X,(N,1)), (N,N,M), order='F').copy()
        Xij = Dx - np.transpose(Dx, (1,0,2))
        Xij = Xij / SGX2
        H = Xij * np.tile(Kx[:,:,np.newaxis], (1,1,M))

        R = np.zeros((M,M), order='F')

        nabla_k = np.reshape(H, (N,N*M), order='F')
        Fm = np.reshape(np.matmul(F,nabla_k), (N,N,M), order='F')

        for k in range(N):
            R = R + np.dot(H[k,:,:].T, Fm[k,:,:])

        L, V = np.linalg.eigh(R)

        assert(np.allclose(V.imag, 0.0))

        idx = np.argsort(L, 0)[::-1] # sort descending

        # record B, along with some bookkeeping parameters
        self.X_scale = X_scale
        self.Y_scale = Y_scale
        self.K = K
        self.B = V[:, idx]


    def __call__(self, X):
        """Calling a gKDR object with a vector of N inputs returns the inputs
        mapped to the reduced space.

        :type X: ndarray, of shape `(N, M)`
        :param X: `N` coordinates (rows) in the unreduced `M`-dimensional space

        :rtype: ndarray, of shape `(N, K)`
        :returns:  `N` coordinates (rows) in the reduced `K`-dimensional space
        """
        return X @ self.B[:,0:self.K]


    @staticmethod
    def _compute_loss( X, Y, train_model, cross_validation_folds, *params, **kwparams):
        """Compute the L1 loss of a model (produced by calling train_model), via
        cross validation.  The model is trained on input parameters `x` that
        are first reduced via the dimension reduction procedure produced by
        calling ``gKDR(x, y, *params)``.

        :type X: ndarray, of shape (N, M)
        :param X: `N` input points with dimension `M`

        :type Y: ndarray, of shape (N,)
        :param Y: the `N` model observations, corresponding to each

        :type train_model: callable with the signature
                           `(ndarray, ndarray) -> ndarray -> ndarray`
        :param train_model: a callable, that when called with model inputs X
                            (shape `(Ntrain, M)`) and Y (shape `(Ntrain, M)`),
                            returns a "model", which is another callable, taking
                            an array (shape `(Npredict, M)`) of the points where
                            a prediction is desired, and returning an array
                            (shape `(Npredict,)`) of the corresponding
                            predictions.

        :type cross_validation_folds: integer
        :param cross_validation_folds: Use this many folds for cross-validation
                                       when tuning the parameters.

        :type params: tuple
        :param params: parameters to pass to :meth:`mogp_emulator.gKDR.__init__`

        :type kwparams: dict
        :param kwparams: keyword parameters to pass to
                         :meth:`mogp_emulator.gKDR.__init__`
        """

        ## combine input and output arrays, such that if
        ## X[i] = [1,2,3] and Y[i] = 4, then XY[i] = [1,2,3,4].
        ## That is,
        ## XY[:, -1] == Y
        ## XY[:, 0:-1] == X
        ##
        XY = np.hstack((X, Y[:,np.newaxis]))

        err = []
        for train, validate in k_fold_cross_validation(XY, cross_validation_folds):
            train = np.array(train)
            validate = np.array(validate)

            dr = gKDR(train[:,0:-1], train[:,-1], *params, **kwparams)

            model = train_model(dr(train[:,0:-1]), train[:,-1])

            error_L1 = np.mean(np.abs(validate[:,-1] - model(dr(validate[:,0:-1]))))
            err.append(error_L1)

        return np.mean(err)


    @classmethod
    def tune_parameters(cls, X, Y, train_model, cXs=None, cYs=None,
                        maxK=None, cross_validation_folds=5,
                        verbose=False):
        """Constructs a gKDR model with the structural dimension (K) and
        kernel scale parameters (cX, cY) that approximately minimize
        the L1 error between Y and the trained model (resulting from
        calling `train_model` on X and Y).

        Currently, this works as follows.  For each choice of cX and
        cY in `cXs` and `cYs`, find K by starting from a K of 1 and
        doubling K until the loss increases (using the value of K just
        before), or until K equals the input dimension (or maxK if
        specified).  The resulting choice of `(cX, cY, K)` is then
        taken as the minimum such choice over the cX, cY.

        :type X: ndarray, of shape (N, M)
        :param X: `N` input points with dimension `M`

        :type Y: ndarray, of shape (N,)
        :param Y: the `N` model observations, corresponding to each

        :type train_model: callable with the signature
                           `(ndarray, ndarray) -> ndarray -> ndarray`
        :param train_model: a callable, that when called with model inputs X
                            (shape `(Ntrain, M)`) and Y (shape `(Ntrain, M)`),
                            returns a "model", which is another callable, taking
                            an array (shape `(Npredict, M)`) of the points where
                            a prediction is desired, and returning an array
                            (shape `(Npredict,)`) of the corresponding
                            predictions.

        :type cXs: Iterable of `float`, or `NoneType`
        :param cXs: (optional, default None). The scale parameter for `X` in the
                    dimension reduction kernel.  Passed as the parameter
                    `X_scale` to the gKDR constructor
                    (:meth:`mogp_emulator.gKDR.__init__`). If None, `[0.5, 1,
                    5.0]` is used.

        :type cYs: Iterable of `float`, or `NoneType`
        :param cYs: (optional, default None). The scale parameter for `Y` in the
                    dimension reduction kernel.  Passed as the parameter
                    `Y_scale` to the gKDR constructor
                    (:meth:`mogp_emulator.gKDR.__init__`). If None, `[0.5, 1,
                    5.0]` is used.

        :type maxK: integer, or NoneType
        :param maxK: (optional default `None`). The largest structural dimension
                     to consider in the optimization.  This is useful when there
                     is a known bound on the dimension, to stop e.g. poor values
                     of cX or cY needlessly extending the search.  It is a good
                     idea to choose this parameter generously.

        :type cross_validation_folds: integer
        :param cross_validation_folds: (optional, default is 5): Use this many
                                       folds for cross-validation when tuning
                                       the parameters.

        :type verbose: bool
        :param verbose: produce a log to stdout of the optimization?

        :returns: A pair of: the gKDR object with parameters tuned according to
                  the above method, and a number representing the L1 loss of the
                  model trained on inputs as reduced by this dimension reduction
                  object.  :rtype: pair of a gKDR and a non-negative float

        *Example*

        Tune the structural dimension and lengthscale parameters
        within the kernel, minimizing the the loss from a
        Gaussian process regression:

          >>> from mogp_emulator import gKDR, GaussianProcess, fit_GP_MAP
          >>> X = ...
          >>> Y = ...
          >>> dr, loss = gKDR.tune_parameters(X, Y, fit_GP_MAP)
          >>> gp = GaussianProcess(dr(X), Y)

        Or, specifying some optional parameters for the lengthscales,
        the maximum value of `K` to use, the number of folds for
        cross-validation, and producing verbose output:

          >>> dr, loss = gKDR.tune_parameters(X, Y, fit_GP_MAP,
          ...                                 cXs = [0.5, 1.0, 2.0], cYs = [2.0],
          ...                                 maxK = 25, cross_validation_folds=4, verbose = True)

        """

        N, M = np.shape(X)

        if cXs is None:
            cXs = [0.5, 1.0, 5.0]

        if cYs is None:
            cYs = [0.5, 1.0, 5.0]

        if maxK is None:
            maxK = M

        assert(maxK >= 1 and maxK <= M)

        def compute_loss(*params):
            return gKDR._compute_loss(
                X, Y, train_model, cross_validation_folds, *params)

        # Search for K and scale parameters that together
        # (approximately) minimize the loss, which could be quite
        # noisy.  For each choice of the scale parameters (cX, cY), K
        # starts at 1 and doubles until the loss increases, or until K
        # equals the input dimension.  Keep the minimum over all of
        # the trial parameter sets.
        #
        min_loss = np.inf
        argmin_loss = None
        for cX in cXs:
            for cY in cYs:
                loss = np.inf
                params = None
                k = 1
                while (k <= maxK):
                    old_params, params = params, (k, cX, cY)
                    old_loss, loss = loss, compute_loss(*params)

                    if verbose:
                        print("loss(K={}, X_scale={}, Y_scale={}) = {}"\
                              .format(*params, loss))

                    if old_loss < loss:
                        if old_loss < min_loss:
                            min_loss = old_loss
                            argmin_loss = old_params
                        break
                    elif k == maxK:
                        if loss < min_loss:
                            min_loss = loss
                            argmin_loss = params
                        break
                    elif 2*k > maxK:
                        k = maxK
                    else:
                        k *= 2

        # construct the object with the full set of data, and
        # parameters found above
        #
        dr = gKDR(X, Y, *argmin_loss)

        return (dr, min_loss)
