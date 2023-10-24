import numpy as np
from mogp_emulator.GaussianProcess import GaussianProcessBase
from mogp_emulator.MultiOutputGP import MultiOutputGPBase
from mogp_emulator.linalg import cholesky_factor
from scipy.stats import f


def mahalanobis(gp, valid_inputs, valid_targets, scaled=False):
    """Compute the Mahalanobis distance on a validation dataset

    Given a fit GP and a set of inputs and targets for validation,
    compute the Mahalanobis distance (the correlated equivalent
    of the sum of the squared standard errors):

    .. math::
        M = (y_{valid} - y_{pred})^T K^{-1} (y_{valid} - y_{pred})

    The Mahalanobis distance is expected to follow a scaled
    Fisher-Snedecor distribution with ``(n_valid, n - n_mean - 2)``
    degrees of freedom. If ``scaled=True`` is selected, then
    the returned distance will be scaled by subtracting the
    expected mean and dividing by the standard deviation of
    this distribution. Note that the Fisher-Snedecor distribution
    is not symmetric, so this cannot be interpreted in the
    same way as standard errors, but this can nevertheless be
    a useful heuristic. By default, the Mahalanobis distance
    is not scaled, and a convenience function
    ``generate_mahal_dist`` is provided to simplify comparison
    of the Mahalanobis distance to the expected distribution.

    ``gp`` must be a fit ``GaussianProcess`` or ``MultiOutputGP``
    object, ``valid_inputs`` must be valid input data to the
    GP/MOGP, and ``valid_targets`` must be valid target data of
    the appropraite shape for the GP/MOGP.

    :param gp: A fit ``GaussianProcess`` or ``MultiOutputGP``
               object. If the GP/MOGP has not been fit, a
               ``ValueError`` will be raised.
    :type gp: ``GaussianProcess`` or ``MultiOutputGP``
    :param valid_inputs: Input points at which the GP will
                         be validated. Must correspond to
                         the appropriate inputs to the
                         provided GP.
    :type valid_inputs: ndarray
    :param valid_targets: Target points at which the GP will
                          be validated. Must correspond to
                          the appropriate target shape for
                          the provided GP.
    :type valid_targets: ndarray
    :param scaled: Flag indicating if the output Mahalanobis
                   distance should be scaled by subtracting
                   the mean and dividing by the standard
                   deviation of the expected Fisher-Snedecor
                   distribution. Optional, default is
                   ``False``.
    :type scaled: bool
    :returns: Mahalanobis distance computed based on the GP
              predictions on the validation data. If a
              multiple outputs are used, then returns a
              numpy array of shape ``(n_emulators,)``
              holding the Mahalanobis distance for each
              target.
    :rtype: ndarray
    """

    pivot_errors = pivoted_errors(gp, valid_inputs, valid_targets)

    if isinstance(gp, GaussianProcessBase):
        errors = pivot_errors[0]
    else:
        errors = np.array([err[0] for err in pivot_errors])

    M = np.sum(errors**2, axis=-1)

    if scaled:
        expected_dists = generate_mahal_dist(gp, valid_inputs)

        if isinstance(gp, GaussianProcessBase):
            M_iter = [M]
            expected_dists_iter = [expected_dists]
        else:
            M_iter = M
            expected_dists_iter = expected_dists

        M_out = []
        for M_val, dist in zip(M_iter, expected_dists_iter):
            mean, var = dist.stats()
            M_out.append((M_val - mean) / np.sqrt(var))

        M = np.array(M_out)

        if isinstance(gp, GaussianProcessBase):
            M = M.squeeze(axis=0)

    return M


def generate_mahal_dist(gp, valid_inputs):
    """Generate the Expected Distribution for the Mahalanobis Distance

    Convenience function for generating a ``scipy.stats.f`` object
    appropriate for the expected Mahalanobis distribution. If a
    ``MultiOutputGP`` object is provided, then a list of distributions
    will be returned. In all cases, the parameters will be "frozen"
    as appropriate for the data.

    :param gp: A fit ``GaussianProcess`` or ``MultiOutputGP``
               object.
    :type gp: ``GaussianProcess`` or ``MultiOutputGP``
    :param valid_inputs: Input points at which the GP will
                         be validated. Must correspond to
                         the appropriate inputs to the
                         provided GP.
    :type valid_inputs: ndarray
    :returns: ``scipy.stats`` distribution or list of distributions.
    :rtype: scipy.stats.rv_continuous or list
    """

    if isinstance(gp, GaussianProcessBase):
        emulators = [gp]
    elif isinstance(gp, MultiOutputGPBase):
        emulators = gp.emulators
    else:
        raise TypeError("Provided GP is not a GaussianProcess or MultiOutputGP")

    n_valid = len(gp._process_inputs(valid_inputs))

    outdists = []
    for em in emulators:
        outdists.append(f(dfn=n_valid, dfd=em.n - em.n_mean - 2, scale=n_valid))

    if len(outdists) == 1:
        return outdists[0]
    else:
        return outdists


def compute_errors(gp, valid_inputs, valid_targets, method):
    """General pattern for computing GP validation errors

    Implements the general pattern of computing errors. User
    must provide a GP to be validated, the validation inputs,
    and the validation targets. Additionally, a class
    must be provided in the ``method`` argument that contains
    the information needed to compute the ordering of the
    errors and the errors themselves. This class must derive
    from the ``Errors`` class and provide the following: it
    must have a boolean class attribute ``full_cov`` that
    determines if the full covariance or the variances are
    needed to compute the error, and a ``__call__`` method
    that accepts three arguments (the target values, the
    mean predicted value, and the variance/covariance of the
    predictions). This function must return a tuple containing
    two numpy arrays: the first contains the error values and
    the second containing the integer indices that indicate
    ordering the validation errors. See the provided classes
    ``StandardErrors`` and ``PivotErrors`` for examples.

    Alternatively, this function can be called using any
    of the following strings for the method argument
    (all strings will be transformed to lower case):
    ``'standard'``, ``standarderrors``, ``'pivot'``,
    ``'pivoterrors``, or the ``StandardErrors`` or
    ``PivotErrors`` classes.

    See also the convenience functions implementing
    standard and pivoted errors.

    ``gp`` must be a fit ``GaussianProcess`` or ``MultiOutputGP``
    object, ``valid_inputs`` must be valid input data to the
    GP/MOGP, and ``valid_targets`` must be valid target data of
    the appropraite shape for the GP/MOGP.

    Returns a tuple (GP) or a list of tuples (MOGP). Each tuple
    applies to a single output and contains two 1D numpy arrays.
    The first array holds the errors, and the second holds
    integer indices indicating the order of the errors (to
    unscramble the inputs, index the inputs using this array of
    integers).

    :param gp: A fit ``GaussianProcess`` or ``MultiOutputGP``
               object. If the GP/MOGP has not been fit, a
               ``ValueError`` will be raised.
    :type gp: ``GaussianProcess`` or ``MultiOutputGP``
    :param valid_inputs: Input points at which the GP will
                         be validated. Must correspond to
                         the appropriate inputs to the
                         provided GP.
    :type valid_inputs: ndarray
    :param valid_targets: Target points at which the GP will
                          be validated. Must correspond to
                          the appropriate target shape for
                          the provided GP.
    :type valid_targets: ndarray
    :param method: Class implementing the error computation
                   method (see above) or a string indicating
                   the method of computing the errors.
    :type method:
    :returns: A tuple holding two 1D numpy arrays of length
              ``n_valid`` or a list of such tuples. The
              first array holds the correlated errors. The
              second array holds the integer index values that
              indicate the ordering of the errors. If a
              ``GaussianProcess`` is provided, a single tuple
              will be returned, while if a ``MultiOutputGP``
              is provided, the return value will be a list
              of length ``n_emulators``.
    :rtype: tuple or list of tuples
    """

    if isinstance(method, str):
        if method.lower in ["standard", "standarderrors"]:
            methodclass = StandardErrors()
        elif method.lower in ["pivot", "pivoterrors"]:
            methodclass = PivotErrors()
        else:
            raise ValueError("Bad value for error method in compute_errors")
    else:
        methodclass = method

    assert issubclass(type(methodclass), Errors), "method must be a subclass of Errors"

    _check_valid_data(gp, valid_inputs, valid_targets)

    pred_iter = _create_iterable_preds(
        gp, valid_inputs, valid_targets, full_cov=methodclass.full_cov
    )

    errors = []

    for target, meanval, covval in pred_iter:
        errors.append(methodclass(target, meanval, covval))

    if isinstance(gp, GaussianProcessBase):
        return errors[0]
    else:
        return errors


def standard_errors(gp, valid_inputs, valid_targets):
    """Compute standard errors on a validation dataset

    Given a fit GP and a set of inputs and targets for validation,
    compute the standard errors (number of standard devations
    between the true and predicted values). Numbers are left
    signed to designate the direction of the discrepancy
    (positive values indicate the emulator predictions are
    larger than the true values).

    The standard errors are re-ordered based on the size
    of the predictive variance. This is done to be
    consistent with the interface for the pivoted errors.
    This can also be useful as a heuristic to indicate where
    the emulator predictions are most uncertain.

    ``gp`` must be a fit ``GaussianProcess`` or ``MultiOutputGP``
    object, ``valid_inputs`` must be valid input data to the
    GP/MOGP, and ``valid_targets`` must be valid target data of
    the appropraite shape for the GP/MOGP.

    Returns a tuple (GP) or a list of tuples (MOGP). Each tuple
    applies to a single output and contains two 1D numpy arrays.
    The first array holds the errors, and the second holds
    integer indices indicating the order of the errors (to
    unscramble the inputs, index the inputs using this array of
    integers).

    :param gp: A fit ``GaussianProcess`` or ``MultiOutputGP``
               object. If the GP/MOGP has not been fit, a
               ``ValueError`` will be raised.
    :type gp: ``GaussianProcess`` or ``MultiOutputGP``
    :param valid_inputs: Input points at which the GP will
                         be validated. Must correspond to
                         the appropriate inputs to the
                         provided GP.
    :type valid_inputs: ndarray
    :param valid_targets: Target points at which the GP will
                          be validated. Must correspond to
                          the appropriate target shape for
                          the provided GP.
    :type valid_targets: ndarray
    :returns: A tuple holding two 1D numpy arrays of length
              ``n_valid`` or a list of such tuples. The
              first array holds the correlated errors. The
              second array holds the integer index values that
              indicate the ordering of the errors. If a
              ``GaussianProcess`` is provided, a single tuple
              will be returned, while if a ``MultiOutputGP``
              is provided, the return value will be a list
              of length ``n_emulators``.
    :rtype: tuple or list of tuples
    """
    return compute_errors(gp, valid_inputs, valid_targets, method=StandardErrors())


def pivoted_errors(gp, valid_inputs, valid_targets):
    """Compute correlated errors on a validation dataset

    Given a fit GP and a set of inputs and targets for validation,
    compute the correlated errors (number of standard devations
    between the true and predicted values, conditional on the
    errors in decreasing order). Note that because the errors are
    conditional, order matters and thus the errors are treated
    with respect to the largest one. The routine returns both
    the correlated errors and the index ordering of the validation
    points (if a ``GaussianProcess`` is provided) or a list of
    tuples containing the errors and indices indicating the
    ordering of the errors for each target (if a ``MultiOutputGP``
    is provided).

    ``gp`` must be a fit ``GaussianProcess`` or ``MultiOutputGP``
    object, ``valid_inputs`` must be valid input data to the
    GP/MOGP, and ``valid_targets`` must be valid target data of
    the appropraite shape for the GP/MOGP.

    Returns a tuple (GP) or a list of tuples (MOGP). Each tuple
    applies to a single output and contains two 1D numpy arrays.
    The first array holds the errors, and the second holds
    integer indices indicating the order of the errors (to
    unscramble the inputs, index the inputs using this array of
    integers).

    :param gp: A fit ``GaussianProcess`` or ``MultiOutputGP``
               object. If the GP/MOGP has not been fit, a
               ``ValueError`` will be raised.
    :type gp: ``GaussianProcess`` or ``MultiOutputGP``
    :param valid_inputs: Input points at which the GP will
                         be validated. Must correspond to
                         the appropriate inputs to the
                         provided GP.
    :type valid_inputs: ndarray
    :param valid_targets: Target points at which the GP will
                          be validated. Must correspond to
                          the appropriate target shape for
                          the provided GP.
    :type valid_targets: ndarray
    :returns: Tuples holding two 1D numpy arrays of length
              ``n_valid`` or a list of such tuples. The
              first array holds the correlated errors. The
              second array holds the integer index values that
              indicate the ordering of the errors. If a
              ``GaussianProcess`` is provided, a single tuple
              will be returned, while if a ``MultiOutputGP``
              is provided, the return value will be a list
              of length ``n_emulators``.
    :rtype: tuple or list of tuples
    """

    return compute_errors(gp, valid_inputs, valid_targets, method=PivotErrors())


class Errors(object):
    """
    Base class implementing a method for computing errors
    """

    full_cov = False

    def __call__(self, target, mean, cov):
        "compute ordering array and errors"
        raise NotImplementedError


class StandardErrors(Errors):
    """
    Class implementing standard errors

    This class implements the required functionality for computing
    standard errors. This includes setting the class attribute
    ``full_cov=False`` and implementing the ``__call__`` method
    to compute the standard errors and their ordering given target
    values and predicted mean/variance
    """

    full_cov = False

    def __call__(self, target, mean, cov):
        """
        Compute standard errors and ordering array

        Returns the standard errors (in decreasing order) and the
        index ordering of the errors.

        :param target: Validation target values as a 1D numpy array
        :type target: ndarray
        :param mean: Predicted mean values as a 1D numpy array
        :type mean: ndarray
        :param cov: Predicted variance values as a 1D numpy array
        :type cov: ndarray
        :returns: Tuple containing two 1D numpy arrays. The first
                  is the standard errors sorted in descending
                  order, the second is the integer indices
                  indicating this ordering.
        :rtype: tuple
        """

        P = np.argsort(cov)[::-1]
        error = ((mean - target) / np.sqrt(cov))[P]

        return error, P


class PivotErrors(Errors):
    """
    Class implementing pivoted errors

    This class implements the required functionality for computing
    pivoted errors. This includes setting the class attribute
    ``full_cov=True`` and implementing the ``__call__`` method
    to compute the pivoted errors and their ordering given target
    values and predicted mean/variance
    """

    full_cov = True

    def __call__(self, target, mean, cov):
        """
        Compute correlated (pivoted) errors and ordering array

        Returns the correlated pivoted errors (sorted in order
        of decreasing variance conditional on all previous
        errors) and the indices indicating this ordering.

        :param target: Validation target values as a 1D numpy array
        :type target: ndarray
        :param mean: Predicted mean values as a 1D numpy array
        :type mean: ndarray
        :param cov: Predicted covariance values as a 2D numpy array
        :type cov: ndarray
        :returns: Tuple containing two 1D numpy arrays. The first
                  is the correlated errors sorted in descending
                  order conditional on the previous errors, the
                  second is the integer indices indicating this
                  ordering.
        :rtype: tuple
        """

        cov_inv, _ = cholesky_factor(cov, 0.0, "pivot")
        error = cov_inv.solve_L(mean - target)

        return error, cov_inv.P


def _create_iterable_preds(gp, valid_inputs, valid_targets, full_cov=False):
    "Ensure that predictions and targets can be iterated over"

    _check_valid_data(gp, valid_inputs, valid_targets)

    mean, cov, _ = gp.predict(valid_inputs, full_cov=full_cov)

    if isinstance(gp, GaussianProcessBase):
        valid_targets_iter = [valid_targets]
        mean_iter = [mean]
        cov_iter = [cov]
    else:
        valid_targets_iter = valid_targets
        mean_iter = mean
        cov_iter = cov

    return zip(valid_targets_iter, mean_iter, cov_iter)


def _check_valid_data(gp, valid_inputs, valid_targets):
    "Perform some checks on the validation data"

    assert isinstance(
        gp, (GaussianProcessBase, MultiOutputGPBase)
    ), "Must provide a GP to validate"

    valid_inputs = gp._process_inputs(valid_inputs)
    valid_targets = np.array(valid_targets)

    if isinstance(gp, GaussianProcessBase):
        assert valid_targets.ndim == 1, "Targets for a GP must be a 1D array"
        assert (
            valid_targets.shape[0] == valid_inputs.shape[0]
        ), "Bad length for validation targets"
    else:
        assert valid_targets.ndim == 2, "Targets for a MultiOutputGP must be a 2D array"
        assert (
            valid_targets.shape[1] == valid_inputs.shape[0]
        ), "Bad shape for validation targets"
