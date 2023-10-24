import numpy as np
import scipy.stats
from scipy.linalg import LinAlgError
from scipy.optimize import minimize
from multiprocessing import Pool
from functools import partial
import platform

from mogp_emulator.GaussianProcess import GaussianProcessBase, GaussianProcess
from mogp_emulator.GaussianProcessGPU import GaussianProcessGPU
from mogp_emulator.MultiOutputGP_GPU import MultiOutputGP_GPU
from mogp_emulator import LibGPGPU
from mogp_emulator.MultiOutputGP import MultiOutputGP


def fit_GP_MAP(*args, n_tries=15, theta0=None, method="L-BFGS-B",
               skip_failures=True, refit=False, **kwargs):
    """Fit one or more Gaussian Processes by attempting to minimize the
    negative log-posterior

    Fits the hyperparameters of one or more Gaussian Processes by
    attempting to minimize the negative log-posterior multiple times
    from a given starting location and using a particular minimization
    method. The best result found among all of the attempts is
    returned, unless all attempts to fit the parameters result in an
    error (see below).

    The arguments to the method can either be an existing
    ``GaussianProcess`` or ``MultiOutputGP`` instance, or a list of
    arguments to be passed to the ``__init__`` method of
    ``GaussianProcess`` or ``MultiOutputGP`` if more than one output
    is detected.  Keyword arguments for creating a new
    ``GaussianProcess`` or ``MultiOutputGP`` object can either be
    passed as part of the ``*args`` list or as keywords (if present in
    ``**kwargs``, they will be extracted and passed separately to the
    ``__init__`` method).

    If the method encounters an overflow (this can result because the
    parameter values stored are the logarithm of the actual
    hyperparameters to enforce positivity) or a linear algebra error
    (occurs when the covariance matrix cannot be inverted, even with
    additional noise added along the diagonal if adaptive noise was
    selected), the iteration is skipped. If all attempts to find
    optimal hyperparameters result in an error, then the emulator is
    skipped and the parameters are reset to ``None``. By default, a
    warning will be printed to the console if this occurs for
    ``MultiOutputGP`` fitting, while an error will be raise for
    ``GaussianProcess`` fitting. This behavior can be changed to raise
    an error for ``MultiOutputGP`` fitting by passing the kwarg
    ``skip_failures=False``. This default behavior is chosen because
    ``MultiOutputGP`` fitting is often done in a situation where human
    review of all fit emulators is not possible, so the fitting
    routine skips over failures and then flags those that failed for
    further review.

    For ``MultiOutputGP`` fitting, by default the routine assumes that
    only GPs that currently do not have hyperparameters fit need to be
    fit. This behavior is controlled by the ``refit`` keyword, which
    is ``False`` by default. To fit all emulators regardless of their
    current fitting status, pass ``refit=True``. The ``refit``
    argument has no effect on fitting of single ``GaussianProcess``
    objects -- standard ``GaussianProcess`` objects will be fit
    regardless of the current value of the hyperparameters.

    The ``theta0`` parameter is the point at which the first iteration
    will start. If more than one attempt is made, subsequent attempts
    will use random starting points. If you are fitting Multiple
    Outputs, then this argument can take any of the following forms:
    (1) None (random start points for all emulators, which are drawn
    from the prior distribution for each fit parameter), (2) a list of
    numpy arrays or ``NoneTypes`` with length ``n_emulators``, (3) a
    numpy array of shape ``(n_params,)`` or ``(n_emulators,
    n_params)`` which with either use the same start point for all
    emulators or the specified start point for all emulators. Note
    that if you us a numpy array, all emulators must have the same
    number of parameters, while using a list allows more flexibility.

    The user can specify the details of the minimization method, using
    any of the gradient-based optimizers available in
    ``scipy.optimize.minimize``. Any additional parameters beyond the
    method specification can be passed as keyword arguments.

    The function returns a fit ``GaussianProcess`` or
    ``MultiOutputGP`` instance, either the original one passed to the
    function, or the new one created from the included arguments.

    :param ``*args``: Either a single ``GaussianProcess`` or
                      ``MultiOutputGP`` instance, or arguments to be
                      passed to the ``__init__`` method when creating
                      a new ``GaussianProcess`` or ``MultiOutputGP``
                      instance.
    :param n_tries: Number of attempts to minimize the negative
                    log-posterior function.  Must be a positive
                    integer (optional, default is 15)
    :type n_tries: int
    :param theta0: Initial starting point for the first iteration. If
                   present, must be array-like with shape
                   ``(n_params,)`` based on the specific
                   ``GaussianProcess`` being fit. If a
                   ``MultiOutputGP`` is being fit it must be a list of
                   length ``n_emulators`` with each entry as either
                   ``None`` or a numpy array of shape ``(n_params,)``,
                   or a numpy array with shape ``(n_emulators,
                   n_params)`` (note that if the various emulators
                   have different numbers of parameters, the numpy
                   array option will not work).  If ``None`` is given,
                   then a random value is chosen. (Default is
                   ``None``)
    :type theta0: None or ndarray
    :param method: Minimization method to be used. Can be any
                   gradient-based optimization method available in
                   ``scipy.optimize.minimize``. (Default is
                   ``'L-BFGS-B'``)
    :type method: str
    :param skip_failures: Boolean controlling how to handle failures
                          in ``MultiOutputGP`` fitting. If set to
                          ``True``, emulator fits will fail silently
                          without raising an error and provide
                          information on the emulators that failed
                          and the end of fitting. If ``False``, any
                          failed fit will raise a ``RuntimeError``.
                          Has no effect on fitting a single
                          ``GaussianProcess``, which will always
                          raise an error. Optional, default is
                          ``True``.
    :type skip_failures: bool
    :param refit: Boolean indicating if previously fit emulators
                  for ``MultiOutputGP`` objects should be fit again.
                  Optional, default is ``False``. Has no effect on
                  ``GaussianProcess`` fitting, which will be fit
                  irrespective of the current hyperparameter values.
    :type refit: bool
    :param ``**kwargs``: Additional keyword arguments to be passed to
                         ``GaussianProcess.__init__``,
                         ``MultiOutputGP.__init__``, or the
                         minimization routine. Relevant parameters for
                         the GP classes are automatically split out
                         from those used in the minimization
                         function. See available parameters in the
                         corresponding functions for details.
    :returns: Fit GP or Multi-Output GP instance
    :rtype: GaussianProcess or MultiOutputGP or GaussianProcessGPU or MultiOutputGP_GPU

    """

    if len(args) == 1:
        gp = args[0]
        if isinstance(gp, MultiOutputGP):
            gp = _fit_MOGP_MAP(gp, n_tries, theta0, method, refit, **kwargs)
        elif isinstance(gp, GaussianProcess):
            gp = _fit_single_GP_MAP(gp, n_tries, theta0, method, **kwargs)
        elif LibGPGPU.gpu_usable() and isinstance(gp, GaussianProcessGPU):
            gp = _fit_single_GPGPU_MAP(gp, n_tries, theta0, method, **kwargs)
        elif LibGPGPU.gpu_usable() and isinstance(gp, MultiOutputGP_GPU):
            gp = _fit_MOGPGPU_MAP(gp, n_tries, theta0, method, **kwargs)
        else:
            raise TypeError("single arg to fit_GP_MAP must be a GaussianProcess or MultiOutputGP instance")
    elif len(args) < 2:
        raise TypeError("missing required inputs/targets arrays to GaussianProcess")
    else:
        gp_kwargs = {}
        for key in ["mean", "kernel", "priors", "nugget", "inputdict", "use_patsy"]:
            if key in kwargs:
                gp_kwargs[key] = kwargs[key]
                del kwargs[key]
        try:
            gp = GaussianProcess(*args, **gp_kwargs)
            gp = _fit_single_GP_MAP(gp, n_tries, theta0, method, **kwargs)
        except AssertionError:
            try:
                gp = MultiOutputGP(*args, **gp_kwargs)
                gp = _fit_MOGP_MAP(gp, n_tries, theta0, method, **kwargs)
            except AssertionError:
                raise ValueError("Bad values for *args in fit_GP_MAP")
    if isinstance(gp, GaussianProcessBase):
        if (isinstance(gp, GaussianProcess) and gp.theta.get_data() is None) or \
            (isinstance(gp, GaussianProcessGPU) and gp.theta is None):
            raise RuntimeError("GP fitting failed")
    else:
        if len(gp.get_indices_not_fit()) > 0:
            failure_string = "Fitting failed for emulators {}".format(
                    gp.get_indices_not_fit())
            if skip_failures:
                print(failure_string)
            else:
                raise RuntimeError(failure_string)
    return gp

def _fit_single_GPGPU_MAP(gp, n_tries=15, theta0=None, method='L-BFGS-B', **kwargs):
    """Fit hyperparameters using MAP for a single GP in the C++/CUDA implementation
       The optimization is done in C++, this Python function is a wrapper for that.
        Returns a single GP object that has its hyperparameters fit to the MAP value.
    """
    if method not in ["L-BFGS", "L-BFGS-B"]:
        raise NotImplementedError("Unknown method for optimizer - only L-BFGS implemented for GPU")
    n_tries = int(n_tries)
    assert n_tries > 0, "number of attempts must be positive"
    if theta0 is None or len(theta0)==0:
        theta0=np.array([])
    LibGPGPU.fit_GP_MAP(gp._densegp_gpu, n_tries, theta0)
    if not gp.theta.data_has_been_set():
        raise RuntimeError("Fitting did not converge")
    return gp

def _fit_MOGPGPU_MAP(gp, n_tries=15, theta0=None, method='L-BFGS-B', **kwargs):
    """Fit hyperparameters using MAP for a multi-output GP in the C++/CUDA implementation
       The optimization is done in C++, this Python function is a wrapper for that.
        Returns an MOGP object that has its hyperparameters fit to the MAP value.
    """
    if method not in ["L-BFGS", "L-BFGS-B"]:
        raise NotImplementedError("Unknown method for optimizer - only L-BFGS implemented for GPU")
    n_tries = int(n_tries)
    assert n_tries > 0, "number of attempts must be positive"
    if theta0 is None or len(theta0)==0:
        theta0=np.array([])
    LibGPGPU.fit_GP_MAP(gp._mogp_gpu, n_tries, theta0)
    return gp

def _fit_single_GP_MAP(gp, n_tries=15, theta0=None, method='L-BFGS-B', **kwargs):
    """Fit hyperparameters using MAP for a single GP

    Returns a single GP object that has its hyperparameters fit to the
    MAP value.  Accepts keyword arguments passed to scipy's
    minimization routine.

    """

    assert isinstance(gp, GaussianProcessBase)
    n_tries = int(n_tries)
    assert n_tries > 0, "number of attempts must be positive"

    np.seterr(divide = 'raise', over = 'raise', invalid = 'raise')

    logpost_values = []
    theta_values = []

    for i in range(n_tries):
        if i == 0 and not theta0 is None:
            theta = np.array(theta0)
            assert theta.shape == (gp.n_params,), "theta0 must be a 1D array with length n_params"
        else:
            theta = gp.priors.sample()
        try:
            min_dict = minimize(gp.logposterior, theta, method = method,
                                jac = gp.logpost_deriv, options = kwargs)

            min_theta = min_dict['x']
            min_logpost = min_dict['fun']

            logpost_values.append(min_logpost)
            theta_values.append(min_theta)
        except LinAlgError:
            print("Matrix not positive definite, skipping this iteration")
        except FloatingPointError:
            print("Floating point error in optimization routine, skipping this iteration")

    if len(logpost_values) == 0:
        print("Minimization routine failed to return a value")
        gp.theta = None
    else:
        logpost_values = np.array(logpost_values)
        idx = np.argmin(logpost_values)

        gp.fit(theta_values[idx])

    return gp

def _fit_single_GP_MAP_bound(gp, theta0, n_tries, method, **kwargs):
    "fitting function accepting theta0 as an argument for parallelization"

    return _fit_single_GP_MAP(gp, n_tries=n_tries, theta0=theta0, method=method, **kwargs)

def _fit_MOGP_MAP(gp, n_tries=15, theta0=None, method='L-BFGS-B',
                  refit=False, **kwargs):
    """Fit hyperparameters using MAP for multiple GPs in parallel

    Uses Python Multiprocessing to fit GPs in parallel by calling the
    above routine for a single GP for each of the emulators in the
    MOGP class. Returns a MultiOutputGP object where all emulators
    have been fit to the MAP value.

    Routine only fits GPs that have not previously been fit (indicated
    by the hyperparameters being set to ``None`` by default. This can
    be overridden by passing ``refit=True``.

    Accepts a ``processes`` argument (integer or None) as a keyword to
    control the number of subprocesses used to fit the individual GPs
    in parallel. Must be positive. Default is ``None``.

    """

    assert isinstance(gp, MultiOutputGP)

    try:
        processes = kwargs['processes']
        del kwargs['processes']
    except KeyError:
        processes = None

    n_tries = int(n_tries)
    assert n_tries > 0, "n_tries must be a positive integer"

    if theta0 is None:
        theta0 = [ None ]*gp.n_emulators
    else:
        if isinstance(theta0, np.ndarray):
            if theta0.ndim == 1:
                theta0 = [theta0]*gp.n_emulators
            else:
                assert theta0.ndim == 2, "theta0 must be a 1D or 2D array"
                assert theta0.shape[0] == gp.n_emulators, "bad shape for fitting starting points"
        elif isinstance(theta0, list):
            assert len(theta0) == gp.n_emulators, "theta0 must be a list of length n_emulators"

    if not processes is None:
        processes = int(processes)
        assert processes > 0, "number of processes must be positive"

    if refit:
        emulators_to_fit = gp.emulators
        indices_to_fit = list(range(len(gp.emulators)))
        thetavals = theta0
    else:
        indices_to_fit = gp.get_indices_not_fit()
        emulators_to_fit = gp.get_emulators_not_fit()
        thetavals = [ theta0[idx] for idx in indices_to_fit]

    if platform.system() == "Windows":

        fit_MOGP = [fit_GP_MAP(emulator, n_tries=n_tries, theta0=t0, method=method, **kwargs)
                    for (emulator, t0) in zip(emulators_to_fit, thetavals)]
    else:
        with Pool(processes) as p:
            fit_MOGP = p.starmap(partial(_fit_single_GP_MAP_bound, n_tries=n_tries, method=method, **kwargs),
                                 [(emulator, t0) for (emulator, t0) in zip(emulators_to_fit, thetavals)])

    for (idx, em) in zip(indices_to_fit, fit_MOGP):
        gp.emulators[idx] = em

    return gp
