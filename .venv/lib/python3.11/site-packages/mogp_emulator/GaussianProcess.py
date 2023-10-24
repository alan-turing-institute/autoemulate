import numpy as np
from mogp_emulator.Kernel import KernelBase
from mogp_emulator.Priors import GPPriors
from mogp_emulator.GPParams import GPParams, _process_nugget
from scipy import linalg
from scipy.optimize import OptimizeResult
from mogp_emulator.linalg import cholesky_factor, calc_Ainv, calc_mean_params, calc_R
from mogp_emulator.linalg import logdet_deriv, calc_A_deriv

try:
    from patsy import dmatrix, dmatrices, PatsyError
except ImportError:
    raise ImportError("patsy is now a required dependency of mogp-emulator")
import warnings

class GaussianProcessBase(object):
    pass


class GaussianProcess(GaussianProcessBase):
    """Implementation of a Gaussian Process Emulator.

    This class provides a representation of a Gaussian Process
    Emulator. It contains methods for fitting the GP to a given set of
    hyperparameters, computing the negative log marginal likelihood
    plus prior (so negative log posterior) and its derivatives, and
    making predictions on unseen data. Note that routines to estimate
    hyperparameters are not included in the class definition, and are
    instead provided externally to facilitate implementation of high
    performance versions.

    The required arguments to initialize a GP is a set of training
    data consisting of the inputs and targets for each of those
    inputs. These must be numpy arrays whose first axis is of the same
    length. Targets must be a 1D array, while inputs can be 2D or
    1D. If inputs is 1D, then it is assumed that the length of the
    second axis is unity.

    Optional arguments are the particular mean function to use
    (default is zero mean), the covariance kernel to use (default is
    the squared exponential covariance), a list of prior distributions
    for each hyperparameter (default is no prior information on any
    hyperparameters) and the method for handling the nugget parameter.
    The nugget is additional "noise" that is added to the diagonal of
    the covariance kernel as a variance. This nugget can represent
    uncertainty in the target values themselves, or simply be used to
    stabilize the numerical inversion of the covariance matrix. The
    nugget can be fixed (a non-negative float), can be found
    adaptively (by passing the string ``"adaptive"`` to make the noise
    only as large as necessary to successfully invert the covariance
    matrix), can be fit as a hyperparameter (by passing the string
    ``"fit"``), or pivoting can be used to ignore any collinear matrix
    rows and ensure a zero nugget is used (by passing the string
    ``"pivot"``).

    The internal emulator structure involves arrays for the inputs,
    targets, and hyperparameters.  Other useful information are the
    number of training examples ``n``, the number of input parameters
    ``D``, and the number of hyperparameters ``n_params``. These
    parameters can be obtained externally by accessing these
    attributes.

    Example: ::

        >>> import numpy as np
        >>> from mogp_emulator import GaussianProcess
        >>> x = np.array([[1., 2., 3.], [4., 5., 6.]])
        >>> y = np.array([4., 6.])
        >>> gp = GaussianProcess(x, y)
        >>> print(gp)
        Gaussian Process with 2 training examples and 3 input variables
        >>> gp.n
        2
        >>> gp.D
        3
        >>> gp.n_params
        5
        >>> gp.fit(np.zeros(gp.n_params))
        >>> x_predict = np.array([[2., 3., 4.], [7., 8., 9.]])
        >>> gp.predict(x_predict)
        (array([4.74687618, 6.84934016]), array([0.01639298, 1.05374973]),
        array([[8.91363045e-05, 7.18827798e-01, 3.74439445e-16],
               [4.64005897e-06, 3.74191346e-02, 1.94917337e-17]]))

    """
    def __init__(self, inputs, targets, mean=None, kernel='SquaredExponential', priors=None,
                 nugget="adaptive", inputdict={}, use_patsy=True):
        r"""Create a new GaussianProcess Emulator

        Creates a new GaussianProcess Emulator from either the input
        data and targets to be fit and optionally a mean function,
        covariance kernel, and nugget parameter/method.

        Required arguments are numpy arrays ``inputs`` and
        ``targets``, described below.  Additional arguments are
        ``mean`` to specify the mean function (default is ``None`` for
        zero mean), ``kernel`` to specify the covariance kernel
        (default is the squared exponential kernel), ``priors`` to
        indicate prior distributions on the hyperparameters, and
        ``nugget`` to specify how the nugget parameter is handled (see
        below; default is to fit the nugget adaptively).

        ``inputs`` is a 2D array-like object holding the input data,
        whose shape is ``n`` by ``D``, where ``n`` is the number of
        training examples to be fit and ``D`` is the number of input
        variables to each simulation. If ``inputs`` is a 1D array, it
        is assumed that ``D = 1``.

        ``targets`` is the target data to be fit by the emulator, also
        held in an array-like object. This must be a 1D array of
        length ``n``.

        ``prior`` can take several different forms. Passing ``None``
        for this argument will create default priors, which are
        uninformative for any mean, covariance, or nugget parameters
        and are inverse gamma distributions with 99% of their mass
        between the minimum and maximum grid spacing. To choose other
        default distributions (options are lognormal and gamma
        distributions), the user must pass a ``GPPriors`` object.
        The ``GPPriors.default_priors`` class method gives more
        control over the exact distribution choice.

        If default priors are not used, ``priors`` must be a ``GPPriors``
        object or a list of prior distributions. If a list, it must
        have a length of ``n_params`` whose elements are either
        ``Prior``-derived objects or ``None``. Note that there are
        some exceptions to the requirement of the list length, depending
        on the method used for fitting the nugget. Each list
        element is used as the prior for the corresponding parameter
        following the ordering (mean, correlation, covariance, nugget),
        with ``None`` indicating an uninformative prior.  Passing
        the empty list as this argument may be used as an abbreviation
        for a list of ``n_params`` where all list elements are
        ``None``. Note that this means that uninformative priors must
        be explicitly set, rather than being the default!

        ``nugget`` controls how additional noise is added to the
        emulator targets when fitting.  This can be specified in
        several ways. If a string is provided, it can take the values
        of ``"adaptive"`` or ``"fit"``, which indicate that the nugget
        will be chosen in the fitting process. The nugget can also be
        chosen as part of the fitting process, with three options for
        nugget handling: ``"adaptive"`` means that the nugget will be
        made only as large as necessary to invert the covariance
        matrix, ``"fit"`` means that the nugget will be treated as a
        hyperparameter to be optimized, and ``"pivot"`` indicates that
        pivoting will be used to ignore any collinear rows and ensure
        use of a nugget of zero. Alternatively, a non-negative float
        can be used to specify a fixed noise level. If no value is
        specified for the ``nugget`` parameter, ``"adaptive"`` is the
        default.


        :param inputs: Numpy array holding emulator input
                       parameters. Must be 1D with length ``n`` or 2D
                       with shape ``n`` by ``D``, where ``n`` is the
                       number of training examples and ``D`` is the
                       number of input parameters for each output.
                       :type inputs: ndarray
        :param targets: Numpy array holding emulator targets. Must be
                        1D with length ``n``
        :type targets: ndarray
        :param mean: Mean function to be used or ``None``. If not
                     ``None`` (i.e. zero mean), most likely to be
                     a string formula, but can be any type
                     accepted by the `patsy.dmatrix` interface.
                     See the `patsy` documentation for more details.
                     Optional, default is "1" (constant mean).
        :type mean: None, str, or other formula-like object
        :param kernel: Covariance kernel to be used (optional, default
                     is Squared Exponential) Can provide either a
                     ``Kernel`` object or a string matching the kernel
                     type to be used.
        :type kernel: Kernel or str
        :param priors: Priors to be used. Must be a ``GPPriors`` object,
                       a dictionary that can be used to construct a
                       ``GPPriors`` object, or ``None`` to use the
                       default priors.
        :type priors: GPPriors, dict, or None
        :param nugget: Noise to be added to the diagonal, specified as
                     a string or a float.  A non-negative float
                     specifies the noise level explicitly, while a
                     string indicates that the nugget will be found
                     via fitting, either as ``"adaptive"``, ``"fit"``,
                     or ``"pivot"`` (see above for a
                     description). Default is ``"fit"``.
        :type nugget: float or str
        :returns: New ``GaussianProcess`` instance
        :rtype: GaussianProcess

        """
        inputs = self._process_inputs(inputs)

        targets = np.array(targets)
        assert targets.ndim == 1
        assert targets.shape[0] == inputs.shape[0]

        if isinstance(kernel, str):
            try:
                import mogp_emulator.Kernel
                kernel_class = getattr(mogp_emulator.Kernel, kernel)
                kernel = kernel_class()
            except AttributeError:
                raise ValueError("provided kernel '{}' not a supported kernel type".format(kernel))
        if not issubclass(type(kernel), KernelBase):
            raise ValueError("provided kernel is not a subclass of KernelBase")

        self._inputs = inputs
        self._targets = targets

        if not inputdict == {}:
            warnings.warn("The inputdict interface for mean functions has been deprecated. " +
                          "You must input your mean formulae using the x[0] format directly " +
                          "in the formula.", DeprecationWarning)

        if not use_patsy:
            warnings.warn("patsy is now required to parse all formulae and form design " +
                          "matrices in mogp-emulator. The use_patsy=False option will be ignored.",
                          DeprecationWarning)

        self._mean = mean
        self._dm = self.get_design_matrix(self._inputs)

        self.kernel = kernel

        _, self._nugget_type = _process_nugget(nugget)

        self._set_priors(priors)

        self._theta = GPParams(n_mean=self.n_mean,
                               n_corr=self.n_corr,
                               nugget=nugget)

        self.Kinv = None
        self.Kinvt = None
        self.current_logpost = None

    @property
    def inputs(self):
        """
        Returns inputs for the emulator as a numpy array

        :returns: Emulator inputs, 2D array with shape ``(n, D)``
        :rtype: ndarray
        """
        return self._inputs

    @property
    def targets(self):
        """
        Returns targets for the emulator as a numpy array

        :returns: Emulator targets, 1D array with shape ``(n,)``
        :rtype: ndarray
        """
        return self._targets

    @property
    def n(self):
        """
        Returns number of training examples for the emulator

        :returns: Number of training examples for the emulator object
        :rtype: int
        """

        return self.inputs.shape[0]

    @property
    def D(self):
        """
        Returns number of inputs for the emulator

        :returns: Number of inputs for the emulator object
        :rtype: int
        """

        return self.inputs.shape[1]

    @property
    def n_mean(self):
        """Returns number of mean parameters

        :returns: Number of mean parameters
        :rtype: int
        """
        return self._dm.shape[1]

    @property
    def n_corr(self):
        """Returns number of correlation length parameters

        :returns: Number of correlation length parameters
        :rtype: int
        """
        return self.kernel.get_n_params(self._inputs)

    @property
    def n_params(self):
        """Returns number of hyperparameters

        Returns the number of fitting hyperparameters for the
        emulator. The number depends on the choice of covariance
        function, nugget strategy, and possibly the number of
        inputs depending on the covariance function. Note that
        this does not include the mean function, which is fit
        analytically.

        :returns: Number of hyperparameters
        :rtype: int

        """

        return self.theta.n_params

    @property
    def has_nugget(self):
        """Boolean indicating if the GP has a nugget parameter

        :returns: Boolean indicating if GP has a nugget
        :rtype: bool
        """
        return (not self._nugget_type == "pivot")

    @property
    def nugget_type(self):
        """Returns method used to select nugget parameter

        Returns a string indicating how the nugget parameter is
        treated, either ``"adaptive"``, ``"pivot"``, ``"fit"``, or
        ``"fixed"``. This is automatically set when changing the
        ``nugget`` property.

        :returns: Nugget fitting method
        :rtype: str

        """
        return self._nugget_type

    @property
    def nugget(self):
        """Returns emulator nugget parameter

        Returns current value of the nugget parameter. If the nugget
        is to be selected adaptively, by pivoting, or by fitting the
        emulator and the nugget has not been fit, returns ``None``.

        :returns: Current nugget value, either a float or ``None``
        :rtype: float or None
        """

        return self.theta.nugget


    @property
    def theta(self):
        """Returns emulator hyperparameters

        Returns current hyperparameters for the emulator as a ``GPParams``
        object.  If no parameters have been fit, the data for that
        object will be set to ``None``. Note that the number of parameters
        depends on the mean function and whether or not a nugget is fit,
        so the length of this array will vary across instances.

        :returns: Current parameter values as a ``GPParams`` object.
                  If the emulator has not been fit, the parameter
                  values will not have been set.
        :rtype: GPParams

        When parameter values have been set, pre-calculates the matrices
        needed to compute the log-likelihood and its derivatives and
        make subsequent predictions. This is called any time the
        hyperparameter values are changed in order to ensure that
        all the information is needed to evaluate the log-likelihood
        and its derivatives, which are needed when fitting the optimal
        hyperparameters.

        The method computes the mean function and covariance matrix
        and inverts the covariance matrix using the method specified
        by the value of ``nugget``. The factorized matrix and the
        product of the inverse with the difference between the targets
        and the mean are cached for later use, and the negative
        marginal log-likelihood is also cached.  This method has no
        return value, but it does modify the state of the object.

        :param theta: Values of the hyperparameters to use in
                      fitting. Must be a GPParams object, a
                      numpy array with length ``n_params``, or
                      ``None`` to specify no parameters.
        :type theta: GPParams
        """

        return self._theta

    @theta.setter
    def theta(self, theta):
        """
        Fits the emulator and sets the parameters (property-based setter
        alias for ``fit``)

        Pre-calculates the matrices needed to compute the
        log-likelihood and its derivatives and make subsequent
        predictions. This is called any time the hyperparameter values
        are changed in order to ensure that all the information is
        needed to evaluate the log-likelihood and its derivatives,
        which are needed when fitting the optimal hyperparameters.

        The method computes the mean function and covariance matrix
        and inverts the covariance matrix using the method specified
        by the value of ``nugget``. The factorized matrix and the
        product of the inverse with the difference between the targets
        and the mean are cached for later use, and the negative
        marginal log-likelihood is also cached.  This method has no
        return value, but it does modify the state of the object.

        :param theta: Values of the hyperparameters to use in
                      fitting. Must be a `GPParams`` object, a
                      numpy array with length ``n_params``, or
        :type theta: ndarray
        :returns: None
        """

        if theta is None:
            self._theta.set_data(None)
            self.current_logpost = None
            self.Kinv = None
            self.Kinv_t = None
            self.Ainv = None
        else:
            self.fit(theta)

    @property
    def priors(self):
        """
        Specifies the priors using a ``GPPriors`` object. Property returns ``GPPriors``
        object. Can be set with either a ``GPPriors`` object, a dictionary holding
        the arguments needed to construct a ``GPPriors`` object, or ``None`` to
        use the default priors.
        
        If ``None`` is provided, use default priors, which are weak prior information
        for the mean function, Inverse Gamma distributions for the correlation
        lengths and nugget (if the nugget is fit), and weak prior information for
        the covariance. The Inverse Gamma distributions are chosen to put 99% of
        the distribution mass between the minimum and maximum spacing of the inputs.
        More fine-grained control of the default priors can be obtained by
        using the class method of the ``GPPriors`` object.
        
        Note that the default priors are not weak priors -- to use weak prior
        information, the user must explicitly construct a ``GPPriors`` object
        with the appropriate number of parameters and nugget type.
        """
        return self._priors

    def _set_priors(self, newpriors):
        """
        Method for setting the priors
        
        Set the priors to a new ``GPPriors`` object and perform some checks
        for consistency with the number of parameters and nugget type.
        
        Note that this is not a public method, and no setter method for the
        ``priors`` property is provided. This is because the ``GPParams``
        object depends on the choice of priors, and thus setting the
        ``GPParams`` depends on the priors already being set. Calling
        this after the object is initialized could lead to some errors
        when fitting if the new priors change the way the underlying
        parameters are found.
        """
        if newpriors is None:
            self._priors = GPPriors.default_priors(self.inputs, self.n_corr, self.nugget_type)
        elif isinstance(newpriors, GPPriors):
            self._priors = newpriors
        else:
            try:
                self._priors = GPPriors(**newpriors)
            except TypeError:
                raise TypeError("Provided arguments for priors are not valid inputs " +
                                "for a GPPriors object.")

        if self._priors.n_mean > 0:
            assert self._priors.n_mean == self.n_mean
        assert self._priors.n_corr == self.n_corr, "bad number of correlation lengths in new GPPriors object"
        assert self._priors.nugget_type == self.nugget_type, "nugget type of GPPriors object does not match"

    def get_design_matrix(self, inputs):
        """Returns the design matrix for a set of inputs

        For a given set of inputs, compute the design matrix based on the GP
        mean function.
        
        :param inputs: 2D numpy array for input values to be used in computing
                       the design matrix. Second dimension must match the
                       number of dimensions of the input data (``D``).
        """

        inputs = self._process_inputs(inputs)
        assert inputs.shape[1] == self.D, "bad shape for inputs"
        
        if self._mean is None or self._mean == "0" or self._mean == "-1":
            dm = np.zeros((inputs.shape[0], 0))
        elif self._mean == "1" or self._mean == "-0":
            dm = np.ones((inputs.shape[0], 1))
        else:
            try:
                dm = dmatrix(self._mean, data={"x": inputs.T})
            except PatsyError:
                try:
                    y, dm = dmatrices(self._mean, data={"x": inputs.T, "y": np.zeros(inputs.shape[0])})
                except PatsyError:
                    raise ValueError("Provided mean function is invalid")
            dm = np.array(dm)
            if not dm.shape[0] == inputs.shape[0]:
                raise ValueError("Provided design matrix is of the wrong shape")

        return dm

    def get_cov_matrix(self, other_inputs):
        """Computes the covariance matrix for a set of inputs
        
        Compute the covariance matrix for the emulator. Assumes
        the second set of inputs is the inputs on which the GP
        is conditioned. Thus, calling with the inputs returns the
        covariance matrix, while calling with a different set of
        values gives the information needed to make predictions.
        Note that this does not include the nugget, which (if
        relevant) is computed separately.
        
        :param otherinputs: Input values for which the covariance is
                       desired. Must be a 2D numpy array
                       with the second dimension matching ``D``
                       for this emulator.
        :type otherinputs: ndarray
        :returns: Covariance matrix for the provided inputs
                  relative to the emulator inputs. If the
                  ``other_inputs`` array has first dimension
                  of length ``M``, then the returned array
                  will have shape ``(n,M)``
        :rtype:
        """
        other_inputs = self._process_inputs(other_inputs)
        
        return self.theta.cov*self.kernel.kernel_f(self.inputs, other_inputs,
                                                   self.theta.corr_raw)

    def get_K_matrix(self):
        """Returns current value of the covariance matrix
        
        Computes the covariance matrix (covariance of the inputs
        with respect to themselves) as a numpy
        array. Does not include the nugget parameter, as this is
        dependent on how the nugget is fit.

        :returns: Covariance matrix conditioned on the emulator
                  inputs. Will be a numpy array with shape
                  ``(n,n)``.
        :rtype: ndarray
        """
        return self.get_cov_matrix(self.inputs)

    def _process_inputs(self, inputs):
        "Change inputs into an array and reshape if required"

        inputs = np.array(inputs)
        if inputs.ndim == 1:
            if (not hasattr(self, "_inputs") or self.D == 1):
                inputs = np.reshape(inputs, (-1, 1))
            else:
                inputs = np.reshape(inputs, (1, -1))
        assert inputs.ndim == 2, "bad shape for input"
        if hasattr(self, "_inputs"):
            assert (
                inputs.shape[1] == self.D
            ), "second dimension of other inputs must be the same as the number of input parameters"
        
        return inputs

    def _check_theta(self, newtheta):
        """
        Check that thet provided array/GPParams object is correct
        
        Performs a check on the provided new values for the
        hyperparameters. Can accept either a ``GPParams``
        object or a numpy array (which should be a 1D array
        of length ``n_params``). Will raise an ``AssertionError``
        if the conditions are not met.
        
        :param newtheta: New value of parameters to check. Must
                         be a ``GPParams`` object or a numpy
                         array of the appropriate shape.
        :type newtheta: GPParams or ndarray
        :returns: None
        """

        if isinstance(newtheta, GPParams):
            assert self.theta.same_shape(newtheta)
            if not newtheta.mean is None:
                warnings.warn("Setting mean parameters with a GPParams object is " +
                              "not supported. The provided values will be overwritten " +
                              "with the analytical mean solution.")
            self._theta = newtheta
        else:
            newtheta = np.array(newtheta)
            assert self.theta.same_shape(newtheta), "bad shape for hyperparameters"
            self.theta.set_data(newtheta)
    
    def _refit(self, newtheta):
        """Checks if emulator needs to be refit
        
        When computing the log posterior or gradient, the
        inverse of the covariance matrix (among other things)
        is computed and cached. If the parameters have not been
        changed, the cached values can be used. This method
        checks if the values have changed significantly and
        returns a boolean indicating if the parameters have
        changed.
        
        Note that this method only accepts numpy arrays for
        the new value of theta, rather than a ``GPParams``
        object (as can be the case for the ``fit`` method
        or ``theta`` property). This is because this is only
        called when fitting and accepts values from
        ``logposterior`` or ``logpost_deriv``. In other
        cases, the emulator is refit regardless.
        """

        return (self.theta.get_data() is None or
                not np.allclose(newtheta, self.theta.get_data(), rtol=1.e-10, atol=1.e-15))

    def fit(self, theta):
        """Fits the emulator and sets the parameters

        Pre-calculates the matrices needed to compute the
        log-likelihood and its derivatives and make subsequent
        predictions. This is called any time the hyperparameter values
        are changed in order to ensure that all the information is
        needed to evaluate the log-likelihood and its derivatives,
        which are needed when fitting the optimal hyperparameters.

        The method computes the mean function and covariance matrix
        and inverts the covariance matrix using the method specified
        by the value of ``nugget_type``. The factorized matrix, the
        product of the inverse with the difference between the targets
        and the mean are cached for later use, a second inverted matrix
        needed to compute the mean function, and the negative
        marginal log-likelihood are all also cached.  This method has no
        return value, but it does modify the state of the object.

        :param theta: Values of the hyperparameters to use in
                      fitting. Must be a numpy array with length
                      ``n_params`` or a ``GPParams`` object.
        :type theta: ndarray or GPParams
        :returns: None
        """

        self._check_theta(theta)

        m = self.priors.mean.dm_dot_b(self._dm)
        K = self.get_K_matrix()

        self.Kinv, newnugget = cholesky_factor(K, self.theta.nugget, self._nugget_type)
        self.Ainv = calc_Ainv(self.Kinv, self._dm, self.priors.mean)
        
        if self._nugget_type == "adaptive":
            self.theta.nugget = newnugget

        self.Kinv_t = self.Kinv.solve(self.targets - m)
        H_Kinv_t = np.dot(self._dm.T, self.Kinv_t)
        
        self.theta.mean = calc_mean_params(self.Ainv, self.Kinv_t,
                                           self._dm, self.priors.mean)
                                           
        self.Kinv_t_mean = self.Kinv.solve(self.targets - np.dot(self._dm, self.theta.mean))

        if self.priors.mean.has_weak_priors:
            n_coeff = self.n - self.n_mean
        else:
            n_coeff = self.n

        self.current_logpost = 0.5*(np.dot(self.targets - m, self.Kinv_t) -
                                    np.dot(H_Kinv_t, self.Ainv.solve(H_Kinv_t)) +
                                    self.Kinv.logdet() + self.Ainv.logdet() +
                                    self.priors.mean.logdet_cov() + 
                                    n_coeff*np.log(2.*np.pi))

        self.current_logpost -= self._priors.logp(self.theta)


    def logposterior(self, theta):
        """Calculate the negative log-posterior at a particular value of the hyperparameters

        Calculate the negative log-posterior for the given set of
        parameters. Calling this method sets the parameter values and
        computes the needed inverse matrices in order to evaluate the
        log-posterior and its derivatives. In addition to returning
        the log-posterior value, it stores the current value of the
        hyperparameters and log-posterior in attributes of the object.

        :param theta: Value of the hyperparameters. Must be array-like
                      with shape ``(n_data,)``
        :type theta: ndarray
        :returns: negative log-posterior
        :rtype: float

        """

        if self._refit(theta):
            self.fit(theta)

        return self.current_logpost

    def logpost_deriv(self, theta):
        """Calculate the partial derivatives of the negative log-posterior

        Calculate the partial derivatives of the negative
        log-posterior with respect to the hyperparameters. Note that
        this function is normally used only when fitting the
        hyperparameters, and it is not needed to make predictions.

        During normal use, the ``logpost_deriv`` method is called
        after evaluating the ``logposterior`` method. The
        implementation takes advantage of this by reusing cached
        results, as the factorized covariance matrix is expensive to
        compute and is used by the ``logposterior``,
        ``logpost_deriv``, and ``logpost_hessian`` methods.  If the
        function is evaluated with a different set of parameters than
        was previously used to set the log-posterior, the method calls
        ``fit`` (and subsequently resets the cached information).

        :param theta: Value of the hyperparameters. Must be array-like
                      with shape ``(n_data,)``
        :type theta: ndarray
        :returns: partial derivatives of the negative log-posterior
                  with respect to the hyperparameters (array with
                  shape ``(n_data,)``)
        :rtype: ndarray
        """

        if self._refit(theta):
            self.fit(theta)

        partials = np.zeros(self.n_params)

        dKdtheta = self.theta.cov*self.kernel.kernel_deriv(self.inputs, self.inputs,
                                                           self.theta.corr_raw)
        dAdtheta = calc_A_deriv(self.Kinv, self._dm, dKdtheta)
        
        Kinv_H_Ainv_H_Kinv_t = self.Kinv.solve(np.dot(self._dm,
                                                      self.Ainv.solve(np.dot(self._dm.T,
                                                                             self.Kinv_t))))
                                            
        partials[:self.n_corr] = 0.5*(-np.dot(self.Kinv_t, np.dot(dKdtheta, self.Kinv_t).T) +
                                       2.*np.dot(self.Kinv_t,
                                                    np.dot(dKdtheta, Kinv_H_Ainv_H_Kinv_t).T) -
                                       np.dot(Kinv_H_Ainv_H_Kinv_t,
                                                 np.dot(dKdtheta, Kinv_H_Ainv_H_Kinv_t).T) +
                                       logdet_deriv(self.Kinv, dKdtheta) +
                                       logdet_deriv(self.Ainv, dAdtheta))
        
        dKdcov = np.reshape(self.get_K_matrix(), (1, self.n, self.n))
        dAdcov = calc_A_deriv(self.Kinv, self._dm, dKdcov)
        partials[self.n_corr] = 0.5*(-np.dot(self.Kinv_t, np.dot(dKdcov[0], self.Kinv_t)) +
                                      2.*np.dot(self.Kinv_t,
                                                np.dot(dKdcov[0], Kinv_H_Ainv_H_Kinv_t)) -
                                      np.dot(Kinv_H_Ainv_H_Kinv_t,
                                             np.dot(dKdcov[0], Kinv_H_Ainv_H_Kinv_t)) +
                                      logdet_deriv(self.Kinv, dKdcov) +
                                      logdet_deriv(self.Ainv, dAdcov))
                                       
        if self.nugget_type == "fit":
            dKdnugget = np.reshape(np.eye(self.n), (1, self.n, self.n))
            dAdnugget = calc_A_deriv(self.Kinv, self._dm, dKdnugget)
            partials[-1] = 0.5*self.nugget*(-np.dot(self.Kinv_t, self.Kinv_t) +
                                             2.*np.dot(self.Kinv_t,
                                                          Kinv_H_Ainv_H_Kinv_t) -
                                             np.dot(Kinv_H_Ainv_H_Kinv_t,
                                                       Kinv_H_Ainv_H_Kinv_t) +
                                             logdet_deriv(self.Kinv, dKdnugget) +
                                             logdet_deriv(self.Ainv, dAdnugget))

        partials -= self._priors.dlogpdtheta(self.theta)

        return partials

    def logpost_hessian(self, theta):
        """Calculate the Hessian of the negative log-posterior
        
        **NOTE: NOT CURRENTLY SUPPORTED**

        Calculate the Hessian of the negative log-posterior with
        respect to the hyperparameters. Note that this function is
        normally used only when fitting the hyperparameters, and it is
        not needed to make predictions. It is also used to estimate an
        appropriate step size when fitting hyperparameters using the
        lognormal approximation or MCMC sampling.

        When used in an optimization routine, the ``logpost_hessian``
        method is called after evaluating the ``logposterior``
        method. The implementation takes advantage of this by storing
        the inverse of the covariance matrix, which is expensive to
        compute and is used by the ``logposterior`` and
        ``logpost_deriv`` methods as well.  If the function is
        evaluated with a different set of parameters than was
        previously used to set the log-posterior, the method calls
        ``fit`` to compute the needed information and changes the
        cached values.

        :param theta: Value of the hyperparameters. Must be array-like
                      with shape ``(n_params,)``
        :type theta: ndarray
        :returns: Hessian of the negative log-posterior (array with
                  shape ``(n_params, n_params)``)
        :rtype: ndarray

        """

        raise NotImplementedError("Hessian computation is not currently supported")

    def predict(self, testing, unc=True, deriv=False, include_nugget=True, full_cov=False):
        """Make a prediction for a set of input vectors for a single set of hyperparameters

        Makes predictions for the emulator on a given set of input
        vectors. The input vectors must be passed as a ``(n_predict,
        D)``, ``(n_predict,)`` or ``(D,)`` shaped array-like object,
        where ``n_predict`` is the number of different prediction
        points under consideration and ``D`` is the number of inputs
        to the emulator. If the prediction inputs array is 1D and ``D
        == 1`` for the GP instance, then the 1D array must have shape
        ``(n_predict,)``. Otherwise, if the array is 1D it must have
        shape ``(D,)``, and the method assumes ``n_predict == 1``. The
        prediction is returned as an ``(n_predict, )`` shaped numpy
        array as the first return value from the method.

        Optionally, the emulator can also calculate the variances in
        the predictions. If the uncertainties are computed, they are
        returned as the second output from the method, either as an
        ``(n_predict,)`` shaped numpy array if ``full_cov=False``
        or an array with ``(n_predict, n_predict)`` if the full
        covariance is computed (default is only to compute the
        variance, which is the diagonal of the full covariance).
        
        Derivatives have been deprecated due to changes in how the
        mean function is computed, so setting ``deriv=True`` will
        have no effect and will raise a ``DeprecationWarning``.

        The ``include_nugget`` kwarg determines if the predictive
        variance should include the nugget or not. For situations
        where the nugget represents observational error and
        predictions are estimating the true underlying function, this
        should be set to ``False``. However, most other cases should
        probably include the nugget, as the emulator is using it to
        represent some of the uncertainty in the underlying simulator,
        so the default value is ``True``.

        :param testing: Array-like object holding the points where
                        predictions will be made.  Must have shape
                        ``(n_predict, D)`` or ``(D,)`` (for a single
                        prediction)
        :type testing: ndarray
        :param unc: (optional) Flag indicating if the uncertainties
                    are to be computed.  If ``False`` the method
                    returns ``None`` in place of the uncertainty
                    array. Default value is ``True``.
        :type unc: bool
        :param include_nugget: (optional) Flag indicating if the
                               nugget should be included in the
                               predictive variance. Only relevant if
                               ``unc = True``.  Default is ``True``.
        :type include_nugget: bool
        :param full_cov: (optional) Flag indicating if the full
                         covariance should be computed for the
                         uncertainty. Only relevant if ``unc = True``.
                         Default is ``False``.
        :type full_cov: bool
        :returns: ``PredictResult`` object holding numpy arrays with
                  the predictions and uncertainties. Predictions
                  and uncertainties have shape ``(n_predict,)``.
                  If the ``unc`` or flag is set to ``False``, then
                  the ``unc`` array is replaced by ``None``.
        :rtype: PredictResult
        """

        if self.theta.get_data() is None:
            raise ValueError("hyperparameters have not been fit for this Gaussian Process")

        testing = self._process_inputs(testing)

        dmtest = self.get_design_matrix(testing)
        mtest = np.dot(dmtest, self.theta.mean)
        Ktest = self.get_cov_matrix(testing)

        mu = mtest + np.dot(Ktest.T, self.Kinv_t_mean)

        var = None
        if unc:

            Kinv_Ktest = self.Kinv.solve(Ktest)
            R = calc_R(Kinv_Ktest, self._dm, dmtest)
            
            if full_cov:
                sigma_2 = self.theta.cov*self.kernel.kernel_f(
                    testing, testing, self.theta.corr_raw
                )
                
                if include_nugget and not self.nugget_type == "pivot":
                    sigma_2 += np.eye(testing.shape[0])*self.theta.nugget
                
                Linv_Ktest = self.Kinv.solve_L(Ktest)
                LAinv_R = self.Ainv.solve_L(R)
                
                var = (sigma_2 - np.dot(Linv_Ktest.T, Linv_Ktest) +
                                 np.dot(LAinv_R.T, LAinv_R))
            else:
                sigma_2 = self.theta.cov

                if include_nugget and not self.nugget_type == "pivot":
                    sigma_2 += self.theta.nugget
                    
                var = np.maximum(sigma_2 - np.sum(Ktest*Kinv_Ktest, axis=0) +
                                 np.sum(R*self.Ainv.solve(R), axis=0),
                                 0.)

        inputderiv = None
        if deriv:
            warnings.warn("Prediction derivatives have been deprecated and are no longer supported",
                          DeprecationWarning)

        return PredictResult(mean=mu, unc=var, deriv=inputderiv)

    def __call__(self, testing):
        """A Gaussian process object is callable: calling it is the same as
        calling `predict` without uncertainty predictions.

        """
        return (self.predict(testing, unc=False, deriv=False)[0])

    def __str__(self):
        """Returns a string representation of the model

        :returns: A string representation of the model (indicates
                  number of training examples and inputs)
        :rtype: str

        """

        return ("Gaussian Process with " + str(self.n) + " training examples and " +
                str(self.D) + " input variables")

class PredictResult(dict):
    """Prediction results object

    Dictionary-like object containing mean, uncertainty (variance),
    and derivatives with respect to the inputs of an emulator
    prediction. Values can be accessed like a dictionary with keys
    ``'mean'``, ``'unc'``, and ``'deriv'`` (or indices 0, 1, and 2 for
    the mean, uncertainty, and derivative for backwards
    compatability), or using attributes (``p.mean`` if ``p`` is an
    instance of ``PredictResult``). Also supports iteration and
    unpacking with the ordering ``(mean, unc, deriv)`` to be
    consistent with indexing behavior.

    Code is mostly based on scipy's ``OptimizeResult`` class, with
    some additional code to support iteration and integer indexing.

    :ivar mean: Predicted mean for each input point. Numpy array with
                shape ``(n_predict,)``
    :type mean: ndarray

    :ivar unc: Predicted variance for each input point. Numpy array
               with shape ``(n_predict,)``
    :type mean: ndarray

    :ivar deriv: Predicted derivative with respect to the inputs for
                 each input point.  Numpy array with shape
                 ``(n_predict, D)``
    :type deriv: ndarray

    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index == 0:
            self.index += 1
            return self['mean']
        elif self.index == 1:
            self.index += 1
            return self['unc']
        elif self.index == 2:
            self.index += 1
            return self['deriv']
        else:
            raise StopIteration


    def __getitem__(self, key):
        if not isinstance(key, (int, str)):
            raise KeyError(key)
        if key == 0:
            newkey = "mean"
        elif key == 1:
            newkey = "unc"
        elif key == 2:
            newkey = "deriv"
        else:
            newkey = key

        return dict.__getitem__(self, newkey)

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in zip(["mean", "unc", "deriv"], self)])
        else:
            return self.__class__.__name__ + "()"
