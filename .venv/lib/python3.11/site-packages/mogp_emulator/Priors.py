import numpy as np
from scipy.optimize import root
from scipy.special import gammaln
from scipy.linalg import cho_factor, cho_solve
import scipy.stats
from mogp_emulator.GPParams import CovTransform, CorrTransform, GPParams
import warnings

class GPPriors(object):
    """
    Class representing prior distributions on GP Hyperparameters
    
    This class combines together the prior distributions over the
    hyperparameters for a GP. These are separated out into
    ``mean`` (which is a separate ``MeanPriors`` object),
    ``corr`` (a list of distributions for the correlation
    length parameters), ``cov`` for the covariance, and
    ``nugget`` for the nugget. These can be specified when
    initializing a new object using the appropriate kwargs,
    or can be set once a ``GPPriors`` object has already
    been created.
    
    In addition to kwargs for the distributions, an additional
    kwarg ``n_corr`` can be used to specify the number of
    correlation lengths (in the event that ``corr`` is not
    provided). If ``corr`` is specified, then this will override
    ``n_corr`` in determining the number of correlation lengths,
    so if both are provided then ``corr`` is used preferrentially.
    If neither ``corr`` or ``n_corr`` is provided, an exception
    will be raised.
    
    Finally, the nugget type is required to be specified when
    initializing a new object.
    
    :param mean: Priors on mean, must be a ``MeanPriors`` object.
                 Optional, default is ``None`` (indicating weak
                 prior information).
    :type mean: MeanPriors
    :param corr: Priors on correlation lengths, must be a list
                 of prior distributions (objects derived from
                 ``WeakPriors``). Optional, default is ``None``
                 (indicating weak prior information, which will
                 automatically create an appropriate list
                 of ``WeakPriors`` objects of the length specified
                 by ``n_corr``).
    :type corr: list
    :param cov: Priors on covariance. Must be a ``WeakPriors``
                derived object. Optional, default is ``None``
                (indicating weak prior information).
    :type cov: WeakPriors
    :param nugget: Priors on nugget. Only valid if the nugget
                   is fit. Must be a ``WeakPriors`` derived
                   object. Optional, default is ``None``
                   (indicating weak prior information).
    :type nugget: WeakPriors
    :param n_corr: Integer specifying number of correlation lengths.
                   Only used if ``corr`` is not specified. Optional,
                   default is ``None`` to indicate number of
                   correlation lengths is specified by ``corr``.
    :type n_corr: int
    :param nugget_type: String indicating nugget type. Must be
                        ``"fixed"``, ``"adaptive"``, ``"fit"``,
                        or ``"pivot"``. Optional, default is
                        ``"fit"``
    :type nugget_type: str
    """
    def __init__(self, mean=None, corr=None, cov=None, nugget=None, n_corr=None, nugget_type="fit"):
        """Create new ``GPPriors`` object.
        """
        
        if corr is None and n_corr is None:
            raise ValueError("Must provide an argument for either corr or n_corr in GPPriors")
            
        self.mean = mean
        
        self._n_corr = n_corr
        self.corr = corr
        
        self.cov = cov
        
        assert nugget_type in ["fit", "adaptive", "fixed", "pivot"], "Bad value for nugget type in GPPriors"
        self._nugget_type = nugget_type
        self.nugget = nugget
        
    @classmethod
    def default_priors(cls, inputs, n_corr, nugget_type="fit", dist="invgamma"):
        """
        Class Method to create a ``GPPriors`` object with default values
        
        Class method that creates priors with defaults for correlation
        length priors and nugget. For the correlation lengths, the
        values of the inputs are used to infer a distribution that
        puts 99% of the mass between the minimum and maximum grid
        spacing. For the nugget (if fit), a default is used that
        preferrentially uses a small nugget. The mean and covariance
        priors are kept as weak prior information.
        
        :param inputs: Input values on which the GP will be fit. Must
                       be a 2D numpy array with the same restrictions
                       as the inputs to the GP class.
        :type inputs: ndarray
        :param n_corr: Number of correlation lengths. Because some
                       kernels only use a single correlation length,
                       this parameter specifies how to treat the
                       inputs to derive the default correlation
                       length priors. Must be a positive integer.
        :type n_corr: int
        :param nugget_type: String indicating nugget type. Must be
                            ``"fixed"``, ``"adaptive"``, ``"fit"``,
                            or ``"pivot"``. Optional, default is
                            ``"fit"``
        :type nugget_type: str
        :param dist: Distribution to fit to the correlation lengths.
                     Must be either a class derived from ``WeakPriors``
                     with a ``default_prior`` class method, or
                     ``"lognormal"``, ``"gamma"``, or ``"invgamma"``.
                     Default is ``"invgamma"``.
        :type dist: str or WeakPriors derived class
        :
        """
        
        assert nugget_type in ["fit", "adaptive", "fixed", "pivot"], "Bad value for nugget type in GPPriors"
        
        if dist.lower() == "lognormal":
            dist_obj = LogNormalPrior
        elif dist.lower() == "gamma":
            dist_obj = GammaPrior
        elif dist.lower() == "invgamma":
            dist_obj = InvGammaPrior
        else:
            if not isinstance(dist, (LogNormalPrior, GammaPrior, InvGammaPrior)):
                raise TypeError("dist must be a prior distribution to contstruct default priors")
            dist_obj = dist
        
        if inputs.shape[1] == n_corr:
            modified_inputs = np.transpose(inputs)
        elif n_corr == 1:
            modified_inputs = np.reshape(inputs, (1, -1))
        else:
            raise ValueError("Number of correlation lengths not compatible with input array")
        
        priors = [dist_obj.default_prior_corr(param) for param in modified_inputs]
        
        priors_updated = [p if isinstance(p, dist_obj) else InvGammaPrior.default_prior_corr_mode(param)
                          for (p, param) in zip(priors, modified_inputs)]
        
        if nugget_type == "fit":
            nugget = InvGammaPrior.default_prior_nugget()
        else:
            nugget = None
            
        return cls(mean=None, corr=priors_updated, cov=None, nugget=nugget, nugget_type=nugget_type)
    
    @property
    def mean(self):
        """
        Mean Prior information
        
        The mean prior information is held in a ``MeanPriors`` object.
        Can be set using a ``MeanPriors`` object or ``None``
        """
        return self._mean
        
    @mean.setter
    def mean(self, newmean):
        "Setter method for mean"
        
        if newmean is None:
            self._mean = MeanPriors()
        elif isinstance(newmean, MeanPriors):
            self._mean = newmean
        else:
            try:
                self._mean = MeanPriors(*newmean)
            except TypeError:
                raise ValueError("Bad value for defining a MeanPriors object in GPPriors, " +
                                 "argument must be an iterable containing the mean " +
                                 "vector and the covariance as a float/vector/matrix")
                                 
    @property
    def n_mean(self):
        """
        Number of mean parameters
        
        :returns: Number of parameters for the ``MeanPrior`` object. If
                  the mean prior is weak or there is no mean function,
                  returns ``None``.
        :rtype: int or None
        """
        return self.mean.n_params
        
    @property
    def corr(self):
        """
        Correlation Length Priors
        
        Must be a list of distributions/None. When class object is initialized, must
        either set number of correlation parameters explicitly or pass a list of
        prior objects. If only number of parameters, will generate a list of NoneTypes
        of that length (assumes weak prior information). If list provided, will use
        that and override the value of number of correlation parameters.
        
        Can change the length by setting this attribute. n_corr will automatically update.
        """
        return self._corr
        
    @corr.setter
    def corr(self, newcorr):
        "setter method for corr"
        if newcorr is None:
            newcorr = [WeakPrior()]*self.n_corr
        try:
            list(newcorr)
        except TypeError:
            raise TypeError("Correlation priors must be a list of WeakPrior derived objects")
        assert len(newcorr) > 0, "Correlation priors must be a list of nonzero length"
        for d in newcorr:
            if not issubclass(type(d), WeakPrior):
                raise TypeError("Correlation priors must be a list of WeakPrior derived objects")
        self._corr = list(newcorr)
        if not self.n_corr is None and not self.n_corr == len(self._corr):
            print("Length of corr argument differs from specified value of n_corr. " +
                  "Defaulting to the value given by the corr argument.") 
        self._n_corr = len(self._corr)
        
    @property
    def n_corr(self):
        """
        Number of correlation length parameters
        """
        return self._n_corr
    
    @property
    def cov(self):
        """Covariance Scale Priors
        
        Prior distribution on Covariance Scale. Can be set using a ``WeakPriors``
        derived object.
        """
        return self._cov
        
    @cov.setter
    def cov(self, newcov):
        "Setter method for cov"
        if newcov is None:
            newcov = WeakPrior()
        if not issubclass(type(newcov), WeakPrior):
            raise TypeError("Covariance prior must be a WeakPrior derived object")
        self._cov = newcov

    @property
    def nugget_type(self):
        """
        Nugget fitting method for the parent GP.
        """
        return self._nugget_type
    
    @property
    def nugget(self):
        """
        Nugget prior distribution
        
        If a nugget is fit, this determines the prior used. If the nugget
        is not fit, will automatically set this to ``None``.
        """
        return self._nugget
        
    @nugget.setter
    def nugget(self, newnugget):
        "Setter method for nugget"
        if self.nugget_type in ["pivot", "adaptive", "fixed"] and not newnugget is None:
            print("Nugget type does not support prior distribution, setting to None")
            newnugget = None
        if newnugget is None and self.nugget_type == "fit":
            newnugget = WeakPrior()
        if not (newnugget is None or issubclass(type(newnugget), WeakPrior)):
            raise TypeError("Nugget prior must be a WeakPrior derived object or None")
        self._nugget = newnugget
    
    def _check_theta(self, theta):
        """
        Perform checks on a ``GPParams`` object to ensure it matches this ``GPPriors`` object.
        """
        
        if not isinstance(theta, GPParams):
            raise TypeError("theta must be a GPParams object when computing priors in GPPriors")
        assert self.n_corr == theta.n_corr, "Provided GPParams object does not have the correct number of parameters"
        assert self.nugget_type == theta.nugget_type, "Provided GPParams object does not have the correct nugget type"
        assert not theta.get_data() is None, "Provided GPParams object does not have its data set"
    
    def logp(self, theta):
        """
        Compute log probability given a ``GPParams`` object
        
        Takes a ``GPParams`` object, this method computes the
        sum of the log probability of all of the sub-distributions.
        Returns a float.
        
        :param theta: Hyperparameter values at which the log prior is
                      to be computed. Must be a ``GPParams`` object
                      whose attributes match this ``GPPriors`` object.
        :type theta: GPParams
        :returns: Sum of the log probability of all prior distributions
        :rtype: float
        """
        
        self._check_theta(theta)
        
        logposterior = 0.
        
        for dist, val in zip(self._corr, theta.corr):
            logposterior += dist.logp(val)
        
        logposterior += self._cov.logp(theta.cov)
            
        if self.nugget_type == "fit":
            logposterior += self._nugget.logp(theta.nugget)
                
        return logposterior
        
    def dlogpdtheta(self, theta):
        """
        Compute derivative of the log probability given a ``GPParams`` object
        
        Takes a ``GPParams`` object, this method computes the
        derivative of the log probability of all of the
        sub-distributions with respect to the raw hyperparameter
        values. Returns a numpy array of length ``n_params`` (the number
        of fitting parameters in the ``GPParams`` object).
        
        :param theta: Hyperparameter values at which the log prior 
                      derivative is to be computed. Must be a
                      ``GPParams`` object whose attributes match
                      this ``GPPriors`` object.
        :type theta: GPParams
        :returns: Gradient of the log probability. Length will be
                  the value of ``n_params`` of the ``GPParams``
                  object.
        :rtype: ndarray
        """
        
        self._check_theta(theta)
    
        partials = []
        
        for dist, val in zip(self._corr, theta.corr):
            partials.append(dist.dlogpdtheta(val, CorrTransform))
        
        partials.append(self._cov.dlogpdtheta(theta.cov, CovTransform))
            
        if self.nugget_type == "fit":
            partials.append(self._nugget.dlogpdtheta(theta.nugget, CovTransform))
                
        return np.array(partials)
        
    def d2logpdtheta2(self, theta):
        """
        Compute the second derivative of the log probability
        given a ``GPParams`` object
        
        Takes a ``GPParams`` object, this method computes the
        second derivative of the log probability of all of the
        sub-distributions with respect to the raw hyperparameter
        values. Returns a numpy array of length ``n_params`` (the number
        of fitting parameters in the ``GPParams`` object).
        
        :param theta: Hyperparameter values at which the log prior 
                      second derivative is to be computed. Must be a
                      ``GPParams`` object whose attributes match
                      this ``GPPriors`` object.
        :type theta: GPParams
        :returns: Hessian of the log probability. Length will be
                  the value of ``n_params`` of the ``GPParams``
                  object. (Note that since all mixed partials
                  are zero, this returns the diagonal
                  of the Hessian as an array)
        :rtype: ndarray
        """
        
        self._check_theta(theta)
    
        hessian = []
        
        for dist, val in zip(self._corr, theta.corr):
            hessian.append(dist.d2logpdtheta2(val, CorrTransform))

        hessian.append(self._cov.d2logpdtheta2(theta.cov, CovTransform))
            
        if self.nugget_type == "fit":
            hessian.append(self._nugget.d2logpdtheta2(theta.nugget, CovTransform))
                
        return np.array(hessian)
    
    def sample(self):
        """
        Draw a set of samples from the prior distributions
        
        Draws a set of samples from the prior distributions associated with
        this GPPriors object. Used in fitting to initialize the minimization
        algorithm.
        
        :returns: Random draw from each distribution, transformed to the
                  raw hyperparameter values. Will be a numpy array
                  with length ``n_params`` of the associated ``GPParams``
                  object.
        """
        
        sample_pt = []
        
        for dist in self._corr:
            sample_pt.append(dist.sample(CorrTransform))

        sample_pt.append(self._cov.sample(CovTransform))
        
        if self.nugget_type == "fit":
            sample_pt.append(self._nugget.sample(CovTransform))

        return np.array(sample_pt)
        
    def __str__(self):
        return str(self._priors)

class MeanPriors(object):
    """
    Object holding mean priors (mean vector and covariance float/vector/matrix
    assuming a multivariate normal distribution). Includes methods for
    computing the inverse and determinant of the covariance and the inverse
    of the covariance multiplied by the mean.
    
    Note that if weak prior information is provided, or if there is no
    mean function, the methods here will still work correctly given the desired
    calling context.
    
    :param mean: Mean vector of the multivariate normal prior distribution
    :type mean: ndarray
    :param cov: Scalar variance, vector variance, or covariance matrix of the
                covariance of the prior distribution. Must be a float or 1D
                or 2D numpy array.
    :type cov: float or ndarray
    """
    def __init__(self, mean=None, cov=None):
        if mean is None:
            self.mean = None
            if not cov is None:
                warnings.warn("Both mean and cov need to be set to form a valid nontrivial " +
                              "MeanPriors object. mean is not provided, so ignoring the " +
                              "provided cov.")
            self.cov = None
            self.Lb = None
        else:
            self.mean = np.reshape(np.array(mean), (-1,))
            if cov is None:
                raise ValueError("Both mean and cov need to be set to form a valid MeanPriors object")
            self.cov = np.array(cov)
            self.Lb = None
            if self.cov.ndim == 0:
                assert self.cov > 0., "covariance term must be greater than zero in MeanPriors"
            elif self.cov.ndim == 1:
                assert len(self.cov) == len(self.mean), "mean and variances must have the same length in MeanPriors"
                assert np.all(self.cov > 0.), "all variances must be greater than zero in MeanPriors"
            elif self.cov.ndim == 2:
                assert self.cov.shape[0] == len(self.mean), "mean and covariances must have the same shape in MeanPriors"
                assert self.cov.shape[1] == len(self.mean), "mean and covariances must have the same shape in MeanPriors"
                assert np.all(np.diag(self.cov) > 0.), "all covariances must be greater than zero in MeanPriors"
                self.Lb = cho_factor(self.cov)
            else:
                raise ValueError("Bad shape for the covariance in MeanPriors")
    
    @property
    def n_params(self):
        r"""
        Number of parameters associated with the mean
        
        :returns: number of mean parameters (or zero if
                  prior information is weak)
        :rtype: int
        """
        if self.mean is None:
            return 0
        else:
            return len(self.mean)
    
    @property
    def has_weak_priors(self):
        r"""
        Property indicating if the Mean has weak prior information
        
        :returns: Boolean indicating if prior information is weak
        :rtype: bool
        """
        return self.mean is None

    def dm_dot_b(self, dm):
        r"""
        Take dot product of mean with a design matrix
        
        Returns the dot product of a design matrix with
        the prior distribution mean vector. If prior
        information is weak or there is no mean function,
        returns zeros of the appropriate shape.
        
        :param dm: Design matrix, array with shape
                   ``(n, n_mean)``
        :type dm: ndarray or patsy.DesignMatrix
        :returns: dot product of design matrix with
                  prior distribution mean vector.
        :rtype: ndarray
        """
        
        if self.mean is None:
            return np.zeros(dm.shape[0])
        else:
            return np.dot(dm, self.mean)
        
    def inv_cov(self):
        r"""
        Compute the inverse of the covariance matrix
        
        Returns the inverse covariance matrix or zero
        if prior information is weak. Returns a float
        or a 2D numpy array with shape ``(n_mean, n_mean)``.
        
        :returns: Inverse of the covariance matrix or
                  zero if prior information is weak.
                  If the inverse is returned, it will
                  be a numpy array of shape
                  ``(n_mean, n_mean)``.
        :rtype: ndarray or float
        """
        if self.cov is None:
            return 0.
        elif self.cov.ndim < 2:
            inv_cov = np.zeros((len(self.mean), len(self.mean)))
            np.fill_diagonal(inv_cov, np.broadcast_to(1./self.cov, (len(self.mean),)))
            return inv_cov
        else:
            return cho_solve(self.Lb, np.eye(len(self.mean)))
        
    def inv_cov_b(self):
        r"""
        Compute the inverse of the covariance matrix times the mean vector
        
        In the log posterior computations, the inverse of the
        covariance matrix multiplied by the mean is required.
        This method correctly returns zero in the event mean
        prior information is weak.
        
        :returns: Inverse covariance matrix multiplied by the
                  mean of the prior distribution. Returns
                  an array with length of the number of mean
                  parameters or a float (in the event of weak
                  prior information)
        :rtype: ndarray or float
        """
        if self.cov is None:
            return 0.
        elif self.cov.ndim < 2:
            return self.mean/self.cov
        else:
            return cho_solve(self.Lb, self.mean)
            
    def logdet_cov(self):
        r"""
        Compute the log of the determinant of the covariance
        
        Computes the log determininant of the mean prior
        covariance. Correctly returns zero if the prior
        information on the mean is weak.
        
        :returns: Log determinant of the covariance matrix
        :rtype: float
        """
        if self.cov is None:
            return 0.
        elif self.cov.ndim < 2:
            return np.sum(np.log(np.broadcast_to(self.cov, (len(self.mean),))))
        else:
            return 2.*np.sum(np.log(np.diag(self.Lb[0])))
            
    def __str__(self):
        return "MeanPriors with mean = {} and cov = {}".format(self.mean, self.cov)            

class WeakPrior(object):
    r"""
    Base Prior class implementing weak prior information
    
    This was implemented to avoid using ``None`` to signify
    weak prior information, which required many different
    conditionals that made the code clunky. In this
    implementation, all parameters have a prior distribution
    to simplify implementation and clarify the methods
    for computing the log probabilities.
    """
    def logp(self, x):
        r"""
        Computes log probability at a given value
        
        :param x: Value of (transformed) variable
        :type x: float
        :returns: Log probability
        :rtype: float
        """
        return 0.

    def dlogpdx(self, x):
        r"""
        Computes derivative of log probability with respect
        to the transformed variable at a given value
        
        :param x: Value of (transformed) variable
        :type x: float
        :returns: Derivative of Log probability
        :rtype: float
        """
        return 0.
        
    def dlogpdtheta(self, x, transform):
        r"""
        Computes derivative of log probability with respect
        to the raw variable at a given value. Requires
        passing the transform to apply to the variable
        to correctly compute the derivative.
        
        :param x: Value of (transformed) variable
        :type x: float
        :param transform: Transform to apply to the derivative
                          to use the chain rule to compute the
                          derivative. Must be one of ``CorrTransform``
                          or ``CovTransform``.
        :type transform: CorrTransform or CovTransform
        :returns: Derivative of Log probability
        :rtype: float
        """
        return float(self.dlogpdx(x)*transform.dscaled_draw(x))

    def d2logpdx2(self, x):
        r"""
        Computes second derivative of log probability with respect
        to the transformed variable at a given value
        
        :param x: Value of (transformed) variable
        :type x: float
        :returns: Second derivative of Log probability
        :rtype: float
        """
        return 0.
        
    def d2logpdtheta2(self, x, transform):
        r"""
        Computes second derivative of log probability with respect
        to the raw variable at a given value. Requires
        passing the transform to apply to the variable
        to correctly compute the derivative.
        
        :param x: Value of (transformed) variable
        :type x: float
        :param transform: Transform to apply to the derivative
                          to use the chain rule to compute the
                          derivative. Must be one of ``CorrTransform``
                          or ``CovTransform``.
        :type transform: CorrTransform or CovTransform
        :returns: Derivative of Log probability
        :rtype: float
        """
        return float(self.d2logpdx2(x)*transform.dscaled_draw(x)**2 + 
                     self.dlogpdx(x)*transform.d2scaled_draw2(x))
        
    def sample(self, transform=None):
        r"""
        Draws a random sample from the distribution and
        transform to the raw parameter values
        
        :param transform: Transform to apply to the sample.
                          Must be one of ``CorrTransform``
                          or ``CovTransform``. Note that
                          for a ``WeakPrior`` object this
                          argument is optional as it is
                          ignored, though derived classes
                          require this argument.
        :type transform: CorrTransform or CovTransform
        :returns: Raw random sample from the distribution
        :rtype: float
        """
    
        return float(5.*(np.random.rand() - 0.5))


class PriorDist(WeakPrior):
    r"""
    Generic Prior Distribution Object
    
    This implements the generic methods for all non-weak prior
    distributions such as default priors and sampling methods.
    Requires a derived method to implement ``logp``, ``dlogpdx``,
    ``d2logpdx2``, and ``sample_x``.
    """
    @classmethod
    def default_prior(cls, min_val, max_val):
        r"""
        Computes default priors given a min and max val between which
        99% of the mass should be found.
    
        Both min and max must be positive as the supported distributions
        are defined over :math:`[0, +\inf]`
    
        This stabilizes the solution, as it prevents the algorithm
        from getting stuck outside these ranges as the likelihood tends
        to be flat on those areas.
    
        Optionally, can change the distribution to be a lognormal or
        gamma distribution by specifying the ``dist`` argument.
    
        Note that the function assumes only a single input dimension is
        provided. Thus, any input array will be flattened before processing.
    
        If the root-finding algorithm fails, then the function will return
        ``None`` to revert to a flat prior.
        
        :param min_val: Minimum value of the input spacing
        :type min_val: float
        :param max_val: Maximum value of the input spacing
        :type max_val: float
        :returns: Distribution with fit parameters
        :rtype: Type derived from ``PriorDist``
        """

        if cls == InvGammaPrior:
            dist_obj = scipy.stats.invgamma
        elif cls == GammaPrior:
            dist_obj = scipy.stats.gamma
        elif cls == LogNormalPrior:
            dist_obj = scipy.stats.lognorm
        else:
            raise ValueError("Default prior must be invgamma, gamma, or lognormal")
        
        assert min_val > 0., "min_val must be positive for InvGamma, Gamma, or LogNormal distributions"
        assert max_val > 0., "max_val must be positive for InvGamma, Gamma, or LogNormal distributions"
        
        assert min_val < max_val, "min_val must be less than max_val"
    
        def f(x):
            assert len(x) == 2
            cdf = dist_obj(np.exp(x[0]), scale=np.exp(x[1])).cdf
            return np.array([cdf(min_val) - 0.005, cdf(max_val) - 0.995])
        
        result = root(f, np.zeros(2))
    
        if not result["success"]:
            print("Prior solver failed to converge")
            return WeakPrior()
        else:
            return cls(np.exp(result["x"][0]), np.exp(result["x"][1]))

    @classmethod
    def default_prior_corr(cls, inputs):
        r"""
        Compute default priors on a set of inputs for the correlation length
        
        Takes a set of inputs and computes the min and max spacing before
        calling the ``default_prior`` method of the class in question to
        generate a distribution. Used in computing the correlation length
        default prior.
        
        :param inputs: Input values on which the distribution will be fit.
                       Must be a 1D numpy array (note that 2D arrays will
                       be flattened).
        :type inputs: ndarray
        :returns: Prior distribution with fit parameters
        :rtype: PriorDist derived object
        """

        min_val = min_spacing(inputs)
        max_val = max_spacing(inputs)
    
        if min_val == 0. or max_val == 0.:
            print("Too few unique inputs; defaulting to flat priors")
            return WeakPrior()

        return cls.default_prior(min_val, max_val)
        
    def sample_x(self):
        r"""
        Draws a random sample from the distribution
        
        :returns: Transformed random sample from the distribution
        :rtype: float
        """
        raise NotImplementedError("PriorDist does not implement a sampler")
        
    def sample(self, transform):
        r"""
        Draws a random sample from the distribution and
        transform to the raw parameter values
        
        :param transform: Transform to apply to the sample.
                          Must be one of ``CorrTransform``
                          or ``CovTransform``.
        :type transform: CorrTransform or CovTransform
        :returns: Raw random sample from the distribution
        :rtype: float
        """
        
        return transform.inv_transform(self.sample_x())
        

class NormalPrior(PriorDist):
    r"""
    Normal Distribution Prior object

    Admits input values from -inf/+inf.

    Take two parameters: mean and std. Mean can take any numeric value, while std must be positive.
    """
    def __init__(self, mean, std):
        self.mean = mean
        assert std > 0., "std parameter must be positive"
        self.std = std

    def logp(self, x):
        r"""
        Computes log probability at a given value
        
        :param x: Value of (transformed) variable
        :type x: float
        :returns: Log probability
        :rtype: float
        """
        return -0.5*((x - self.mean)/self.std)**2 - np.log(self.std) - 0.5*np.log(2.*np.pi)

    def dlogpdx(self, x):
        r"""
        Computes derivative of log probability with respect
        to the transformed variable at a given value
        
        :param x: Value of (transformed) variable
        :type x: float
        :returns: Derivative of Log probability
        :rtype: float
        """
        return -(x - self.mean)/self.std**2

    def d2logpdx2(self, x):
        r"""
        Computes second derivative of log probability with respect
        to the transformed variable at a given value
        
        :param x: Value of (transformed) variable
        :type x: float
        :returns: Second derivative of Log probability
        :rtype: float
        """
        return -self.std**(-2)
        
    def sample_x(self):
        r"""
        Draws a random sample from the distribution
        
        :returns: Transformed random sample from the distribution
        :rtype: float
        """
        return float(scipy.stats.norm.rvs(size=1, loc=self.mean, scale=self.std))

class LogNormalPrior(PriorDist):
    r"""
    Normal Distribution Prior object

    Admits input values from 0/+inf.

    Take two parameters: shape and scale, both of which must be positive
    """
    def __init__(self, shape, scale):
        assert shape > 0., "shape must be greater than zero"
        assert scale > 0., "scale must be greater than zero"
        self.shape = shape
        self.scale = scale
        
    def logp(self, x):
        r"""
        Computes log probability at a given value
        
        :param x: Value of (transformed) variable
        :type x: float
        :returns: Log probability
        :rtype: float
        """
        assert x > 0
        return (-0.5*(np.log(x/self.scale)/self.shape)**2
                - 0.5*np.log(2.*np.pi) - np.log(x) - np.log(self.shape))
                
    def dlogpdx(self, x):
        r"""
        Computes derivative of log probability with respect
        to the transformed variable at a given value
        
        :param x: Value of (transformed) variable
        :type x: float
        :returns: Derivative of Log probability
        :rtype: float
        """
        assert x > 0.
        return -np.log(x/self.scale)/self.shape**2/x - 1./x
        
    def d2logpdx2(self, x):
        r"""
        Computes second derivative of log probability with respect
        to the transformed variable at a given value
        
        :param x: Value of (transformed) variable
        :type x: float
        :returns: Second derivative of Log probability
        :rtype: float
        """
        assert x > 0.
        return (-1./self.shape**2 + np.log(x/self.scale)/self.shape**2 + 1.)/x**2
        
    def sample_x(self):
        r"""
        Draws a random sample from the distribution
        
        :returns: Transformed random sample from the distribution
        :rtype: float
        """
        return float(scipy.stats.lognorm.rvs(size=1, s=self.shape, scale=self.scale))
    

class GammaPrior(PriorDist):
    r"""
    Gamma Distribution Prior object

    Admits input values from 0/+inf.

    Take two parameters: shape :math:`{\alpha}` and scale :math:`{\beta}`. Both must be positive,
    and they are defined such that

    :math:`{p(x) = \frac{\beta^{-\alpha}x^{\alpha - 1}}{\Gamma(/alpha)} \exp(-x/\beta)}`
    """
    def __init__(self, shape, scale):
        assert shape > 0., "shape parameter must be positive"
        self.shape = shape
        assert scale > 0., "scale parameter must be positive"
        self.scale = scale

    def logp(self, x):
        r"""
        Computes log probability at a given value
        
        :param x: Value of (transformed) variable
        :type x: float
        :returns: Log probability
        :rtype: float
        """
        assert x > 0.
        return (-self.shape*np.log(self.scale) - gammaln(self.shape) +
                (self.shape - 1.)*np.log(x) - x/self.scale)

    def dlogpdx(self, x):
        r"""
        Computes derivative of log probability with respect
        to the transformed variable at a given value
        
        :param x: Value of (transformed) variable
        :type x: float
        :returns: Derivative of Log probability
        :rtype: float
        """
        assert x > 0.
        return (self.shape - 1.)/x - 1./self.scale

    def d2logpdx2(self, x):
        r"""
        Computes second derivative of log probability with respect
        to the transformed variable at a given value
        
        :param x: Value of (transformed) variable
        :type x: float
        :returns: Second derivative of Log probability
        :rtype: float
        """
        assert x > 0.
        return -(self.shape - 1.)/x**2
        
    def sample_x(self):
        r"""
        Draws a random sample from the distribution
        
        :returns: Transformed random sample from the distribution
        :rtype: float
        """
        return float(scipy.stats.gamma.rvs(size=1, a=self.shape, scale=self.scale))

class InvGammaPrior(PriorDist):
    r"""
    Inverse Gamma Distribution Prior object

    Admits input values from 0/+inf.

    Take two parameters: shape :math:`{\alpha}` and scale :math:`{\beta}`. Both must be positive,
    and they are defined such that

    :math:`{p(x) = \frac{\beta^{\alpha}x^{-\alpha - 1}}{\Gamma(/alpha)} \exp(-\beta/x)}`
    
    Note that ``InvGammaPrior`` supports both the usual distribution finding
    methods ``default_prior`` as well as some fallback methods that use
    the mode of the distribution to set the parameters.
    """
    def __init__(self, shape, scale):
        assert shape > 0., "shape parameter must be positive"
        self.shape = shape
        assert scale > 0., "scale parameter must be positive"
        self.scale = scale
    
    @classmethod
    def default_prior_mode(cls, min_val, max_val):
        r"""
        Compute default priors on a set of inputs for the correlation length
        
        In some cases, the default correlation prior can fail to fit
        the distribution to the provided values. This method is
        more stable as it does not attempt to fit the lower
        bound of the distribution but instead fits the mode 
        (which can be analytically related to the distribution
        parameters). The mode is chosen to be the the geometric mean
        of the min/max values and 99.5% of the mass is below the max value.
        This approach can fit distributions to wider ranges of parameters
        and is used as a fallback for correlation lengths and the
        default for the nugget (if the nugget is fit)
        
        :param min_val: Minimum value of the input spacing
        :type min_val: float
        :param max_val: Maximum value of the input spacing
        :type max_val: float
        :returns: InvGammaPrior distribution with fit parameters
        :rtype: InvGammaPrior
        """
    
        assert min_val > 0.
        assert max_val > 0.
        assert min_val < max_val, "min_val must be less than max_val"
    
        mode = np.sqrt(min_val*max_val)

        def f(x):
            a = np.exp(x)
            return scipy.stats.invgamma(a, scale=(1. + a)*mode).cdf(max_val) - 0.995

        result = root(f, 0.)

        if not result["success"]:
            print("Prior solver failed to converge")
            return WeakPrior()
        else:
            a = np.exp(result["x"])
            return cls(a, scale=(1. + a)*mode)
    
    @classmethod
    def default_prior_corr_mode(cls, inputs):
        r"""
        Compute default priors on a set of inputs for the correlation length
        
        Takes a set of inputs and computes the min and max spacing before
        calling the ``default_prior_mode`` method. This method is more stable
        than the standard default and is used as a fallback in the event
        that the usual method fails (which can happen if the inputs have
        too broad a range of spacing values).
        
        :param inputs: Input values on which the distribution will be fit.
                       Must be a 1D numpy array (note that 2D arrays will
                       be flattened).
        :type inputs: ndarray
        :returns: InvGammaPrior distribution with fit parameters
        :rtype: InvGammaPrior
        """
        
        min_val = min_spacing(inputs)
        max_val = max_spacing(inputs)
    
        if min_val == 0. or max_val == 0.:
            print("Too few unique inputs; defaulting to flat priors")
            return WeakPrior()
        
        return cls.default_prior_mode(min_val, max_val)
    
    @classmethod
    def default_prior_nugget(cls, min_val=1.e-8, max_val=1.e-6):
        r"""
        Compute default priors on a set of inputs for the nugget
        
        Computes a distribution with given bounds using the
        ``default_prior_mode`` method. This method is more stable
        than the standard default and is used as a fallback in the event
        that the usual method fails (which can happen if the inputs have
        too broad a range of spacing values). Is well suited for the
        nugget, which in most cases is desired to be small.
        
        :param min_val: Minimum value of the input spacing. Optional,
                        default is ``1.e-8``        
        :type min_val: float
        :param max_val: Maximum value of the input spacing. Optional,
                        default is ``1.e-6``
        :type max_val: float
        :returns: InvGammaPrior distribution with fit parameters
        :rtype: InvGammaPrior
        """
        return cls.default_prior_mode(min_val, max_val)

    def logp(self, x):
        r"""
        Computes log probability at a given value
        
        :param x: Value of (transformed) variable
        :type x: float
        :returns: Log probability
        :rtype: float
        """
        return (self.shape*np.log(self.scale) - gammaln(self.shape) -
                (self.shape + 1.)*np.log(x) - self.scale/x)

    def dlogpdx(self, x):
        r"""
        Computes derivative of log probability with respect
        to the transformed variable at a given value
        
        :param x: Value of (transformed) variable
        :type x: float
        :returns: Derivative of Log probability
        :rtype: float
        """
        return -(self.shape + 1.)/x + self.scale/x**2

    def d2logpdx2(self, x):
        r"""
        Computes second derivative of log probability with respect
        to the transformed variable at a given value
        
        :param x: Value of (transformed) variable
        :type x: float
        :returns: Second derivative of Log probability
        :rtype: float
        """
        return (self.shape + 1)/x**2 - 2.*self.scale/x**3
        
    def sample_x(self):
        r"""
        Draws a random sample from the distribution
        
        :returns: Transformed random sample from the distribution
        :rtype: float
        """
        return float(scipy.stats.invgamma.rvs(size=1, a=self.shape, scale=self.scale))

def max_spacing(input):
    r"""
    Computes the maximum spacing of a particular input
    
    :param input: Input values over which the maximum is to
                  be computed. Must be a numpy array
                  (will be flattened).
    :type input: ndarray
    :returns: Maximum difference between any pair of values
    :rtype: float
    """
    
    input = np.unique(np.array(input).flatten())
    
    if len(input) <= 1:
        return 0.
    
    input_sorted = np.sort(input)
    return input_sorted[-1] - input_sorted[0]

def min_spacing(input):
    r"""
    Computes the median spacing of a particular input
    
    :param input: Input values over which the median is to
                  be computed. Must be a numpy array
                  (will be flattened).
    :type input: ndarray
    :returns: Median spacing of the sorted inputs
    :rtype: float
    """
    
    input = np.unique(np.array(input).flatten())
    
    if len(input) <= 2:
        return 0.
    
    return np.median(np.diff(np.sort(input)))
