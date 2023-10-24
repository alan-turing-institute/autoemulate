"""
extends GaussianProcess with an (optional) GPU implementation
"""

import os
import re
import numpy as np

from mogp_emulator.Kernel import SquaredExponential, Matern52
from mogp_emulator.MeanFunction import MeanFunction, MeanBase

import mogp_emulator.LibGPGPU as LibGPGPU

from mogp_emulator.GaussianProcess import GaussianProcessBase, PredictResult
from mogp_emulator.Priors import (
    GPPriors, 
    InvGammaPrior, 
    GammaPrior, 
    LogNormalPrior, 
    WeakPrior,
    PriorDist
)

class GPUUnavailableError(RuntimeError):
    """Exception type to use when a GPU, or the GPU library, is unavailable"""
    pass


def ndarray_coerce_type_and_flags(arr):
    """
    Helper function for the GaussianProcessGPU methods that call
    CUDA/C++ functions (those wrapped by _dense_gpgpu) and that take
    numpy arrays as arguments.  Ensures that an array is of the
    correct type for this purpose.

    Takes an array or array-like.

    Returns an ndarray with the same data, that has:
    - dtype of float64
    - flags.writable
    - flags.c_contiguous

    The array returned may reference the original array, be a copy of
    it, or be newly constructed from the argument.
    """

    arr = np.array(arr, copy=False)

    # cast into float64, just in case we were given integers and ensure contiguous (C type)
    arr_contiguous_float64 = np.ascontiguousarray(arr.astype(np.float64, copy=False))
    if not arr_contiguous_float64.flags['WRITEABLE']:
        return np.copy(arr_contiguous_float64)
    else:
        return arr_contiguous_float64


def parse_meanfunc_formula(formula):
    """
    Assuming the formula has already been parsed by the Python MeanFunction interface,
    we expect it to be in a standard form, with parameters denoted by 'c' and variables
    denoted by 'x[dim_index]' where dim_index < D.

    :param formula: string representing the desired mean function formula
    :type formula: str

    :returns: Instance of LibGPGPU.ConstMeanFunc or LibGPGPU.PolyMeanFunc implementing formula,
       or None if the formula could not be parsed, or is not currently implemented in C++ code.
    :rtype: LibGPGPU.ConstMeanFunc or LibGPGPU.PolyMeanFun or None
    """
    # convert to a raw string
    if formula == "c":
        return LibGPGPU.ConstMeanFunc()
    else:
        # see if the formula is a string representation of a number
        try:
            m = float(formula)
            return LibGPGPU.FixedMeanFunc(m)
        except:
            pass
    # if we got here, we hopefully have a parse-able formula
    terms = formula.split("+")
    def find_index_and_power(term):
        variables = re.findall(r"x\[[\d+]\]",term)
        if len(variables) == 0:
            # didn't find a non-const term
            return None
        indices = [int(re.search(r"\[([\d+])\]", v).groups()[0]) for v in variables]
        if indices.count(indices[0]) != len(indices):
            raise NotImplementedError("Cross terms, e.g. x[0]*x[1] not implemented in GPU version.")
        # first guess at the power to which the index is raised is how many times it appears
        # e.g. if written as x[0]*x[0]
        power = len(indices)
        # however, it's also possible to write 'x[0]^2' or even, 'x[0]*x[0]^2' or even 'x[0]^2*x[0]^2'
        # so look at all the numbers appearing after a '^'.
        more_powers = re.findall(r"\^[\d]+",term)
        more_powers = [int(re.search(r"\^([\d]+)",p).groups()[0]) for p in more_powers]
        # now add these on to the original power number
        # (subtracting one each time, as we already have x^1 implicitly)
        for p in more_powers:
            power += p - 1
        return [indices[0], power]
    indices_powers = []
    for term in terms:
        ip = find_index_and_power(term)
        if ip:
            indices_powers.append(ip)
    if len(indices_powers) > 0:
        return LibGPGPU.PolyMeanFunc(indices_powers)
    else:
        return None

def interpret_nugget(nugget):
    """
    Interpret a provided 'nugget' value (str or float) as the C++ friendly nugget type and nugget size.
    :param: nugget, must be either a str with value 'adaptive' or 'fit,
                    or a non-negative float.
    :returns: 
    :rtype: LibGPGPU.nugget_type, float
    """
    if not isinstance(nugget, (str, float)):
        try:
            nugget = float(nugget)
        except TypeError:
            raise TypeError("nugget parameter must be a string or a non-negative float")

    if isinstance(nugget, str):
        if nugget == "adaptive":
            nugget_type = LibGPGPU.nugget_type.adaptive
        elif nugget == "fit":
            nugget_type = LibGPGPU.nugget_type.fit
        else:
            raise ValueError("nugget must be a string set to 'adaptive', 'fit', or a float")
        nugget_size = 0.
    else:
        # nugget is fixed
        if nugget < 0.:
            raise ValueError("nugget parameter must be non-negative")
        nugget_type = LibGPGPU.nugget_type.fixed
        nugget_size = nugget
    # return info needed to set the nugget on the C++ object
    return nugget_type, nugget_size

def create_prior_params(**kwargs):
    """
    Extract the parameters needed for the C++ DenseGP_GPU instance to 
    create its GPPriors object.
    Can accept either an existing GPPriors object, or the parameters needed
    to construct one, or otherwise will create a default object.
    :param newpriors: set of priors to use
    :type newpriors: GPPriors or dict
    :param inputs: the inputs of the GP
    :type inputs: numpy array
    :param n_corr: the number of correlation length parameters
    :type n_corr: int
    :param nugget_type: whether nugget is adaptive, fit, or fixed
    "type nugget_type: LibGPGPU.nugget_type

    :returns: list of values needed to call the C++ constructor
            [n_corr,
             [(corr_i_prior_dist, [corr_i_prior_params])],
             (cov_prior_dist, [cov_prior_params])
             (nug_prior_dist, [nug_prior_params])]
    :rtype: list
    """
    # if we are not given priors object, make default priors given the supplied args.
    # Note that any meanfunction parameters will be given weak priors.
    if all(x in kwargs.keys() for x in ["inputs", "n_corr", "nugget_type"]):
        priors = GPPriors.default_priors(kwargs["inputs"], kwargs["n_corr"], kwargs["nugget_type"])
    elif "newpriors" in kwargs.keys():
        newpriors = kwargs["newpriors"]
        if isinstance(newpriors, GPPriors):
            priors = newpriors
        else:
            try:
                priors = GPPriors(**newpriors)
            except TypeError:
                raise TypeError("Provided arguments for priors are not valid inputs " +
                                "for a GPPriors object.")
    else:
        raise TypeError("Unrecognized keyword arguments for create_prior_params " +
                        " - should be 'newpriors' or ('inputs','n_corr','nugget_type')")

    def get_prior_params(prior):
        """
        for a WeakPrior-derived object, return a tuple (prior_type,[prior_params])
        """
        type_dict = {"InvGammaPrior" : LibGPGPU.prior_type.InvGamma,
                     "GammaPrior" : LibGPGPU.prior_type.Gamma,
                     "LogNormalPrior" : LibGPGPU.prior_type.LogNormal
                }
        prior_as_str = type(prior).__name__
        if prior_as_str in type_dict.keys():
            return (type_dict[prior_as_str], [prior.shape, prior.scale])
        elif (isinstance(prior, WeakPrior) \
            and not isinstance(prior, PriorDist)) \
            or prior == None:
            return (LibGPGPU.prior_type.Weak, [0.,0.])
        else:
            raise TypeError("Unknown prior type {} for C++/GPU implementation".format(type(prior)))

    prior_params = [priors.n_corr]
    prior_params.append([get_prior_params(prior) for prior in priors.corr])
    prior_params.append(get_prior_params(priors.cov))
    prior_params.append(get_prior_params(priors.nugget))
    return prior_params


class GaussianProcessGPU(GaussianProcessBase):
    """
    This class implements the same interface as
    :class:`mogp_emulator.GaussianProcess.GaussianProcess`, but using a GPU if available.
    Will raise a RuntimeError if a CUDA-compatible GPU, GPU-interface library libgpgpu
    could not be found.
    Note that while the class uses a C++/CUDA implementation of the SquaredExponential or
    Matern52 kernels for the "fit" and "predict" methods, the 'kernel' data member (and
    hence the results of e.g. 'gp.kernel.kernel_f(theta)' will be the pure Python versions,
    for compatibility with the interface of the GaussianProcess class.
    """

    def __init__(self, inputs, targets, mean=None, kernel=SquaredExponential(), priors=None,
                 nugget="adaptive", inputdict = {}, use_patsy=True, max_batch_size=2000):

        if not LibGPGPU.HAVE_LIBGPGPU:
            raise RuntimeError("Cannot construct GaussianProcessGPU: "
                               "The GPU library (libgpgpu) could not be loaded")

        elif not LibGPGPU.gpu_usable():
            raise RuntimeError("Cannot construct GaussianProcessGPU: "
                               "A compatible GPU could not be found")

        inputs = ndarray_coerce_type_and_flags(inputs)
        if inputs.ndim == 1:
            inputs = np.reshape(inputs, (-1, 1))
        assert inputs.ndim == 2

        targets = ndarray_coerce_type_and_flags(targets)
        assert targets.ndim == 1
        assert targets.shape[0] == inputs.shape[0]

        self._inputs = inputs
        self._targets = targets
        self._max_batch_size = max_batch_size

        if mean is None:
            self.mean = LibGPGPU.ZeroMeanFunc()
        else:
            if not issubclass(type(mean), MeanBase):
                if isinstance(mean, str):
                    mean = MeanFunction(mean, inputdict, use_patsy)
                else:
                    raise ValueError("provided mean function must be a subclass of MeanBase,"+
                                     " a string formula, or None")

            # at this point, mean will definitely be a MeanBase.  We can call its __str__ and
            # parse this to create an instance of a C++ MeanFunction
            self.mean = parse_meanfunc_formula(mean.__str__())
            # if we got None back from that function, something went wrong
            if not self.mean:
                raise ValueError("""
                GPU implementation was unable to parse mean function formula {}.
                """.format(mean.__str__())
                )
        # set the kernel.
        # Note that for the "kernel" data member, we use the Python instance
        # rather than the C++/CUDA one (for consistency in interface with
        # GaussianProcess class).  However the C++/CUDA version of the kernel is
        # used when calling fit() or predict()
        if (isinstance(kernel, str) and kernel == "SquaredExponential") \
           or isinstance(kernel, SquaredExponential):
            self.kernel_type = LibGPGPU.kernel_type.SquaredExponential
            self.kernel = SquaredExponential()
        elif (isinstance(kernel, str) and kernel == "Matern52") \
           or isinstance(kernel, Matern52):
            self.kernel_type = LibGPGPU.kernel_type.Matern52
            self.kernel = Matern52()
        else:
            raise ValueError("GPU implementation requires kernel to be SquaredExponential or Matern52")

        # the nugget parameter passed to constructor can be str or float,
        # disambiguate it here to pass values to C++ constructor.
        nugget_type, nugget_size = interpret_nugget(nugget)
        self._nugget_type = nugget_type
        self._init_nugget_size = nugget_size
        # instantiate the DenseGP_GPU class
        self._densegp_gpu = None
        self._init_gpu()   
        # set the GPPriors
        self._set_priors(priors)

    @classmethod
    def from_cpp(cls, denseGP_GPU):
        inputs = denseGP_GPU.inputs()
        targets = denseGP_GPU.targets()
        obj = cls.__new__(cls)
        obj._densegp_gpu = denseGP_GPU
        obj._inputs = inputs,
        obj._targets = targets
        obj._nugget_type = denseGP_GPU.get_nugget_type()
        obj.kernel_type = denseGP_GPU.get_kernel_type()
        obj.mean = denseGP_GPU.get_meanfunc()
        return obj



    def _init_gpu(self):
        """
        Instantiate the DenseGP_GPU C++/CUDA class, if it doesn't already exist.
        """
        if not self._densegp_gpu:
            self._densegp_gpu = LibGPGPU.DenseGP_GPU(self._inputs,
                                                     self._targets,
                                                     self._max_batch_size,
                                                     self.mean,
                                                     self.kernel_type,
                                                     self._nugget_type,
                                                     self._init_nugget_size)

    def _set_priors(self, newpriors=None):
        """
        Tell the C++ DenseGP_GPU object to create a priors object with specified parameters

        :param newpriors(optional): Priors object to use (otherwise will use default)
        :type newpriors: GPPriors object, or dict
        """
        if newpriors:
            prior_params = create_prior_params(newpriors=newpriors)
        else:
            prior_params = create_prior_params(
                inputs=self.inputs, n_corr=self.n_corr, nugget_type=self.nugget_type
            )
        self._densegp_gpu.create_gppriors(
            *prior_params
        )         

    @property 
    def priors(self):
        """
        Returns the Python binding to  the C++ GPPriors object, which holds 
        MeanPriors, correlation_length priors, covariance prior, and nugget prior
        """
        return self._densegp_gpu.get_gppriors()

    @property
    def inputs(self):
        """
        Returns inputs for the emulator as a numpy array

        :returns: Emulator inputs, 2D array with shape ``(n, D)``
        :rtype: ndarray
        """
        return self._densegp_gpu.inputs()

    @property
    def targets(self):
        """
        Returns targets for the emulator as a numpy array

        :returns: Emulator targets, 1D array with shape ``(n,)``
        :rtype: ndarray
        """
        return self._densegp_gpu.targets()

    @property
    def n(self):
        """
        Returns number of training examples for the emulator

        :returns: Number of training examples for the emulator object
        :rtype: int
        """
        return self._densegp_gpu.n()

    @property
    def D(self):
        """
        Returns number of inputs (dimensions) for the emulator

        :returns: Number of inputs for the emulator object
        :rtype: int
        """
        return self._densegp_gpu.D()


    @property
    def n_corr(self):
        """
        Returns number of correlation length parameters

        :returns: Number of parameters
        :rtype: int
        """
        return self._densegp_gpu.n_corr()

    @property
    def n_params(self):
        """
        Returns number of hyperparameters

        Returns the number of hyperparameters for the emulator. The number depends on the
        choice of mean function, covariance function, and nugget strategy, and possibly the
        number of inputs for certain choices of the mean function.

        :returns: Number of hyperparameters
        :rtype: int
        """
        return self._densegp_gpu.get_theta().get_n_data() + \
               self._densegp_gpu.get_theta().get_n_mean()

    @property
    def nugget_type(self):
        """
        Returns method used to select nugget parameter

        Returns a string indicating how the nugget parameter is treated, either ``"adaptive"``,
        ``"fit"``, or ``"fixed"``. This is automatically set when changing the ``nugget``
        property.

        :returns: Current nugget fitting method
        :rtype: str
        """
        return self._nugget_type.__str__().split(".")[1]

    @property
    def nugget(self):
        """
        See :func:`mogp_emulator.GaussianProcess.GaussianProcess.nugget`

        Use the value cached in the C++ class, as we can't rely on the Python fit()
        function being called.
        """
        return self._densegp_gpu.get_nugget_size()

    @nugget.setter
    def nugget(self, nugget):
        nugget_type, nugget_size = interpret_nugget(nugget)
        self._nugget_type = nugget_type
        self._densegp_gpu.set_nugget_size(nugget_size)
        self._densegp_gpu.set_nugget_type(nugget_type)
         

    @property
    def theta(self):
        """
        Returns emulator hyperparameters
        see
        :func:`mogp_emulator.GaussianProcess.GaussianProcess.theta`

        :type theta: ndarray
        """
        theta = self._densegp_gpu.get_theta()
        return theta

    @theta.setter
    def theta(self, theta):
        """
        Fits the emulator and sets the parameters (property-based setter
        alias for ``fit``)

        See :func:`mogp_emulator.GaussianProcess.GaussianProcess.theta`

        :type theta: ndarray or GPParams object
        :returns: None
        """
        if theta is None:
            self._densegp_gpu.reset_theta_fit_status()
        else:
            self.fit(theta)

    @property
    def L(self):
        """
        Return the lower triangular Cholesky factor.

        :returns: np.array
        """
        result = np.zeros((self.n, self.n))
        self._densegp_gpu.get_cholesky_lower(result)
        return np.tril(result.transpose())

    @property 
    def Kinv_t(self):
        """
        Return the product of inverse covariance matrix with the target values     

        :returns: np.array
        """
        if not self._densegp_gpu.theta_fit_status():
            return None
        invQt_result = np.zeros(self.n)
        self._densegp_gpu.get_invQt(invQt_result)
        return invQt_result
        
    @property
    def current_logpost(self):
        """
        Return the current value of the log posterior.  This is cached in the C++ class.

        :returns: double
        """
        if not self._densegp_gpu.theta_fit_status():
            return None
        return self.logposterior(self.theta)

    def get_K_matrix(self):
        """
        Returns current value of the inverse covariance matrix as a numpy array.

        Does not include the nugget parameter, as this is dependent on how the
        nugget is fit.
        """
        result = np.zeros((self.n, self.n))
        self._densegp_gpu.get_K(result)
        return result

    def fit(self, theta):
        """
        Fits the emulator and sets the parameters.

        Implements the same interface as
        :func:`mogp_emulator.GaussianProcess.GaussianProcess.fit`
        """
        theta = ndarray_coerce_type_and_flags(theta)

        self._densegp_gpu.fit(theta)
        

    def logposterior(self, theta):
        """
        Calculate the negative log-posterior at a particular value of the hyperparameters

        See :func:`mogp_emulator.GaussianProcess.GaussianProcess.logposterior`

        :param theta: Value of the hyperparameters. Must be array-like with shape ``(n_params,)``
        :type theta: ndarray
        :returns: negative log-posterior
        :rtype: float
        """

        return self._densegp_gpu.get_logpost(theta)

    def logpost_deriv(self, theta):
        """
        Calculate the partial derivatives of the negative log-posterior

        See :func:`mogp_emulator.GaussianProcess.GaussianProcess.logpost_deriv`

        :param theta: Value of the hyperparameters. Must be array-like with shape ``(n_params,)``
        :type theta: ndarray
        :returns: partial derivatives of the negative log-posterior with respect to the
                  hyperparameters (array with shape ``(n_params,)``)
        :rtype: ndarray
        """
        theta = np.array(theta, copy=False)

        assert theta.shape == (self.n_params,), "bad shape for new parameters"

        if self.theta is None or not np.allclose(theta, self.theta.get_data(), rtol=1.e-10, atol=1.e-15):
            self.fit(theta)

        result = np.zeros(self.n_params)
        self._densegp_gpu.logpost_deriv(result)
        return result

    def logpost_hessian(self, theta):
        """
        Calculate the Hessian of the negative log-posterior

        See :func:`mogp_emulator.GaussianProcess.GaussianProcess.logpost_hessian`

        :param theta: Value of the hyperparameters. Must be array-like with shape
                      ``(n_params,)``
        :type theta: ndarray
        :returns: Hessian of the negative log-posterior (array with shape
                  ``(n_params, n_params)``)
        :rtype: ndarray
        """
        raise GPUUnavailableError(
            "The Hessian calculation is not currently implemented in the GPU version of MOGP."
        )


    def predict(self, testing, unc=True, deriv=True, include_nugget=True):
        """
        Make a prediction for a set of input vectors for a single set of hyperparameters.
        This method implements the same interface as
        :func:`mogp_emulator.GaussianProcess.GaussianProcess.predict`
        """

        if not self.theta.data_has_been_set():
            raise ValueError("hyperparameters have not been fit for this Gaussian Process")

        testing = ndarray_coerce_type_and_flags(testing)

        if self.D == 1 and testing.ndim == 1:
            testing = np.reshape(testing, (-1, 1))
        elif testing.ndim == 1:
            testing = np.reshape(testing, (1, len(testing)))
        assert testing.ndim == 2

        n_testing, D = np.shape(testing)

        assert D == self.D

        means = np.zeros(n_testing)

        if unc:
            variances = np.zeros(n_testing)
            for i in range(0, n_testing, self._max_batch_size):
                self._densegp_gpu.predict_variance_batch(
                    testing[i:i+self._max_batch_size],
                    means[i:i+self._max_batch_size],
                    variances[i:i+self._max_batch_size])
            if include_nugget:
                variances += self.nugget
        else:
            for i in range(0, n_testing, self._max_batch_size):
                self._densegp_gpu.predict_batch(
                    testing[i:i+self._max_batch_size],
                    means[i:i+self._max_batch_size])
            variances = None
        if deriv:
            deriv_result = np.zeros((n_testing,self.D))
            for i in range(0, n_testing, self._max_batch_size):
                self._densegp_gpu.predict_deriv(
                    testing[i:i+self._max_batch_size],
                    deriv_result[i:i+self._max_batch_size])
        else:
            deriv_result = None
        return PredictResult(mean=means, unc=variances, deriv=deriv_result)


    def __call__(self, testing):
        """A Gaussian process object is callable: calling it is the same as
        calling `predict` without uncertainty and derivative
        predictions, and extracting the zeroth component for the
        'mean' prediction.
        """
        return (self.predict(testing, unc=False, deriv=False)[0])


    def __str__(self):
        """
        Returns a string representation of the model

        :returns: A string representation of the model
        (indicates number of training examples and inputs)
        :rtype: str
        """
        return ("Gaussian Process with " + str(self.n) + " training examples and " +
                str(self.D) + " input variables")


    ## __setstate__ and __getstate__ for pickling: don't pickle "_dense_gp_gpu",
    ## and instead reinitialize this when unpickling
    ## (Pickling is required to use multiprocessing.)
    def __setstate__(self, state):
        self.__dict__ = state
        self.init_gpu()
        if self._theta is not None:
            self.fit(self._theta)


    def __getstate__(self):
        copy_dict = self.__dict__.copy()
        del copy_dict["_densegp_gpu"]
        copy_dict["_densegp_gpu"] = None
        return copy_dict
