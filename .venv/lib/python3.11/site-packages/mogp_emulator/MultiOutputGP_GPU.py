import platform
import numpy as np
from mogp_emulator.GaussianProcess import (
    GaussianProcessBase,
    GaussianProcess,
    PredictResult
)

from mogp_emulator.GaussianProcessGPU import (
    GaussianProcessGPU, 
    parse_meanfunc_formula,
    create_prior_params
)
from mogp_emulator.MultiOutputGP import MultiOutputGPBase
from mogp_emulator.MeanFunction import MeanBase
from mogp_emulator.Kernel import SquaredExponential, Matern52
from mogp_emulator.Priors import (
    GPPriors, 
    WeakPrior,
    InvGammaPrior,
    GammaPrior,
    LogNormalPrior
)
from mogp_emulator import LibGPGPU


class MultiOutputGP_GPU(MultiOutputGPBase):
    """Implementation of a multiple-output Gaussian Process Emulator, using the GPU.

    Essentially a parallelized wrapper for the predict method. To fit
    in parallel, use the ``fit_GP_MAP`` routine

    Required arguments are ``inputs`` and ``targets``, both of which
    must be numpy arrays. ``inputs`` can be 1D or 2D (if 1D, assumes
    second axis has length 1). ``targets`` can be 1D or 2D (if 2D,
    assumes a single emulator and the first axis has length 1).

    Optional arguments specify how each individual emulator is
    constructed, including the mean function, kernel, priors, and how
    to handle the nugget. Each argument can take values allowed by the
    base ``GaussianProcess`` class, in which case all emulators are
    assumed to use the same value. Any of these arguments can
    alternatively be a list of values with length matching the number
    of emulators to set those values individually.

    Additional keyword arguments include ``inputdict``, and
    ``use_patsy``, which control how strings are parsed to mean
    functions, if using.

    """

    def __init__(self, inputs, targets, mean=None, kernel="SquaredExponential", priors=None,
                 nugget="adaptive", inputdict={}, use_patsy=True, batch_size=16000):
        """
        Create a new multi-output GP Emulator
        """

        # check input types and shapes, reshape as appropriate for the case of a single emulator
        inputs = np.array(inputs)
        targets = np.array(targets)
        if len(inputs.shape) == 1:
            inputs = np.reshape(inputs, (-1, 1))
        if len(targets.shape) == 1:
            targets = np.reshape(targets, (1, -1))
        elif not (len(targets.shape) == 2):
            raise ValueError("targets must be either a 1D or 2D array")
        if not (len(inputs.shape) == 2):
            raise ValueError("inputs must be either a 1D or 2D array")
        if not (inputs.shape[0] == targets.shape[1]):
            raise ValueError("the first dimension of inputs must be the same length as the second dimension of targets (or first if targets is 1D))")

        # nugget type
        nugsize = 0.
        if nugget == "adaptive": 
            nugtype = LibGPGPU.nugget_type(0)
        elif nugget == "fit":
            nugtype = LibGPGPU.nugget_type(1)
        elif isinstance(nugget, float):
            nugtype = LibGPGPU.nugget_type(2)
            nugsize = nugget
            nugget = "fixed"
        else:
            raise TypeError("nugget parameter must be a string or a non-negative float")

        # set the Mean Function
        if mean is None:
            meanfunc = LibGPGPU.ZeroMeanFunc()
        else:
            if not issubclass(type(mean), MeanBase):
                if isinstance(mean, str):
                    mean = MeanFunction(mean, inputdict, use_patsy)
                else:
                    raise ValueError("provided mean function must be a subclass of MeanBase,"+
                                     " a string formula, or None")
            # at this point, mean will definitely be a MeanBase.  We can call its __str__ and
            # parse this to create an instance of a C++ MeanFunction
            meanfunc = parse_meanfunc_formula(mean.__str__())
            # if we got None back from that function, something went wrong
            if not meanfunc:
                raise ValueError("""
                GPU implementation was unable to parse mean function formula {}.
                """.format(mean.__str__())
                )

        # set the kernel type
        if (isinstance(kernel, str) and kernel == "SquaredExponential") \
           or isinstance(kernel, SquaredExponential):
            kernel_type = LibGPGPU.kernel_type.SquaredExponential
        elif (isinstance(kernel, str) and kernel == "Matern52") \
           or isinstance(kernel, Matern52):
            kernel_type = LibGPGPU.kernel_type.Matern52
        else:
            raise ValueError("GPU implementation requires kernel to be SquaredExponential or Matern52")

        # instantiate the C++ object
        self._mogp_gpu = LibGPGPU.MultiOutputGP_GPU(inputs, targets, batch_size, meanfunc, kernel_type, nugtype, nugsize)

        # deal with priors - first make sure the priors are in the form of a list of correct size
        if isinstance(priors, (GPPriors, dict)) or priors is None:
            priorslist = self.n_emulators*[priors]
        else:
            priorslist = list(priors)
            assert isinstance(priorslist, list), ("priors must be a GPPriors object, None, or arguments to construct " +
                                                  "a GPPriors object or a list of length n_emulators containing the above")

            assert len(priorslist) == self.n_emulators, "Bad length for list provided for priors to MultiOutputGP"

        self._set_priors(priorslist, nugget)


    def _set_priors(self, priorslist, nugget_type):
        """
        Assign a prior to each emulator.  If an entry is None, create default prior.
        """
        for i in range(self.n_emulators):
            if priorslist[i]:
                prior_params = create_prior_params(priorslist=priorslist[i])
            else:
                prior_params = create_prior_params(
                    inputs=self.inputs, n_corr=self.n_corr[i], nugget_type=nugget_type
                )
            self._mogp_gpu.create_priors_for_emulator(
                i, *prior_params
            )

    @property
    def inputs(self):
        return self._mogp_gpu.inputs()

    @property
    def targets(self):
        return np.array(self._mogp_gpu.targets())

    @property
    def D(self):
        return self._mogp_gpu.D()

    @property
    def n(self):
        return self._mogp_gpu.n()

    @property
    def nugget_type(self):
        return self._mogp_gpu.get_nugget_type()

    @property
    def nugget(self):
        return self._mogp_gpu.get_nugget_size()

    @property
    def n_params(self):
        return self._mogp_gpu.n_data_params()

    @property
    def n_emulators(self):
        return self._mogp_gpu.n_emulators()

    @property
    def n_corr(self):
        return self._mogp_gpu.n_corr_params()

    def reset_fit_status(self):
        self._mogp_gpu.reset_fit_status()

    def predict(self, testing, unc=True, deriv=True, include_nugget=True,
                allow_not_fit=False, processes=None):
        """Make a prediction for a set of input vectors

        Makes predictions for each of the emulators on a given set of
        input vectors. The input vectors must be passed as a
        ``(n_predict, D)`` or ``(D,)`` shaped array-like object, where
        ``n_predict`` is the number of different prediction points
        under consideration and ``D`` is the number of inputs to the
        emulator. If the prediction inputs array has shape ``(D,)``,
        then the method assumes ``n_predict == 1``.  The prediction
        points are passed to each emulator and the predictions are
        collected into an ``(n_emulators, n_predict)`` shaped numpy
        array as the first return value from the method.

        Optionally, the emulator can also calculate the uncertainties
        in the predictions (as a variance) and the derivatives with
        respect to each input parameter. If the uncertainties are
        computed, they are returned as the second output from the
        method as an ``(n_emulators, n_predict)`` shaped numpy
        array. If the derivatives are computed, they are returned as
        the third output from the method as an ``(n_emulators,
        n_predict, D)`` shaped numpy array. Finally, if uncertainties
        are computed, the ``include_nugget`` flag determines if the
        uncertainties should include the nugget. By default, this is
        set to ``True``.

        The ``allow_not_fit`` flag determines how the object handles
        any emulators that do not have fit hyperparameter values
        (because fitting presumably failed). By default,
        ``allow_not_fit=False`` and the method will raise an error
        if any emulators are not fit. Passing ``allow_not_fit=True``
        will override this and ``NaN`` will be returned from any
        emulators that have not been fit.

        As with the fitting, this computation can be done
        independently for each emulator and thus can be done in
        parallel.

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
        :param deriv: (optional) Flag indicating if the derivatives
                      are to be computed.  If ``False`` the method
                      returns ``None`` in place of the derivative
                      array. Default value is ``True``.
        :type deriv: bool
        :param include_nugget: (optional) Flag indicating if the
                                nugget should be included in the
                                predictive variance. Only relevant if
                                ``unc = True``.  Default is ``True``.
        :type include_nugget: bool
        :param allow_not_fit: (optional) Flag that allows predictions
                              to be made even if not all emulators have
                              been fit. Default is ``False`` which
                              will raise an error if any unfitted
                              emulators are present.
        :type allow_not_fit: bool
        :param processes: (optional) Number of processes to use when
                          making the predictions.  Must be a positive
                          integer or ``None`` to use the number of
                          processors on the computer (default is
                          ``None``)
        :type processes: int or None
        :returns: ``PredictResult`` object holding numpy arrays
                  containing the predictions, uncertainties, and
                  derivatives, respectively. Predictions and
                  uncertainties have shape ``(n_emulators,
                  n_predict)`` while the derivatives have shape
                  ``(n_emulators, n_predict, D)``. If the ``do_unc``
                  or ``do_deriv`` flags are set to ``False``, then
                  those arrays are replaced by ``None``.
        :rtype: PredictResult

        """
        if not allow_not_fit and len(self.get_indices_not_fit()) > 0:
            raise ValueError("Hyperparameters have not been fit for this Gaussian Process")
        testing = np.array(testing)
        if self.D == 1 and testing.ndim == 1:
            testing = np.reshape(testing, (-1, 1))
        elif testing.ndim == 1:
            testing = np.reshape(testing, (1, len(testing)))
        assert testing.ndim == 2, "testing must be a 2D array"

        n_testing, D = np.shape(testing)
        assert D == self.D, "second dimension of testing must be the same as the number of input parameters"

        means = np.zeros([self.n_emulators, n_testing])
        uncs = np.zeros([self.n_emulators, n_testing])
        derivs = np.zeros([self.n_emulators, n_testing, self.D])
        if unc: 
            self._mogp_gpu.predict_variance_batch(testing, means, uncs)
            if include_nugget:
                uncs += self.nugget
        else:
            self._mogp_gpu.predict_batch(testing, means)
        if deriv:
            self._mogp_gpu.predict_deriv(testing, derivs)
        # if one or more emulators not fit, fill the appropriate
        # rows in the result with NaNs.
        if allow_not_fit and len(self.get_indices_not_fit()) > 0:
            for index in self.get_indices_not_fit():
                means[index,:] = np.nan
                uncs[index,:] = np.nan
                derivs[index,:,:] = np.nan
        return PredictResult(mean=means, unc=uncs, deriv=derivs)

    def __call__(self, testing, process=None):
        """Interface to predict means by calling the object

        A MultiOutputGP object is callable, which makes predictions of
        the mean only for a given set of inputs. Works similarly to
        the same method of the base GP class.
        """

        return self.predict(testing, unc=False, deriv=False, processes=processes)[0]

    def fit(self, thetas):
        """
        Fit all emulators
        :param thetas: hyperparameters for all emulators. 2D array, 
                    must have first dimension of size n_emulators
        :type thetas: np.array
        """
        self._mogp_gpu.fit(thetas)

    def fit_emulator(self, index, theta):
        """
        Fit a specific emulator
        :param index: index of emulator whose hyperparameters we will set 
        :type index: int
        :param theta: hyperparameters for all emulators. 1D array of length n_param 
        :type theta: np.array
        """
        self._mogp_gpu.fit_emulator(index, theta)

    def get_indices_fit(self):
        """Returns the indices of the emulators that have been fit

        When a ``MultiOutputGP`` class is initialized, none of the
        emulators are fit. Fitting is done by passing the object to an
        external fitting routine, which modifies the object to fit the
        hyperparameters. Any emulators where the fitting fails are
        returned to the initialized state, and this method is used to
        determine the indices of the emulators that have succesfully
        been fit.

        Returns a list of non-negative integers indicating the
        indices of the emulators that have been fit.

        :returns: List of integer indicies indicating the emulators
                  that have been fit. If no emulators have been
                  fit, returns an empty list.
        :rtype: list of int

        """
        return  self._mogp_gpu.get_fitted_indices()  

    def get_indices_not_fit(self):
        """Returns the indices of the emulators that have not been fit

        When a ``MultiOutputGP`` class is initialized, none of the
        emulators are fit. Fitting is done by passing the object to an
        external fitting routine, which modifies the object to fit the
        hyperparameters. Any emulators where the fitting fails are
        returned to the initialized state, and this method is used to
        determine the indices of the emulators that have not been fit.

        Returns a list of non-negative integers indicating the
        indices of the emulators that have not been fit.

        :returns: List of integer indicies indicating the emulators
                  that have not been fit. If all emulators have been
                  fit, returns an empty list.
        :rtype: list of int

        """
        return  self._mogp_gpu.get_unfitted_indices() 

    def __str__(self):
        """Returns a string representation of the model

        :returns: A string representation of the model (indicates
                  number of sub-components and array shapes) :rtype:
                  str
        """

        return ("Multi-Output Gaussian Process with:\n"+
                 str(self.n_emulators)+" emulators\n"+
                 str(self.n)+" training examples\n"+
                 str(self.D)+" input variables")
