from multiprocessing import Pool
import platform
import numpy as np
from mogp_emulator.GaussianProcess import (
    GaussianProcessBase,
    GaussianProcess,
    PredictResult
)
from mogp_emulator.Kernel import KernelBase
from mogp_emulator.Priors import GPPriors
from patsy import ModelDesc

class MultiOutputGPBase(object):
    """Base class for Multi-Output GPs. CPU and GPU versions derive from
    this class.
    """
    pass

class MultiOutputGP(MultiOutputGPBase):
    """Implementation of a multiple-output Gaussian Process Emulator.

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

    """

    def __init__(self, inputs, targets, mean=None, kernel="SquaredExponential", priors=None,
                 nugget="adaptive", inputdict={}, use_patsy=True):
        """
        Create a new multi-output GP Emulator
        """ 
        if not inputdict == {}:
            warnings.warn("The inputdict interface for mean functions has been deprecated. " +
                          "You must input your mean formulae using the x[0] format directly " +
                          "in the formula.", DeprecationWarning)
                          
        if not use_patsy:
            warnings.warn("patsy is now required to parse all formulae and form design " +
                          "matrices in mogp-emulator. The use_patsy=False option will be ignored.")

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

        self._n_emulators = targets.shape[0]
        self._n = inputs.shape[0]
        self._D = inputs.shape[1]

        if not isinstance(mean, list):
            mean = self.n_emulators*[mean]

        assert isinstance(mean, list), "mean must be None, a string, a valid patsy model description, or a list of None/string/mean functions"
        assert len(mean) == self.n_emulators
        
        if any([isinstance(m, ModelDesc) for m in mean]):
            warnings.warn("Specifying mean functions using a patsy ModelDesc does not support parallel " +
                          "fitting and prediction with MultiOutputGPs")

        if isinstance(kernel, str) or issubclass(type(kernel), KernelBase):
            kernel = self.n_emulators*[kernel]

        assert isinstance(kernel, list), "kernel must be a Kernel subclass or a list of Kernel subclasses"
        assert len(kernel) == self.n_emulators

        if isinstance(priors, (GPPriors, dict)) or priors is None:
            priorslist = self.n_emulators*[priors]
        else:
            priorslist = list(priors)
            assert isinstance(priorslist, list), ("priors must be a GPPriors object, None, or arguments to construct " +
                                                  "a GPPriors object or a list of length n_emulators containing the above")

            assert len(priorslist) == self.n_emulators, "Bad length for list provided for priors to MultiOutputGP"

        if isinstance(nugget, (str, float)):
            nugget = self.n_emulators*[nugget]

        assert isinstance(nugget, list), "nugget must be a string, float, or a list of strings and floats"
        assert len(nugget) == self.n_emulators

        self.emulators = [ GaussianProcess(inputs, single_target, m, k, p, n)
                           for (single_target, m, k, p, n) in zip(targets, mean, kernel, priorslist, nugget)]


    @property
    def inputs(self):
        """Full array of emulator inputs
        
        Returns input array (2D array of shape
        ``(n, D)``).
        
        :returns: Array of input values
        :rtype: ndarray
        """
        return self.emulators[0].inputs

    @property
    def targets(self):
        """Full array of emulator targets
        
        Returns target array (2D array of shape
        ``(n_emulators, n)``).
        
        :returns: Array of target values
        :rtype: ndarray
        """
        targetlist = []
        for em in self.emulators:
            targetlist.append(em.targets)
        return np.array(targetlist)

    @property
    def D(self):
        """Number of Dimensions in inputs
        
        Returns number of dimentions of the input data
        
        :returns: Number of dimentions of inputs
        :rtype: int
        """
        return self._D

    @property
    def n(self):
        """Number of training examples in inputs
        
        Returns length of training data.
        
        :returns: Length of training inputs
        :rtype: int
        """
        return self._n

    @property
    def n_params(self):
        """Returns the number of parameters for all emulators
        as a list of integers
        
        :returns: Number of parameters for each emulator
                  as a list of integers
        :rtype: list
        """
        return [em.n_params for em in self.emulators]

    @property
    def n_emulators(self):
        return self._n_emulators

    def reset_fit_status(self):
        """Reset the fit status of all emulators
        """
        for em in self.emulators:
            em.theta = None
            
    def _process_inputs(self, inputs):
        "Obtain inputs that are compatible with underlying GPs"
        
        return self.emulators[0]._process_inputs(inputs)

    def predict(self, testing, unc=True, deriv=False, include_nugget=True,
                full_cov = False, allow_not_fit=False, processes=None):
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
                
        If desired, the full covariance can be computed by
        setting ``full_cov=True``. In that case, the returned
        uncertainty will have shape
        ``(n_emulators, n_predict, n_predict)``. This argument is
        optional and the default is to only compute the variance,
        not the full covariance.
                
        Derivatives have been deprecated due to changes in how the
        mean function is computed, so setting ``deriv=True`` will
        have no effect and will raise a ``DeprecationWarning``.

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
        :param include_nugget: (optional) Flag indicating if the
                                nugget should be included in the
                                predictive variance. Only relevant if
                                ``unc = True``.  Default is ``True``.
        :type include_nugget: bool
        :param full_cov: (optional) Flag indicating if the full
                         predictive covariance should be computed.
                         Only relevant if ``unc = True``.
                         Default is ``False``.
        :type full_cov: bool
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

        testing = np.array(testing)
        if self.D == 1 and testing.ndim == 1:
            testing = np.reshape(testing, (-1, 1))
        elif testing.ndim == 1:
            testing = np.reshape(testing, (1, len(testing)))
        assert testing.ndim == 2, "testing must be a 2D array"

        n_testing, D = np.shape(testing)
        assert D == self.D, "second dimension of testing must be the same as the number of input parameters"
        
        if deriv:
            warnings.warn("Prediction derivatives have been deprecated and are no longer supported",
                          DeprecationWarning)

        if not processes is None:
            processes = int(processes)
            assert processes > 0, "number of processes must be a positive integer"

        if allow_not_fit:
            predict_method = _gp_predict_default_NaN
        else:
            predict_method = GaussianProcess.predict

        serial_predict = (platform.system() == "Windows" or
                          any([isinstance(em._mean, ModelDesc) for em in self.emulators]))

        if serial_predict:
            predict_vals = [predict_method(gp, testing, unc, deriv, include_nugget, full_cov)
                            for gp in self.emulators]
        else:
            with Pool(processes) as p:
                predict_vals = p.starmap(predict_method,
                                         [(gp, testing, unc, deriv, include_nugget, full_cov)
                                          for gp in self.emulators])

        # repackage predictions into numpy arrays

        predict_unpacked, unc_unpacked, deriv_unpacked = [np.array(t) for t in zip(*predict_vals)]

        if not unc:
            unc_unpacked = None
        deriv_unpacked = None

        return PredictResult(mean=predict_unpacked, unc=unc_unpacked, deriv=deriv_unpacked)

    def __call__(self, testing, processes=None):
        """Interface to predict means by calling the object

        A MultiOutputGP object is callable, which makes predictions of
        the mean only for a given set of inputs. Works similarly to
        the same method of the base GP class.
        """

        return self.predict(testing, unc=False, deriv=False, processes=processes)[0]

    def fit(self, thetas):
        """
        Fit all emulators
        
        Fit all emulators given an 2D array of hyperparameter values
        (if all emulators have the same number of parameters) or
        an iterable containing 1D numpy arrays (if the emulators
        have a variable number of hyperparameter values).
        
        Note that this routine does not run in parallel for the CPU
        version.
        
        :param thetas: hyperparameters for all emulators. 2D array
                       or iterable containing 1D numpy arrays, must
                       have first dimension of size ``n_emulators``
        :type thetas: np.array or iterable
        """
        for thetaval, em in zip(thetas, self.emulators):
            em.fit(thetaval)

    def fit_emulator(self, index, theta):
        """
        Fit a specific emulator
        :param index: index of emulator whose hyperparameters we will set 
        :type index: int
        :param theta: hyperparameters for all emulators. 1D array of length n_param 
        :type theta: np.array
        """
        self.emulators[index].fit(theta)


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

        return [idx for (failed_fit, idx)
                in zip([em.theta.get_data() is None for em in self.emulators],
                       list(range(len(self.emulators)))) if not failed_fit]

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

        return [idx for (failed_fit, idx)
                in zip([em.theta.get_data() is None for em in self.emulators],
                       list(range(len(self.emulators)))) if failed_fit]

    def get_emulators_fit(self):
        """Returns the emulators that have been fit

        When a ``MultiOutputGP`` class is initialized, none of the
        emulators are fit. Fitting is done by passing the object to an
        external fitting routine, which modifies the object to fit the
        hyperparameters. Any emulators where the fitting fails are
        returned to the initialized state, and this method is used to
        obtain a list of emulators that have successfully been fit.

        Returns a list of ``GaussianProcess`` objects which have been
        fit (i.e. those which have a current valid set of
        hyperparameters).

        :returns: List of ``GaussianProcess`` objects indicating the
                  emulators that have been fit. If no emulators
                  have been fit, returns an empty list.
        :rtype: list of ``GaussianProcess`` objects

        """

        return [gpem for (failed_fit, gpem)
                in zip([em.theta.get_data() is None for em in self.emulators],
                       self.emulators) if not failed_fit]

    def get_emulators_not_fit(self):
        """Returns the indices of the emulators that have not been fit

        When a ``MultiOutputGP`` class is initialized, none of the
        emulators are fit. Fitting is done by passing the object to an
        external fitting routine, which modifies the object to fit the
        hyperparameters. Any emulators where the fitting fails are
        returned to the initialized state, and this method is used to
        obtain a list of emulators that have not been fit.

        Returns a list of ``GaussianProcess`` objects which have
        not been fit (i.e. those which do not have a current set of
        hyperparameters).

        :returns: List of ``GaussianProcess`` objects indicating the
                  emulators that have not been fit. If all emulators
                  have been fit, returns an empty list.
        :rtype: list of ``GaussianProcess`` objects

        """

        return [gpem for (failed_fit, gpem)
                in zip([em.theta.get_data() is None for em in self.emulators],
                       self.emulators) if failed_fit]


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


def _gp_predict_default_NaN(gp, testing, unc, deriv, include_nugget, full_cov):
    """Prediction method for GPs that defaults to NaN for unfit GPs

    Wrapper function for the ``GaussianProcess`` predict method that
    returns NaN if the GP is not fit. Allows ``MultiOutputGP`` objects
    that do not have all emulators fit to still return predictions
    with unfit emulator predictions replaced with NaN.

    The first argument to this function is the GP that will be used
    for prediction. All other arguments are the same as the
    arguments for the ``predict`` method of ``GaussianProcess``.

    :param gp: The ``GaussianProcess`` object (or related class) for
               which predictions will be made.
    :type gp: GaussianProcess
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
    :param full_cov: (optional) Flag indicating if the full
                     predictive covariance should be computed.
                     Only relevant if ``unc = True``.
                     Default is ``False``.
    :type full_cov: bool
    :returns: Tuple of numpy arrays holding the predictions,
              uncertainties, and derivatives,
              respectively. Predictions and uncertainties have
              shape ``(n_predict,)`` while the derivatives have
              shape ``(n_predict, D)``. If the ``unc`` or
              ``deriv`` flags are set to ``False``, then those
              arrays are replaced by ``None``.
    :rtype: tuple
    """

    assert isinstance(gp, GaussianProcessBase)

    try:
        return gp.predict(testing, unc, deriv, include_nugget)
    except ValueError:

        n_predict = testing.shape[0]

        if unc:
            unc_array = np.array([np.nan]*n_predict)
        else:
            unc_array = None

        if deriv:
            deriv_array = np.reshape(np.array([np.nan]*n_predict*gp.D),
                                     (n_predict, gp.D))
        else:
            deriv_array = None

        return PredictResult(mean =np.array([np.nan]*n_predict),
                             unc=unc_array, deriv=deriv_array)
