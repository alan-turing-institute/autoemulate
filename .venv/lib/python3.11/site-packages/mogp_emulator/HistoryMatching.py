import numpy as np
from mogp_emulator.GaussianProcess import GaussianProcessBase, PredictResult
from mogp_emulator.MultiOutputGP import MultiOutputGPBase

class HistoryMatching(object):
    r"""
    Class containing tools to implement history matching between the outputs of a
    gaussian process and an observation.

    Primary usage is the ``get_implausibility`` method, which calculates the
    implausibility metric (a number of standard deviations between the emulator mean
    and the observation) for a number of query points:

    .. math::
        {I_i(\bar{x_0}) = \frac{|z_i - E(f_i(\bar{x_0}))|}
        {\sqrt{Var\left[z_i - E(f_i(\bar{x_0}))\right]}}}

    The code allows a number of ways to specify variances, and all are summed to
    compute the final variance that determines the implausibility measure.
    Variances on the observation itself can be provided with the observations,
    while variances arising from the GP prediction are included in the emulator
    expectations. Additional variances can be included to account for model
    discrepancy when computing the implausibility metric, and these can be uniform
    for all query points, or vary across different query points if desired. Any
    number of additional variances can be included in the calculation by passing
    them when computing the implausibility.

    Once implausibility is computed, the class can determine query points that
    can be ruled out, as well as points that are not ruled out yet, based on the
    threshold class attribute. See the methods ``get_RO``, ``get_NROY``, and
    ``set_threshold`` for more details.

    Example - implausibility computation for a 1D GP::

        import math
        import numpy as np
        from HistoryMatching import HistoryMatching
        from mogp_emulator import GaussianProcess

        # Set up a gaussian process
        x_training = np.array([[0.],[10.],[20.],[30.],[43.],[50.]])
        def get_y_training(x):
            n_points = len(x)
            f = np.zeros(n_points)
            for i in range(n_points):
                f[i] = np.sin(2*math.pi*x[i] / 50)
            return f
        y_training = get_y_training(x_training)

        gp = GaussianProcess(x_training, y_training)
        np.random.seed(47)
        gp.learn_hyperparameters()

        # Define the observation to which to compare
        obs = [-0.8, 0.0004]

        # Coordinates to predict
        n_rand = 2000
        coords = np.sort(np.random.rand(n_rand),axis=0)*(56)-3
        coords = coords[:,None]

        # Generate Expectations
        expectations = gp.predict(coords)

        # Calculate implausibility
        hm = HistoryMatching(obs=obs, expectations=expectations)
        I = hm.get_implausibility()


    """

    def __init__(self, gp=None, obs=None, coords=None, expectations=None, threshold=3.):
        r"""
        Create a new instance of history matching.

        The primary function of this method is to initialise the ``self.gp``,
        ``.obs``, ``.coords``, ``.expectations``, ``.ndim``, ``.ncoords``,
        ``.threshold``, ``.I``, ``.NROY``, and ``.RO`` variables as placeholders.
        The optional keyword arguments can also be used to set user-defined values for
        various quantities in a single call to ``HistoryMatching()``. Setting sufficient
        numbers of these to allow the number of dimensions and/or number of coordinates to
        be computed also causes these to be set at this stage.

        :param gp: (Optional) ``GaussianProcessBase`` or ``MultiOutputGPBase``
                   object used to make predictions in history matching.
                   Optional, can instead provide predictions directly
                   via the ``expectations`` argument. Default is ``None``
        :type gp: GaussianProcessBase, MultiOutputGP, or None
        :param obs: (Optional) Observations against which the predictions will be
                    compared. If provided, must either be a float (assumes no
                    uncertainty in the observations) or a list of two floats
                    holding the observation and its variance. Required for history
                    matching, but can be provided when calling ``get_implausibility``.
                    Default is ``None``.
        :type obs: float, list, or None
        :param coords: (Optional) Inputs at which the emulator values will be computed
                       to compare the GP output to the observations. If provided, must
                       be a numpy array matching the input for the GP (must be a 2D
                       array with dimensions ``(n, D)`` where ``n`` is the number of
                       points to be considered and ``D`` is the number of parameters
                       for the GP). Only required if ``expectations`` is not provided.
                       Default is ``None``.
        :type coords: ndarray
        :param expectations: (Optional) Tuple of 3 numpy arrays or 2 numpy arrays and
                             ``None`` of the form expected from GP predictions. The first
                             array must hold the predicted mean at all query points, and
                             the second array must hold the variances from the GP predictions
                             at all query points. The third is not used, so can simply be a
                             dummy array or ``None``. Can instead provide a GP object and the
                             points to query to have the predictions made internally. Default
                             is ``None``.
        :type expectations: tuple holding 3 ndarrays
        """

        # Place-holder values for user-defined quantities that can be passed in
        # as args.
        self.gp = None
        self.obs = None
        self.coords = None
        self.expectations = None

        # Place-holder or default values for optional or derived quantities.
        self.ndim = None
        self.ncoords = None
        self.threshold = None
        self.I = None
        self.NROY = None
        self.RO = None

        # If suitable keyword arguments are provided, update the relevent
        # variable(s)
        if self.check_gp(gp):
            self.set_gp(gp)
        if self.check_obs(obs):
            self.set_obs(obs)
        if self.check_coords(coords):
            self.set_coords(coords)
        if self.check_expectations(expectations):
            self.set_expectations(expectations)
        if self.check_threshold(threshold):
            self.set_threshold(threshold)

        self.update()

    def get_n_obs(self):
        """
        Returns the number of observations
        
        Determines the number of observations. This is the length
        of the first argument of the ``obs`` attribute (which will
        be a numpy array)
        """
        return len(self.obs[0])

    def _select_expectations(self):
        r"""
        Determine the emulator predictions to use in HistoryMatching

        This method queries the internal state of the HistoryMatching object to get
        the expectations to use in the history matching calculation. It either
        uses the expectations that were directly provided, or makes predictions
        at the query points using the provided GP. If both methods are set
        simultaneously, the method raises an exception.

        :returns: Expectations to use in history matching. These are either provided
                  explicitly, or made internally using the GP and query points.
        :rtype: tuple of 3 ndarrays or 2 ndarrays and None
        """
        # Check that we have exactly 1 valid combination of parameters
        UseCoordGP = False
        UseExpectations = False
        if (self.check_coords(self.coords) and self.check_gp(self.gp)):
            UseCoordGP = True
        if self.check_expectations(self.expectations):
            UseExpectations = True
        if UseCoordGP and UseExpectations:
            raise ValueError("Multiple valid parameter combinations are set. Previously set " +
                             "parameters can be removed by setting them to None")
        if not UseCoordGP and not UseExpectations:
            raise ValueError("Expectations are not provided, nor is a GP and coordinates." +
                             "must set one in order to perform History Matching")

        # Confirm that the ncoords parameter is set
        if self.ncoords is None:
            raise ValueError("ncoords is not set despite a valid parameter combination "+
                             "being found.")

        # From this point on, the calculation works off the self.expectations value.
        # If this is not set, calculate it from the other parameters
        if UseCoordGP:
            expectations = self.gp.predict(self.coords)
        else:
            expectations = self.expectations

        return expectations

    def get_implausibility(self, discrepancy=0., rank=1):
        r"""
        Compute Implausibility measure for all query points

        Carries out the implausibility calculation given by:

        .. math::
            {I_i(\bar{x_0}) = \frac{|z_i - E(f_i(\bar{x_0}))|}
            {\sqrt{Var\left[z_i - E(f_i(\bar{x_0}))\right]}}}

        to return an implausibility value (a number of standard deviations between the
        emulator mean and the observation)for each of the provided coordinates.

        Requires that the observation parameter is set, and that at least one of the
        following conditions are met:

          a) The coords and gp parameters are set
          b) The expectations parameter is set

        Note that using the GP internally assumes that the standard prediction settings
        will be used when making predictions. If the user wishes to have more control
        over the prediction method (i.e. make the predictions from MCMC samples),
        the user should explicitly pass ``expectations`` to the object.

        An additional variance can be included that represents a general model
        discrepancy that describes the prior beliefs regarding how well the model
        matches reality. In practice, the model discrepancy is essential to have
        predictions that are not overly confident, however it can be hard to estimate
        (see further discussion in Brynjarsdottir and O\'Hagan, 2014).
        
        If dealing with multiple outputs, the ``rank`` argument allows the user
        to specify how to score different predictions by giving an ordering.
        If the value is ``0``, then the maximum implausibility score will be
        used. If the value is ``1``, then the second largest implausibility
        is used (i.e. the scoring ignores the largest component). By default,
        uses the second largest score, which is in general more forgiving for
        a badly performing emulator that is overconfident in its predictions.
        Must be a non-negative integer.

        As the implausibility calculation linearly sums variances, the result is
        agnostic to the precise provenance of any of the included variances

        :param discrepancy: Additional variance to be included in the implausibility
                            calculation. Must be a non-negative array or float.
                            Optional, default is ``0.``
        :type discrepancy: float or ndarray
        :param rank: Scoring method for multiple outputs. Must be a non-negative
                    integer less than the number of observations, which denotes
                    the location in the rank ordering of implausibility values
                    where the score is evaluated (i.e. the default value of ``1``
                    indicates that the second largest implausibility will be used).
        :type rank: int
        :returns: Array holding implausibility metric for all query points accounting
                  for all variances, ndarray of length ``(ncoords,)``
        :rtype: ndarray
        """

        # Confirm that observation parameter is set
        if not self.check_obs(self.obs):
            raise ValueError("implausibility calculation requires that the observation value is " +
                             "set. This can be done using the set_obs method.")

        assert np.all(discrepancy >= 0.), "Model discrepancy variance cannot be negative"
        discrepancy = np.atleast_1d(discrepancy)

        expectations = self._select_expectations()

        n_obs = self.get_n_obs()
        assert n_obs == np.atleast_2d(expectations[0]).shape[0]
        assert n_obs == np.atleast_2d(expectations[1]).shape[0]
        
        if n_obs == 1:
            rank = 0
        
        assert rank >= 0, "rank must be a non-negative integer"
        assert rank < n_obs, "rank must be less than the number of observations"

        # Compute implausibility for each expectation value
        Vs = np.zeros((n_obs, self.ncoords))
        Vs += expectations[1]                           # prediction variance
        Vs += discrepancy[:, np.newaxis]                # model discrepancy
        Vs += self.obs[1][:, np.newaxis]                # variance on observation
        I = (np.abs(self.obs[0][:, np.newaxis] - expectations[0]) / np.sqrt(Vs))
        # implausibility is the -rank index in the sorted array (i.e. if rank is 0,
        # return the largest, if rank is 1, return the second largest, etc.),
        # which can be correctly found in O(n) using partition.
        # partition puts the kth largest element in its correct position along
        # the selected axis, so we just need to select the kth row of the
        # partitioned array along axis 0
        self.I = np.partition(I, n_obs - rank - 1, axis=0)[n_obs - rank - 1]
        
        return self.I


    def get_NROY(self, discrepancy=0., rank=1):
        r"""
        Return set of indices that are not yet ruled out

        Returns a list of indices for ``self.I`` that correspond to entries that are not
        yet ruled out. Points that are ruled out have an implausibility metric that
        exceeds the threshold (can be set when initializing history matching or using
        the ``set_threshold`` method). If the implausibility metric has not yet been
        computed for the desired points, it is calculated. If a model discrepancy
        is to be included, it can be passed here.

        :param discrepancy: Additional variance to be included in the implausibility
                            calculation. Must be a non-negative float. Optional, default
                            is ``0.``
        :type discrepancy: float
        :returns: List of integer indices that have not yet been ruled out.
        :rtype: list
        """
        if self.I is None:
            self.get_implausibility(discrepancy, rank)

        self.NROY = list(np.where(self.I <= self.threshold)[0])

        return self.NROY


    def get_RO(self, discrepancy=0., rank=1):
        r"""
        Return set of indices that have been ruled out

        Returns a list of indices for ``self.I`` that correspond to entries that have
        been ruled out. Points that are ruled out have an implausibility metric that
        exceeds the threshold (can be set when initializing history matching or using
        the ``set_threshold`` method). If the implausibility metric has not yet been
        computed for the desired points, it is calculated. If a model discrepancy
        is to be included, it can be passed here.

        :param discrepancy: Additional variance to be included in the implausibility
                            calculation. Must be a non-negative float. Optional, default
                            is ``0.``
        :type discrepancy: float
        :returns: List of integer indices that have been ruled out.
        :rtype: list
        """
        if self.I is None:
            self.get_implausibility(discrepancy, rank)

        self.RO = list(np.where(self.I > self.threshold)[0])

        return self.RO


    def set_gp(self, gp):
        r"""
        Set the Gaussian Process to use with history matching

        Sets the ``self.gp`` variable to the provided ``GaussianProcess`` argument.

        :param gp: ``GaussianProcess`` object to use for history matching.
        :type gp: GaussianProcess
        :returns: None
        """
        if not self.check_gp(gp):
            raise TypeError("bad input for set_gp - expects a GaussianProcess object.")
        self.gp = gp


    def set_obs(self, obs):
        r"""
        Set the observations to be used for history matching

        Sets the ``self.obs`` variable to the provided ``obs`` argument. The object
        must pass the ``check_obs`` requirements, meaning that it must be a
        float (assumes that the observation has no error associated with it) or a
        list containing two floats (representing an observation and its variance).
        Note that the variance must be non-negative.

        :param obs: Observations to be used for history matching. Must be a float
                    or a list of two floats.
        :type obs: float or list
        :returns: None
        """
        if not self.check_obs(obs):
            raise TypeError("bad input for set_obs")
        if isinstance(obs, float):
            self.obs = [np.array([obs]), np.array([0.])]
        elif isinstance(list(obs), list):
            if len(obs) == 1:
                self.obs = [np.atleast_1d(obs[0]), np.array([0.])]
            else:
                self.obs = [np.atleast_1d(a) for a in obs]
        else:
            self.obs = np.array(obs)


    def set_coords(self, coords):
        r"""
        Set the query points to be used in history matching

        Sets the ``self.coords`` variable to the provided query points ``coords`` if
        it passes the ``check_coords`` requirements, or be ``None`` to remove a set
        of existing query points. ``coords`` must be a numpy array matching the
        inputs to the provided ``GaussianProcess`` object.

        :param coords: Numpy array holding query points (array with shape ``(n, D)``,
                       where ``n`` is the number of query points and ``D`` is the
                       number of inputs to the emulator), or ``None``
        :type coords: ndarray or None
        :returns: None
        """
        # need to allow coords == None, as otherwise can't reset values
        if not self.check_coords(coords) and (not coords is None):
            raise TypeError("bad input for set_coords - expected coords in the form " +
                            "of a list or 1D or 2D ndarray of numerical values")
        if isinstance(coords, np.ndarray):
            if len(coords.shape) == 1:
                self.coords = np.reshape(coords, [-1, 1])
            elif len(coords.shape) == 2:
                self.coords = coords
            else:
                raise Exception("error in exception handling - an ndarray of >2 " +
                                "dimensions somehow got through check_coords")
        elif isinstance(coords, list):
            self.coords = np.reshape(np.asarray(coords), [-1, 1])
        elif coords is None:
            self.coords = None
        else:
            raise TypeError("error in exception handling - an argument of illegal " +
                            "type somehow got through check_coords")
        self.update()


    def set_expectations(self, expectations):
        r"""
        Set the expected output of the simulator to be used in history matching

        Sets the ``self.expectations`` variable to the provided ``expectations``
        argument if it passes the ``check_expectations`` requirements, or ``None``
        can be used to remove an existing set of expectations. Expectations
        must be a tuple of 3 numpy arrays, where the first holds the predicted
        means for all query points and the second holds the predicted variances
        for all query points. The third arrray is not used in the computation,
        but is included as it is an expected output of the GP ``predict`` method.

        :param expectations: GP predictions at all query points. Must be a tuple
                             of 3 numpy arrays, or None to remove existing
                             expectations.
        :type expectations: tuple of 3 ndarrays or None
        :returns: None
        """
        # need to allow expectations == None, as otherwise can't reset values
        if not self.check_expectations(expectations) and (not expectations is None):
            raise TypeError("bad input for set_expectations - expected a Tuple " +
                            "of 3 ndarrays.")
        self.expectations = expectations
        self.update()


    def set_threshold(self, threshold):
        r"""
        Set the threshold value for history matching

        Sets the ``self.threshold variable`` to the provided threshold argument
        if it  passes the ``check_threshold`` requirements. The threshold must
        be a non-negative float.

        :param threshold: New value for threshold, must be a non-negative float.
        :type threshold: float
        :returns: None
        """
        if not self.check_threshold(threshold):
            raise TypeError("bad input for set_expectations - expected a float")
        self.threshold = float(threshold)


    def status(self):
        r"""
        Prints a summary of the current status of the class object.

        Prints a summary of the current status of the class object, including the
        values stored in ``self.gp``, ``.obs``, ``.ndim``, ``.ncoords``, and
        ``.threshold``, and the shapes/sizes ofvalues stored in ``self.obs``,
        ``.coords``, ``.expectations``, ``.I``, ``.NROY``, and ``.RO.``
        No inputs, no return value.

        :returns: None
        """

        print(str(self))


    def check_gp(self, gp):
        r"""
        Checks if the provided argument is consistent with expectations for a GP.

        Returns a boolean that is True if the provided quantity is consistent with
        the requirements for a gaussian process, i.e. is of type ``GaussianProcessBase``.

        :param gp: Input GP or MOGP object to be checked.
        :type gp: GaussianProcessBase or MultiOutputGPBase
        :returns: Boolean indicating if provided object is a ``GaussianProcess``
        :rtype: bool
        """
        return isinstance(gp, (GaussianProcessBase, MultiOutputGPBase))


    def check_obs(self, obs):
        r"""
        Checks if the provided argument is consistent with expectations for
        observations.

        Returns a boolean that is ``True`` if the provided quantity is consistent with
        the requirements for the observation quantity. Possible options include
        a numpy array with the first dimension having length 2, an iterable
        of length 1 or 2 containing floats or arrays (if 2 arrays, they must
        have the same length), or a float for a single observation with no error.

        :param obs: Input for observations to be checked
        :type obs: float, iterable, or ndarray
        :returns: Boolean indicating if provided observations are acceptable
        :rtype: bool
        """

        if obs is None:
            return False
        if isinstance(obs, list) or isinstance(obs, tuple) or isinstance(obs, np.ndarray):
            if isinstance(obs, np.ndarray):
                if obs.ndim > 2:
                    raise ValueError("bad input for HistoryMatching, the obs parameter" +
                                     "cannot be an array of more than 2D")
                assert obs.shape[0] == 2, "first dimension of observations must have length 2"
            else:
                if len(obs) > 2:
                    raise ValueError("bad input type for HistoryMatching - the specified " +
                                     "observation parameter cannot contain more than 2 entries "+
                                     "(value, [variance])")
                    if len(obs) <= 2:
                        try:
                            test = [np.array(a) for a in obs]
                        except TypeError:
                            raise TypeError("bad input type for HistoryMatching - the specified " +
                                            "observation parameter must contain only numerical " +
                                            "values")
                        if not (len(test[0]) == len(test[1]) or len(test[1]) == 1):
                            raise ValueError("Bad input for values to history matching -- " +
                                             "iterable arguments must be of the format " +
                                             "(float, float), (float, array), or (array, array)")
            if len(obs) == 2:
                assert np.all(obs[1] >= 0.), "variance in observations cannot be negative"
            return True
        else:
            try:
                test = float(obs)
            except (TypeError, ValueError):
                raise TypeError("bad input type for HistoryMatching - the specified " +
                                "observation parameter must contain only numerical values")
            return True
        return False


    def check_coords(self, coords):
        r"""
        Checks if the provided argument is consistent with expectations for
        coordinates.

        Returns a boolean that is ``True`` if the provided quantity is consistent with
        the requirements for the coordinates quantity, i.e. a ndarray of
        fewer than 3 dimensions.

        :param coords: Input to check for consistency with coordinates.
        :type coords: ndarray
        :returns: Boolean indicating if coords is consistent
        :rtype: bool
        """

        if coords is None:
            return False

        if isinstance(coords, np.ndarray):
            if len(coords.shape) <= 2:
                return True
        return False


    def check_expectations(self, expectations):
        r"""
        Checks if the provided argument is consistent with expectations for
        Gaussian Process Expectations.

        Returns a boolean that is ``True`` if the provided quantity is consistent with
        the output of the predict method of a GaussianProcess object, i.e. that it
        is a ``GaussianProcess.PredictResult`` object with mean and variance defined.

        :param expectations: Input to check for consistency with expectations
        :type expectations: tuple of 3 numpy arrays or 2 numpy arrays and None
        :returns: Boolean indicating if expectations is consistent
        :rtype: bool
        """
        if expectations is None:
            return False
        if not isinstance(expectations, PredictResult):
            return False
        if not all((isinstance(expectations[0], np.ndarray),
                    isinstance(expectations[1], np.ndarray),
                    (isinstance(expectations[2], np.ndarray) or
                     expectations[2] is None))):
            raise TypeError("bad input type for HistoryMatching - expected expectation " +
                            "values in the form of a PredictResults object with mean" +
                            "and uncertainty set.")
        if not expectations[0].shape == expectations[1].shape:
            raise ValueError("bad input for HistoryMatching - mean and variance " +
                             "expectations do not match")
        assert np.all(expectations[1] >= 0.), "all variances must be non-negative"
        return True


    def check_threshold(self, threshold):
        r"""
        Check value of threshold

        Checks if the provided argument is consistent with expectations for a
        threshold value.

        Returns a boolean that is ``True`` if the provided quantity is consistent with
        the requirements for a threshold value, i.e. that it is a non-negative numerical
        value.

        :param threshold: threshold to be tested
        :type threshold: float
        :returns:  Boolean indicating if the provided argument is consistent with
                   a threshold
        :rtype: bool
        """
        if threshold is None:
            return False
        try:
            test = float(threshold)
            assert test >= 0., "threshold must be non-negative"
            return True
        except TypeError:
            return False

    def update(self):
        r"""
        Update History Matching object

        Checks that sufficient information exists to compute ``ndim`` and/or ``ncoords``
        and, if so, computes these values. This also serves to update these values
        if the information on which they are based is changed. No inputs, no return
        value.

        :returns: None
        """

        if self.check_coords(self.coords):
            self.ndim = self.coords.shape[1]
            self.ncoords = self.coords.shape[0]
        elif self.check_expectations(self.expectations):
            self.ncoords = self.expectations[0].shape[0]


    def __str__(self):
        r"""
        Returns string representation of ``HistoryMatching`` object

        Returns a string representation of a ``HistoryMatching`` object,
        used when printing information to the interpreter or when the
        ``status`` method is called.

        :returns: string representation of the HistoryMatching object.
        :rtype: str
        """

        if self.coords is None:
            coord_str = None
        else:
            coord_str = self.coords.shape

        if self.expectations is None:
            exp_str = None
        else:
            exp_str = (str(type(self.expectations)) + " of len " +
                       str(len(self.expectations)) + " containing arrays of shape: " +
                       str(self.expectations[0].shape))

        if self.I is None:
            I_str = None
        elif isinstance(self.I, np.ndarray):
            I_str = (str(type(self.I)) + " of shape " + str(self.I.shape))

        if self.NROY is None:
            NROY_str = None
        elif isinstance(self.NROY, list):
            NROY_str = (str(type(self.NROY)) + " of length " + len(self.NROY))

        if self.RO is None:
            RO_str = None
        elif isinstance(self.RO, list):
            RO_str = (str(type(self.RO)) + " of length " + len(self.RO))

        return ("History Matching tools created with:\n" +
                "Gaussian Process: {}\n" +
                "Observations: {}\n" +
                "Coords: {}\n" +
                "Expectations: {}\n" +
                "No. of Input Dimensions: {}\n" +
                "No. of Descrete Expectation Values: {}\n" +
                "I_threshold: {}\n" +
                "I: {}\n" +
                "NROY: {}\n" +
                "RO: {}\n").format(self.gp, self.obs, coord_str, exp_str,
                                   self.ndim, self.ncoords, self.threshold,
                                   I_str, NROY_str, RO_str)
