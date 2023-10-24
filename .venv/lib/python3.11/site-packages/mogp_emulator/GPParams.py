import numpy as np
        
class CorrTransform(object):
    r"""
    Class representing correlation length transforms
    
    ``mogp_emulator`` performs coordinate transforms on all
    correlation length parameters to stabilize the fitting
    routines. The scaled correlation length :math:`l` is related
    to the raw parameter :math:`\theta` as follows: ::
    
    :math:`{l = \exp(-0.5\theta)}`
    
    This class groups together the coordinate transforms used for
    correlation length parameters as a collection of static
    methods. One does not need to create an object to use
    it, but this conveniently groups together all methods
    (in the event that multiple transforms are needed to perform
    a calculation). Collected methods are:
    
    * ``transform`` (convert raw parameter to scaled)
    * ``inv_transform`` (convert scaled parameter to raw)
    * ``dscaled_draw`` (compute derivative of scaled with respect to
      raw, as a function of the scaled parameter)
    * ``d2scaled_draw2`` (compute second derivative of scaled
      with respect to raw, as a function of the scaled parameter)
    
    The derivative functions take the scaled parameter as input
    due to the internal details of the required calculations. If
    you wish to compute the derivative using the raw parameter
    as the input, apply the provided ``transform`` method to
    the parameter first.
    """
    @staticmethod
    def transform(r):
        r"""
        Convert raw parameter to scaled
        
        :param r: Input raw parameter
        :type r: float
        :returns: scaled parameter
        :rtype: float
        """
        return np.exp(-0.5*r)
    @staticmethod
    def inv_transform(s):
        r"""
        Convert scaled parameter to raw
        
        :param s: Input scaled parameter
        :type s: float
        :returns: raw parameter
        :rtype: float
        """
        return -2.*np.log(s)
    @staticmethod
    def dscaled_draw(s):
        r"""
        Compute derivative of the scaled parameter with respect to the raw
        (as a function of the scaled parameter).
        
        :param s: Input scaled parameter
        :type s: float
        :returns: transform derivative of scaled with respect to raw
        :rtype: float
        """
        return -0.5*s
    @staticmethod
    def d2scaled_draw2(s):
        r"""
        Compute second derivative of the scaled parameter with respect
        to the raw (as a function of the scaled parameter).
        
        :param s: Input scaled parameter
        :type s: float
        :returns: transform second derivative of scaled with respect
                  to raw
        :rtype: float
        """
        return 0.25*s
    
class CovTransform(object):
    r"""
    Class representing covariance and nugget transforms
    
    ``mogp_emulator`` performs coordinate transforms on all
    correlation length parameters to stabilize the fitting
    routines. The scaled covariance :math:`\sigma^2` or
    scaled nugget :math:`\eta` is related to the scaled
    parameter :math:`\theta` as follows: ::
    
    :math:`{\sigma^2 = \exp(\theta)}` (for covariance), or
    :math:`{\eta = \exp(\theta)}` (for nugget)
    
    This class groups together the coordinate transforms used for
    correlation length parameters as a collection of static
    methods. One does not need to create an object to use
    it, but this conveniently groups together all methods
    (in the event that multiple transforms are needed to perform
    a calculation). Collected methods are:
    
    * ``transform`` (convert raw parameter to scaled)
    * ``inv_transform`` (convert scaled parameter to raw)
    * ``dscaled_draw`` (compute derivative of scaled with respect to
      raw, as a function of the scaled parameter)
    * ``d2scaled_draw2`` (compute second derivative of scaled
      with respect to raw, as a function of the scaled parameter)
    
    The derivative functions take the scaled parameter as input
    due to the internal details of the required calculations. If
    you wish to compute the derivative using the raw parameter
    as the input, apply the provided ``transform`` method to
    the parameter first.
    """
    @staticmethod
    def transform(r):
        r"""
        Convert raw parameter to scaled
        
        :param r: Input raw parameter
        :type r: float
        :returns: scaled parameter
        :rtype: float
        """
        return np.exp(r)
    @staticmethod
    def inv_transform(s):
        r"""
        Convert scaled parameter to raw
        
        :param s: Input scaled parameter
        :type s: float
        :returns: raw parameter
        :rtype: float
        """
        return np.log(s)
    @staticmethod
    def dscaled_draw(s):
        r"""
        Compute derivative of the scaled parameter with respect to the raw
        (as a function of the scaled parameter).
        
        :param s: Input scaled parameter
        :type s: float
        :returns: transform derivative of scaled with respect to raw
        :rtype: float
        """
        return s
    @staticmethod
    def d2scaled_draw2(s):
        r"""
        Compute second derivative of the scaled parameter with respect
        to the raw (as a function of the scaled parameter).
        
        :param s: Input scaled parameter
        :type s: float
        :returns: transform second derivative of scaled with respect
                  to raw
        :rtype: float
        """
        return s

def _process_nugget(nugget):
    """
    Take raw nugget input and convert to a value and a string
    
    :param nugget: Input nugget. Can be a float (fixed nugget) or a string
                   (nugget is inferred via fitting)
    :type nugget: float or str
    :returns: Tuple containing nugget value (float) and nugget type (string)
    :rtype: tuple containing float and str
    """
    
    if not isinstance(nugget, (str, float)):
        try:
            nugget = float(nugget)
        except TypeError:
            raise TypeError("nugget parameter must be a string or a non-negative float")

    if isinstance(nugget, str):
        if nugget == "adaptive":
            nugget_type = "adaptive"
        elif nugget == "fit":
            nugget_type = "fit"
        elif nugget == "pivot":
            nugget_type = "pivot"
        else:
            raise ValueError("bad value of nugget, must be a float or 'adaptive', 'pivot', or 'fit'")
        nugget_value = None
    else:
        if nugget < 0.:
            raise ValueError("nugget parameter must be non-negative")
        nugget_value = float(nugget)
        nugget_type = "fixed"
        
    return nugget_value, nugget_type

def _length_1_array_to_float(arr):
    """
    Safely convert a float or a length one array to a float
    
    Takes any shape argument (float, array of some length with
    a single entry) and converts to a float
    
    :param arr: Input float or array with a single entry but
                unknown shape
    :type arr: float or ndarray
    :returns: array entry converted to float
    :rtype: float
    """
    arr = np.reshape(np.array(arr), (-1,))
    assert arr.shape == (1,)
    return arr[0]

class GPParams(object):
    r"""
    Class representing parameters for a GaussianProcess object
    
    This class serves as a wrapper to a numpy array holding the parameters for a
    ``GaussianProcess`` object. Because the parameters for a GP are transformed
    in different ways (depending on convention and positivity constraints), the
    raw parameter values are often fairly opaque. This class provides more clarity
    on the raw and transformed parameter values used in a particular GP.
    
    The class is a wrapper around the numpy array holding the raw data values.
    When initializing a new ``GPParams`` object, the data can optionally be
    specified to initialize the array. Otherwise, a default value of ``None``
    will be used to indicate that the GP has not been fit.
    
    :param n_mean: The number of parameters in the mean function. Optional, default
                   is 0 (zero or fixed mean function). Must be a non-negative
                   integer.
    :type n_mean: int
    :param n_corr: The number of correlation length parameters. Optional, default
                   is 1. This must be the same as the number of inputs for a
                   particular GP. Must be a positive integer.
    :type n_corr: int
    :param nugget: String or float specifying how nugget is fit. If a float, a
                   fixed nugget is used (and will fix the value held in the
                   ``GPParams`` object). If a string, can be ``'fit'``,
                   ``'adaptive'``, or ``'pivot'``.
    :type nugget: str or float
    
    The transformations between the raw parameters :math:`\theta` and the
    transformed ones :math:`(\beta, l, \sigma^2, \eta^2)` are as follows:
    
    1. **Mean:** No transformation; :math:`{\beta = \theta}`
    2. **Correlation:** The raw values are transformed via
       :math:`{l = \exp(-0.5\theta)}` such that the transformed values are the
       correlation length associated with the given input.
    3. **Covariance:** The raw value is transformed via
       :math:`{\sigma^2 = \exp(\theta)}` so that the transformed value is
       the covariance.
    4. **Nugget:** The raw value is transformed via
       :math:`{\eta^2 = \exp(\theta)}` so that the transformed value is
       the variance associated with the nugget noise.
    """
    
    def __init__(self, n_mean=0, n_corr=1, nugget="fit"):
        r"""
        Create a new empy parameters object. Must specify a number of
        mean parameters, correlation parameters, and nugget type.
        
        :param n_mean: Number of mean parameters, must be a non-negative
                       integer. Optional, default is ``0``.
        :type n_mean: int
        :param n_corr: Number of correlation lengths. Must be a positive
                       integer. Optional, default is ``1``.
        :type n_corr: int
        :param nugget: Method for handling nugget. Must be a string,
                       ``"fit"``, ``"fixed"``, ``"pivot"``, or
                       ``"adaptive"``. Optional, default is ``"fit"``.
        :type nugget: str
        :returns: New instance of ``GPParams``
        """
        assert n_mean >= 0, "Number of mean parameters must be nonnegative"
        self.n_mean = n_mean
        assert n_corr >= 1, "Number of correlation parameters must be positive"
        self.n_corr = n_corr
        
        self._nugget, self._nugget_type = _process_nugget(nugget)
        
        if self.n_mean == 0:
            self._mean = np.array([])
        else:
            self._mean = None
        
        self._cov = None
        
        self._data = None
        
    @property
    def n_params(self):
        r"""
        Number of fitting parameters stored in data array
        
        This is the number of correlation lengths plus one
        (for the covariance) and optionally an additional
        parameter if the nugget is fit.
        """
        return self.n_corr + 1 + int(self.nugget_type == "fit")
        
    @property
    def mean(self):
        r"""
        Mean parameters
        
        The ``mean`` property returns the part of the data array associated with
        the mean function. Returns a numpy array of length ``(n_mean,)``
        or ``None`` if the mean data has not been initialized.
        
        Can be set with a new numpy array of the correct length.
        
        :returns: Numpy array holding the mean parameters
        :rtype: ndarray
        """
        return self._mean
        
    @mean.setter
    def mean(self, new_mean):
        if new_mean is None:
            if self.n_mean > 0:
                self._mean = None
        else: 
            new_mean = np.reshape(np.array(new_mean), (-1,))
            assert new_mean.shape == (self.n_mean,), "Bad shape for new mean parameters"
            self._mean = np.copy(new_mean)

    @property
    def corr_raw(self):
        r"""Raw Correlation Length Parameters
        
        This is used in computing kernels as the kernels perform
        the parameter transformations internally.
        """
        if self._data is None:
            return None
        else:
            return self._data[:self.n_corr]

    @property
    def corr(self):
        r"""
        Transformed correlation length parameters
        
        The ``corr`` property returns the part of the data array, with
        transformation, associated with the correlation lengths.
        Transformation is done via via :math:`{l = \exp(-0.5\theta)}`.
        Returns a numpy array of length ``(n_corr,)`` or ``None`` if the
        data array has not been initialized.
        
        Can be set with a new numpy array of the correct length. New
        parameters must satisfy the positivity constraint and 
        all be :math:`> 0`. If the data array has not been initialized,
        then setting individual parameter values cannot be done.
        
        :returns: Numpy array holding the transformed correlation parameters
        :rtype: ndarray
        """
        if self._data is None:
            return None
        else:
            return CorrTransform.transform(self.corr_raw)
        
    @corr.setter
    def corr(self, new_corr):
        if new_corr is None:
            raise ValueError("Resetting correlation lengths requires resetting the full data array")
        if self._data is None:
            raise ValueError("Must set full data array before modifying individual parameters")
        new_corr = np.reshape(np.array(new_corr), (-1,))
        assert np.all(new_corr > 0.), "Correlation parameters must all be positive"
        assert new_corr.shape == (self.n_corr,)
        self._data[:self.n_corr] = CorrTransform.inv_transform(new_corr)

    @property
    def cov_index(self):
        "Determine the location in the data array of the covariance parameter"
        
        if self.nugget_type == "fit":
            return -2
        else:
            return -1

    @property
    def cov(self):
        r"""
        Transformed covariance parameter
        
        The ``cov`` property returns the covariance transformed according to
        :math:`\sigma^2=\exp(\theta)`. Returns a float or ``None`` if the
        data array has not been initialized.
        
        Can be set with a new float :math:`>0` or numpy array of length 1
        holding a positive float. If the data array has not been initialized,
        then setting individual parameter values cannot be done.
        
        :returns: Transformed covariance parameter
        :rtype: float or None
        """
        if self._data is None:
            return None
        else:
            return CovTransform.transform(self._data[self.cov_index])
        
    @cov.setter
    def cov(self, new_cov):
        if self._data is None:
            raise ValueError("Must set full data array before modifying individual parameters")
        else:
            new_cov = _length_1_array_to_float(new_cov)
            assert new_cov > 0., "Covariance must be positive"
            self._data[self.cov_index] = CovTransform.inv_transform(new_cov)

    @property
    def nugget_type(self):
        """
        Method used to fit nugget
        
        :returns: string indicating nugget fitting method, either
                  ``"fixed"``, ``"adaptive"``, ``"pivot"``, or
                  ``"fit"``
        :rtype: str
        """
        return self._nugget_type

    @property
    def nugget(self):
        r"""
        Transformed nugget parameter
        
        The ``nugget`` property returns the nugget transformed via
        :math:`\eta^2=\exp(\theta)`. Returns a float or ``None`` if the
        emulator parameters have not been initialized or if the emulator
        does not have a nugget.
        
        Can be set with a new float :math:`>0` or numpy array of length 1.
        If the data array has not been initialized or the emulator has no
        nugget, then the nugget setter will return an error.
        
        :returns: Transformed nugget parameter
        :rtype: float or None
        """
        if self.nugget_type in ["fixed", "adaptive", "pivot"]:
            return self._nugget
        elif self.nugget_type == "fit":
            if self._data is None:
                return None
            else:
                return CovTransform.transform(self._data[-1])
        
    @nugget.setter
    def nugget(self, new_nugget):
        if self.nugget_type == "pivot":
            if not new_nugget is None:
                raise ValueError("Cannot explicitly modify nugget for 'pivot' nugget type")
        elif self.nugget_type == "fixed":
            if not np.allclose(self._nugget, new_nugget):
                raise ValueError("Cannot explicitly modify nugget for 'fixed' nugget type")
        elif self.nugget_type == "adaptive":
            if new_nugget is None:
                self._nugget = None
            else:
                new_nugget = _length_1_array_to_float(new_nugget)
                assert new_nugget >= 0., "nugget cannot be negative"
                self._nugget = new_nugget
        else:
            if new_nugget is None:
                raise ValueError("Cannot reset fit nugget individually, must reset full data array")
            if self._data is None:
                raise ValueError("Must initialize parameters before setting individual values")
            new_nugget = _length_1_array_to_float(new_nugget)
            assert new_nugget >= 0., "Nugget must be positive"
            self._data[-1] = CovTransform.inv_transform(new_nugget)

    def get_data(self):
        r"""
        Returns current value of raw parameters as a numpy array
        
        Provides access to the underlying data array for the ``GPParams``
        object. Returns a numpy array or ``None`` if the parameters
        of this object have not yet been set.
        
        :returns: Numpy array holding the raw parameter values
        :rtype: ndarray or None
        """
        return self._data

    def set_data(self, new_params):
        r"""
        Set a new value of the raw parameters
        
        Allows the data underlying the ``GPParams`` object to be set
        at once. Can also be used to reset the underlying data to
        ``None`` for the full array, indicating that no parameters
        have been set. Note that setting the data re-initializes the
        mean and (if necessary) covariance and nugget.
        
        :param new_params: New parameter values as a numpy array with shape
                           ``(n_params,)`` or None.
        :type new_params: ndarray or None
        :returns: None
        """
        if new_params is None:
            self._data = None
        else:
            new_params = np.array(new_params)
            assert self.same_shape(new_params), "Bad shape for new data; expected {} parameters".format(self.n_params)
            self._data = np.copy(new_params)
        self.mean = None
        if self.nugget_type == "adaptive":
            self._nugget = None
        
    def same_shape(self, other):
        """
        Test if two GPParams objects have the same shape
        
        Method to check if a new ``GPParams`` object or numpy array has the
        same length as the current object. If a numpy array, assumes
        that the array represents the fitting parameters and that the
        mean parameters will be handled separately. If a ``GPParams``
        object, it will also check if the number of mean parameters
        match.
        
        Returns a boolean if the number of mean (if a ``GPParams`` object
        only), correlation, and nugget parameters (for a numpy array or
        a ``GPParams`` object) are the same in both object.
        
        :param other: Additional instance of ``GPParams`` or ndarray to be
                      compared with the current one.
        :type other: GPParams or ndarray
        :returns: Boolean indicating if the underlying arrays have the same
                  shape.
        :rtype: bool
        """
        
        if isinstance(other, np.ndarray):
            return other.shape == (self.n_params,)
        elif isinstance(other, GPParams):
            return (self.n_mean == other.n_mean and
                    self.n_corr == other.n_corr and
                    self.nugget_type == other.nugget_type)
        else:
            raise ValueError("other must be a numpy array or another GPParams object in GPParams.same_shape")

    def __str__(self):
        "Returns a string representation of the GPParams class"
        outstr = "GPParams with:"
        if self._data is None:
            outstr += " data = None"
        else:
            outstr += "\nmean = {}".format(self.mean)
            outstr += "\ncorrelation = {}".format(self.corr)
            outstr += "\ncovariance = {}".format(self.cov)
            outstr += "\nnugget = {}".format(self.nugget)
        return outstr