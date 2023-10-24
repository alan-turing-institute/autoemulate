r"""
Kernel module, implements a few standard kernels for use with the
``GaussianProcess`` class. Options include the Squared Exponential
kernel and Matern 5/2 kernel, both with either a single correlation
length (``UniformSqExp``, ``UniformMat52``) or correlation lengths
for each input dimension (``SquaredExponential``, ``Matern52``).
The product form of the Matern 5/2 kernel (``ProductMat52``) is
also available.
"""

import numpy as np

class KernelBase(object):
    "Base Kernel"
    
    def get_n_params(self, inputs):
        """
        Determine number of correlation length parameters based on inputs
        
        Determines the number of parameters required for a given set of inputs.
        Returns the number of parameters as an integer.
        
        :param inputs: Set of inputs for which the number of correlation length
                       parameters is desired.
        :type inputs: ndarray
        :returns: Number of correlation length parameters
        :rtype: int
        """
        inputs = np.array(inputs)
        assert inputs.ndim == 2, "Inputs must be a 2D array"
        
        return inputs.shape[1]
    
    def _check_inputs(self, x1, x2, params):
        r"""
        Common function for checking dimensions of inputs (default version)

        This function checks the inputs to any kernel evaluation for consistency and ensures
        that all input arrays have the correct dimensionality. It returns the reformatted
        arrays, the number of inputs, and the number of hyperparameters. If the method
        determines that the array dimensions are not all consistent with one another,
        it will raise an ``AssertionError``. This method is called internally whenever
        the kernel is evaluated.

        :param x1: First parameter array. Should be a 1-D or 2-D array (1-D is acceptable
                   if either there is only a single point, or each point has only a single
                   parameter). If there is more than one parameter, the last dimension
                   must match the last dimension of ``x2`` and be one less than the length
                   of ``params``.
        :type x1: array-like
        :param x2: Second parameter array. The same restrictions apply that hold for ``x1``
                   described above.
        :type x2: array-like
        :param params: Hyperparameter array. Must have a length that is one more than the
                       last dimension of ``x1`` and ``x2``, hence minimum length is 2.
        :type params: array-like
        :returns: A tuple containing the following: reformatted ``x1``, ``n1``, reformatted
                  ``x2``, ``n2``, ``params``, and ``D``. ``x1`` will be an array with
                  dimensions ``(n1, D - 1)``, ``x2`` will be an array with dimensions
                  ``(n2, D - 1)``, and ``params`` will be an array with dimensions ``(D,)``.
                  ``n1``, ``n2``, and ``D`` will be integers.
        """

        params = np.array(params)
        assert params.ndim == 1, "parameters must be a vector"
        D = len(params)
        assert D >= 1, "minimum number of parameters in a covariance kernel is 1"

        x1 = np.array(x1)

        assert x1.ndim == 1 or x1.ndim == 2, "bad number of dimensions in input x1"

        if x1.ndim == 2:
            assert x1.shape[1] == D, "bad shape for x1"
        else:
            if D == 1:
                x1 = np.reshape(x1, (len(x1), 1))
            else:
                x1 = np.reshape(x1, (1, D))

        n1 = x1.shape[0]

        x2 = np.array(x2)

        assert x2.ndim == 1 or x2.ndim == 2, "bad number of dimensions in input x2"

        if x2.ndim == 2:
            assert x2.shape[1] == D, "bad shape for x2"
        else:
            if D == 1:
                x2 = np.reshape(x2, (len(x2), 1))
            else:
                x2 = np.reshape(x2, (1, D))

        n2 = x2.shape[0]

        return x1, n1, x2, n2, params, D

    def kernel_f(self, x1, x2, params):
        r"""
        Compute kernel values for a set of inputs

        Returns the value of the kernel for two sets of input points and a choice of
        hyperparameters. This function should not need to be modified for different choices
        of the kernel function or distance metric, as after checking the inputs it simply
        calls the routine to compute the distance metric and then evaluates the kernel function
        for those distances.

        :param x1: First input array. Must be a 1-D or 2-D array, with the length of
                   the last dimension matching the last dimension of ``x2`` and
                   one less than the length of ``params``. ``x1`` may be 1-D if either
                   each point consists of a single parameter (and ``params`` has length
                   2) or the array only contains a single point (in which case, the array
                   will be reshaped to ``(1, D - 1)``).
        :type x1: array-like
        :param x2: Second input array. The same restrictions that apply to ``x1`` also
                   apply here.
        :type x2: array-like
        :param params: Hyperparameter array. Must be 1-D with length one greater than
                       the last dimension of ``x1`` and ``x2``.
        :type params: array-like
        :returns: Array holding all kernel values between points in arrays ``x1``
                  and ``x2``. Will be an array with shape ``(n1, n2)``, where ``n1``
                  is the length of the first axis of ``x1`` and ``n2`` is the length
                  of the first axis of ``x2``.
        :rtype: ndarray
        """

        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)

        return self.calc_K(self.calc_r2(x1, x2, params))

    def kernel_deriv(self, x1, x2, params):
        r"""
        Compute kernel gradient for a set of inputs

        Returns the value of the kernel gradient for two sets of input points and a choice of
        hyperparameters. This function should not need to be modified for different choices
        of the kernel function or distance metric, as after checking the inputs it simply
        calls the routine to compute the distance metric, kernel function, and the appropriate
        derivative functions of the distance and kernel functions.

        :param x1: First input array. Must be a 1-D or 2-D array, with the length of
                   the last dimension matching the last dimension of ``x2`` and
                   one less than the length of ``params``. ``x1`` may be 1-D if either
                   each point consists of a single parameter (and ``params`` has length
                   2) or the array only contains a single point (in which case, the array
                   will be reshaped to ``(1, D - 1)``).
        :type x1: array-like
        :param x2: Second input array. The same restrictions that apply to ``x1`` also
                   apply here.
        :type x2: array-like
        :param params: Hyperparameter array. Must be 1-D with length one greater than
                       the last dimension of ``x1`` and ``x2``.
        :type params: array-like
        :returns: Array holding the gradient of the kernel function between points in arrays
                  ``x1`` and ``x2`` with respect to the hyperparameters. Will be an array with
                  shape ``(D, n1, n2)``, where ``D`` is the length of ``params``, ``n1`` is the
                  length of the first axis of ``x1`` and ``n2`` is the length of the first axis
                  of ``x2``. The first axis indicates the different derivative components
                  (i.e. the derivative with respect to the first parameter is [0,:,:], etc.)
        :rtype: ndarray
        """

        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)

        dKdr2 = self.calc_dKdr2(self.calc_r2(x1, x2, params))

        dr2dtheta = self.calc_dr2dtheta(x1, x2, params)

        dKdtheta = dKdr2*dr2dtheta

        return dKdtheta

    def kernel_hessian(self, x1, x2, params):
        r"""
        Calculate the Hessian of the kernel evaluated for all pairs of points with
        respect to the hyperparameters

        Returns the value of the kernel Hessian for two sets of input points and a choice of
        hyperparameters. This function should not need to be modified for different choices
        of the kernel function or distance metric, as after checking the inputs it simply
        calls the routine to compute the distance metric, kernel function, and the appropriate
        derivative functions of the distance and kernel functions.

        :param x1: First input array. Must be a 1-D or 2-D array, with the length of
                   the last dimension matching the last dimension of ``x2`` and
                   one less than the length of ``params``. ``x1`` may be 1-D if either
                   each point consists of a single parameter (and ``params`` has length
                   2) or the array only contains a single point (in which case, the array
                   will be reshaped to ``(1, D - 1)``).
        :type x1: array-like
        :param x2: Second input array. The same restrictions that apply to ``x1`` also
                   apply here.
        :type x2: array-like
        :param params: Hyperparameter array. Must be 1-D with length one greater than
                       the last dimension of ``x1`` and ``x2``.
        :type params: array-like
        :returns: Array holding the Hessian of the pair-wise distances between points in arrays
                  ``x1`` and ``x2`` with respect to the hyperparameters. Will be an array with
                  shape ``(D, D, n1, n2)``, where ``D`` is the length of ``params``, ``n1`` is
                  the length of the first axis of ``x1`` and ``n2`` is the length of the first
                  axis of ``x2``. The first two axes indicates the different derivative components
                  (i.e. the second derivative with respect to the first parameter is [0,0,:,:],
                  the mixed partial with respect to the first and second parameters is [0,1,:,:]
                  or [1,0,:,:], etc.)
        :rtype: ndarray
        """

        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)

        r2_matrix = self.calc_r2(x1, x2, params)
        dKdr2 = self.calc_dKdr2(r2_matrix)
        d2Kdr22 = self.calc_d2Kdr22(r2_matrix)

        dr2dtheta = self.calc_dr2dtheta(x1, x2, params)
        d2r2dtheta2 = self.calc_d2r2dtheta2(x1, x2, params)

        d2Kdtheta2 = (d2Kdr22 * dr2dtheta[:,np.newaxis,:,:] * dr2dtheta[np.newaxis,:,:,:] +
                      dKdr2 * d2r2dtheta2)

        return d2Kdtheta2

class UniformKernel(KernelBase):
    r"""
    Kernel with a single correlation length
    """
    
    def get_n_params(self, inputs):
        """
        Determine number of correlation length parameters based on inputs
        
        Determines the number of parameters required for a given set of inputs.
        Returns the number of parameters as an integer.
        
        :param inputs: Set of inputs for which the number of correlation length
                       parameters is desired.
        :type inputs: ndarray
        :returns: Number of correlation length parameters
        :rtype: int
        """
        return 1
    
    def _check_inputs(self, x1, x2, params):
        r"""
        Common function for checking dimensions of inputs

        This function checks the inputs to any kernel evaluation for consistency and ensures
        that all input arrays have the correct dimensionality. It returns the reformatted
        arrays, the number of inputs, and the number of hyperparameters. If the method
        determines that the array dimensions are not all consistent with one another,
        it will raise an ``AssertionError``. This method is called internally whenever
        the kernel is evaluated.

        :param x1: First parameter array. Should be a 1-D or 2-D array (1-D is acceptable
                   if each point has only a single input parameter).
        :type x1: array-like
        :param x2: Second parameter array. The same restrictions apply that hold for ``x1``
                   described above.
        :type x2: array-like
        :param params: Hyperparameter array. Must have a length that is one more than the
                       last dimension of ``x1`` and ``x2``, hence minimum length is 2.
        :type params: array-like
        :returns: A tuple containing the following: reformatted ``x1``, ``n1``, reformatted
                  ``x2``, ``n2``, ``params``, and ``D``.
                  ``n1``, ``n2``, and ``D`` will be integers.
        """

        params = np.array(params)
        assert params.ndim == 1, "parameters must be a vector"
        D = len(params)
        assert D == 1, "Uniform kernels only support a single correlation length"

        x1 = np.array(x1)

        assert x1.ndim == 1 or x1.ndim == 2, "bad number of dimensions in input x1"

        if not x1.ndim == 2:
            x1 = np.reshape(x1, (-1, 1))

        n1 = x1.shape[0]

        x2 = np.array(x2)

        assert x2.ndim == 1 or x2.ndim == 2, "bad number of dimensions in input x2"

        if not x2.ndim == 2:
            x2 = np.reshape(x2, (-1, 1))

        n2 = x2.shape[0]
        
        assert x1.shape[1] == x2.shape[1], "Input arrays do not have the same number of inputs"

        return x1, n1, x2, n2, params, D
        
    def calc_r2(self, x1, x2, params):
        r"""
        Calculate squared distance between all pairs of points

        This method computes the scaled Euclidean distance between all pairs of points
        in ``x1`` and ``x2``.
        For example, if
        ``x1 = [1.]``, ``x2`` = [2.], and ``params = [2.]`` then ``calc_r`` would
        return :math:`{\sqrt{exp(2)*(1 - 2)^2}=\sqrt{exp(2)}}`
        as an array with shape ``(1,1)``.

        :param x1: First input array. Must be a 1-D or 2-D array, with the length of
                   the last dimension matching the last dimension of ``x2`` and
                   one less than the length of ``params``. ``x1`` may be 1-D if either
                   each point consists of a single parameter (and ``params`` has length
                   2) or the array only contains a single point (in which case, the array
                   will be reshaped to ``(1, D - 1)``).
        :type x1: array-like
        :param x2: Second input array. The same restrictions that apply to ``x1`` also
                   apply here.
        :type x2: array-like
        :param params: Hyperparameter array. Must be 1-D with length one greater than
                       the last dimension of ``x1`` and ``x2``.
        :type params: array-like
        :returns: Array holding all pair-wise squared distances between points in arrays ``x1``
                  and ``x2``. Will be an array with shape ``(n1, n2)``, where ``n1``
                  is the length of the first axis of ``x1`` and ``n2`` is the length
                  of the first axis of ``x2``.
        :rtype: ndarray
        """

        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)
        
        exp_theta = np.exp(params)[0]

        r2_matrix = np.sum(exp_theta*(x1[:, np.newaxis, :] - x2[np.newaxis, :, :])**2, axis=-1)

        if np.any(np.isinf(r2_matrix)):
            raise FloatingPointError("Inf enountered in kernel distance computation")

        return r2_matrix
        
    def calc_dr2dtheta(self, x1, x2, params):
        r"""
        Calculate the first derivative of the distance between all pairs of points with
        respect to the hyperparameters

        This method computes the derivative of the scaled Euclidean distance between
        all pairs of points in ``x1`` and ``x2`` with respect to the hyperparameters.
        The gradient is held in an array with shape ``(D, n1, n2)``, where ``D`` is
        the length of ``params``, ``n1`` is the length of the first axis of ``x1``,
        and ``n2`` is the length of the first axis of ``x2``. This is used in the
        computation of the gradient and Hessian of the kernel. The first index
        represents the different derivatives with respect to each hyperparameter.

        :param x1: First input array. Must be a 1-D or 2-D array, with the length of
                   the last dimension matching the last dimension of ``x2`` and
                   one less than the length of ``params``. ``x1`` may be 1-D if either
                   each point consists of a single parameter (and ``params`` has length
                   2) or the array only contains a single point (in which case, the array
                   will be reshaped to ``(1, D - 1)``).
        :type x1: array-like
        :param x2: Second input array. The same restrictions that apply to ``x1`` also
                   apply here.
        :type x2: array-like
        :param params: Hyperparameter array. Must be 1-D with length one greater than
                       the last dimension of ``x1`` and ``x2``.
        :type params: array-like
        :returns: Array holding the derivative of the pair-wise distances between
                  points in arrays ``x1`` and ``x2`` with respect to the hyperparameters.
                  Will be an array with shape ``(D, n1, n2)``, where ``D`` is the length
                  of ``params``, ``n1`` is the length of the first axis of ``x1`` and
                  ``n2`` is the length of the first axis of ``x2``. The first axis
                  indicates the different derivative components (i.e. the derivative
                  with respect to the first parameter is [0,:,:], etc.)
        :rtype: ndarray
        """

        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)

        return np.reshape(self.calc_r2(x1, x2, params), (1, n1, n2))
        
    def calc_d2r2dtheta2(self, x1, x2, params):
        r"""
        Calculate all second derivatives of the distance between all pairs of points with
        respect to the hyperparameters

        This method computes all second derivatives of the scaled Euclidean distance
        between all pairs of points in ``x1`` and ``x2`` with respect to the
        hyperparameters. The gradient is held in an array with shape ``(D, D, n1, n2)``,
        where ``D`` is the length of ``params``, ``n1`` is the length of the first axis
        of ``x1``, and ``n2`` is the length of the first axis of ``x2``. This is used in
        the computation of the gradient and Hessian of the kernel. The first two indices
        represents the different derivatives with respect to each hyperparameter.

        :param x1: First input array. Must be a 1-D or 2-D array, with the length of
                   the last dimension matching the last dimension of ``x2`` and
                   one less than the length of ``params``. ``x1`` may be 1-D if either
                   each point consists of a single parameter (and ``params`` has length
                   2) or the array only contains a single point (in which case, the array
                   will be reshaped to ``(1, D - 1)``).
        :type x1: array-like
        :param x2: Second input array. The same restrictions that apply to ``x1`` also
                   apply here.
        :type x2: array-like
        :param params: Hyperparameter array. Must be 1-D with length one greater than
                       the last dimension of ``x1`` and ``x2``.
        :type params: array-like
        :returns: Array holding the second derivatives of the pair-wise distances between
                  points in arrays ``x1`` and ``x2`` with respect to the hyperparameters.
                  Will be an array with shape ``(D, D, n1, n2)``, where ``D`` is the length
                  of ``params``, ``n1`` is the length of the first axis of ``x1`` and
                  ``n2`` is the length of the first axis of ``x2``. The first two axes
                  indicates the different derivative components (i.e. the second derivative
                  with respect to the first parameter is [0,0,:,:], the mixed partial with
                  respect to the first and second parameters is [0,1,:,:] or [1,0,:,:], etc.)
        :rtype: ndarray
        """

        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)

        return np.reshape(self.calc_r2(x1, x2, params), (1, 1, n1, n2))

class StationaryKernel(KernelBase):
    r"""
    Generic class representing a stationary kernel

    This base class implements the necessary scaffolding for defining a stationary kernel.
    Stationary kernels are only dependent on a distance measure between any two points, so
    the base class holds all the necessary information for doing the distance computation.
    Individual subclasses will implement the functional dependence of the kernel on the
    distance, plus first and second derivatives (if desired) to compute the gradient or
    Hessian of the kernel with respect to the hyperparameters.

    This implementation uses a scaled euclidean distance metric. Each individual parameter
    has a hyperparameter scale associated with it that is used in the distance computation.
    If a different metric is to be defined, a new base class needs to be defined that
    implements the ``calc_r``, and optionally ``calc_drdtheta`` and ``calc_d2rdtheta2``
    methods if gradient or Hessian computation is desired. The methods ``kernel_f``,
    ``kernel_gradient``, and ``kernel_hessian`` can then be used to compute the appropriate
    quantities with no further modification.

    Note that the Kernel object just collates all of the methods together; the class itself
    does not hold any information on the data point or hyperparamters, which are passed
    directly to the appropriate methods. Thus, no information needs to be provided when
    creating a new ``Kernel`` instance.
    """

    def calc_r2(self, x1, x2, params):
        r"""
        Calculate squared distance between all pairs of points

        This method computes the scaled Euclidean distance between all pairs of points
        in ``x1`` and ``x2``. Each component distance is multiplied by the exponential of
        the corresponding hyperparameter, prior to summing and taking the square root.
        For example, if
        ``x1 = [1.]``, ``x2`` = [2.], and ``params = [2., 2.]`` then ``calc_r`` would
        return :math:`{\sqrt{exp(2)*(1 - 2)^2}=\sqrt{exp(2)}}`
        as an array with shape ``(1,1)``.

        :param x1: First input array. Must be a 1-D or 2-D array, with the length of
                   the last dimension matching the last dimension of ``x2`` and
                   one less than the length of ``params``. ``x1`` may be 1-D if either
                   each point consists of a single parameter (and ``params`` has length
                   2) or the array only contains a single point (in which case, the array
                   will be reshaped to ``(1, D - 1)``).
        :type x1: array-like
        :param x2: Second input array. The same restrictions that apply to ``x1`` also
                   apply here.
        :type x2: array-like
        :param params: Hyperparameter array. Must be 1-D with length one greater than
                       the last dimension of ``x1`` and ``x2``.
        :type params: array-like
        :returns: Array holding all pair-wise squared distances between points in arrays ``x1``
                  and ``x2``. Will be an array with shape ``(n1, n2)``, where ``n1``
                  is the length of the first axis of ``x1`` and ``n2`` is the length
                  of the first axis of ``x2``.
        :rtype: ndarray
        """

        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)
        
        exp_theta = np.exp(params)

        r2_matrix = np.sum(exp_theta*(x1[:, np.newaxis, :] - x2[np.newaxis, :, :])**2, axis=-1)

        if np.any(np.isinf(r2_matrix)):
            raise FloatingPointError("Inf enountered in kernel distance computation")

        return r2_matrix

    def calc_dr2dtheta(self, x1, x2, params):
        r"""
        Calculate the first derivative of the distance between all pairs of points with
        respect to the hyperparameters

        This method computes the derivative of the scaled Euclidean distance between
        all pairs of points in ``x1`` and ``x2`` with respect to the hyperparameters.
        The gradient is held in an array with shape ``(D, n1, n2)``, where ``D`` is
        the length of ``params``, ``n1`` is the length of the first axis of ``x1``,
        and ``n2`` is the length of the first axis of ``x2``. This is used in the
        computation of the gradient and Hessian of the kernel. The first index
        represents the different derivatives with respect to each hyperparameter.

        :param x1: First input array. Must be a 1-D or 2-D array, with the length of
                   the last dimension matching the last dimension of ``x2`` and
                   one less than the length of ``params``. ``x1`` may be 1-D if either
                   each point consists of a single parameter (and ``params`` has length
                   2) or the array only contains a single point (in which case, the array
                   will be reshaped to ``(1, D - 1)``).
        :type x1: array-like
        :param x2: Second input array. The same restrictions that apply to ``x1`` also
                   apply here.
        :type x2: array-like
        :param params: Hyperparameter array. Must be 1-D with length one greater than
                       the last dimension of ``x1`` and ``x2``.
        :type params: array-like
        :returns: Array holding the derivative of the pair-wise distances between
                  points in arrays ``x1`` and ``x2`` with respect to the hyperparameters.
                  Will be an array with shape ``(D, n1, n2)``, where ``D`` is the length
                  of ``params``, ``n1`` is the length of the first axis of ``x1`` and
                  ``n2`` is the length of the first axis of ``x2``. The first axis
                  indicates the different derivative components (i.e. the derivative
                  with respect to the first parameter is [0,:,:], etc.)
        :rtype: ndarray
        """

        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)

        exp_theta = np.exp(params)

        dr2dtheta = np.transpose(exp_theta*(x1[:, np.newaxis, :] - x2[np.newaxis, :, :])**2,
                                 (2, 0, 1))

        return dr2dtheta

    def calc_d2r2dtheta2(self, x1, x2, params):
        r"""
        Calculate all second derivatives of the distance between all pairs of points with
        respect to the hyperparameters

        This method computes all second derivatives of the scaled Euclidean distance
        between all pairs of points in ``x1`` and ``x2`` with respect to the
        hyperparameters. The gradient is held in an array with shape ``(D, D, n1, n2)``,
        where ``D`` is the length of ``params``, ``n1`` is the length of the first axis
        of ``x1``, and ``n2`` is the length of the first axis of ``x2``. This is used in
        the computation of the gradient and Hessian of the kernel. The first two indices
        represents the different derivatives with respect to each hyperparameter.

        :param x1: First input array. Must be a 1-D or 2-D array, with the length of
                   the last dimension matching the last dimension of ``x2`` and
                   one less than the length of ``params``. ``x1`` may be 1-D if either
                   each point consists of a single parameter (and ``params`` has length
                   2) or the array only contains a single point (in which case, the array
                   will be reshaped to ``(1, D - 1)``).
        :type x1: array-like
        :param x2: Second input array. The same restrictions that apply to ``x1`` also
                   apply here.
        :type x2: array-like
        :param params: Hyperparameter array. Must be 1-D with length one greater than
                       the last dimension of ``x1`` and ``x2``.
        :type params: array-like
        :returns: Array holding the second derivatives of the pair-wise distances between
                  points in arrays ``x1`` and ``x2`` with respect to the hyperparameters.
                  Will be an array with shape ``(D, D, n1, n2)``, where ``D`` is the length
                  of ``params``, ``n1`` is the length of the first axis of ``x1`` and
                  ``n2`` is the length of the first axis of ``x2``. The first two axes
                  indicates the different derivative components (i.e. the second derivative
                  with respect to the first parameter is [0,0,:,:], the mixed partial with
                  respect to the first and second parameters is [0,1,:,:] or [1,0,:,:], etc.)
        :rtype: ndarray
        """

        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)

        exp_theta = np.exp(-params[:(D - 1)])

        d2r2dtheta2 = np.zeros((D, D, n1, n2))

        idx = np.arange(D)
        
        d2r2dtheta2[idx, idx] = self.calc_dr2dtheta(x1, x2, params)

        return d2r2dtheta2

class ProductKernel(KernelBase):
    "Product form of kernel"
    
    def calc_r2(self, x1, x2, params):
        r"""
        Calculate squared distance between all pairs of points

        This method computes the scaled Euclidean distance between all pairs of points
        in ``x1`` and ``x2`` along each axis. Each component distance is multiplied by
        the exponential of the corresponding hyperparameter.
        For example, if
        ``x1 = [[1., 2.]]``, ``x2`` = [[2., 4.]], and ``params = [2., 2.]`` then ``calc_r`` would
        return the array :math:`{[exp(2)*(1 - 2)^2, exp(2)*(2 - 4)^2]}`
        as an array with shape ``(2, 1,1)``.

        :param x1: First input array. Must be a 1-D or 2-D array, with the length of
                   the last dimension matching the last dimension of ``x2`` and
                   one less than the length of ``params``. ``x1`` may be 1-D if either
                   each point consists of a single parameter (and ``params`` has length
                   2) or the array only contains a single point (in which case, the array
                   will be reshaped to ``(1, D)``).
        :type x1: array-like
        :param x2: Second input array. The same restrictions that apply to ``x1`` also
                   apply here.
        :type x2: array-like
        :param params: Hyperparameter array. Must be 1-D with length one greater than
                       the last dimension of ``x1`` and ``x2``.
        :type params: array-like
        :returns: Array holding all pair-wise squared distances between points in arrays ``x1``
                  and ``x2``. Will be an array with shape ``(D, n1, n2)``, where ``D`` is
                  the number of dimensions, ``n1`` is the length of the first axis of
                  ``x1`` and ``n2`` is the length of the first axis of ``x2``.
        :rtype: ndarray
        """

        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)
        
        exp_theta = np.exp(params)

        r2_matrix = exp_theta[np.newaxis, np.newaxis, :]*(x1[:, np.newaxis, :] - x2[np.newaxis, :, :])**2

        if np.any(np.isinf(r2_matrix)):
            raise FloatingPointError("Inf enountered in kernel distance computation")

        return r2_matrix
    
    def kernel_f(self, x1, x2, params):
        r"""
        Compute kernel values for a set of inputs

        Returns the value of the kernel for two sets of input points and a choice of
        hyperparameters. This function should not need to be modified for different choices
        of the kernel function or distance metric, as after checking the inputs it simply
        calls the routine to compute the distance metric and then evaluates the kernel function
        for those distances.

        :param x1: First input array. Must be a 1-D or 2-D array, with the length of
                   the last dimension matching the last dimension of ``x2`` and
                   one less than the length of ``params``. ``x1`` may be 1-D if either
                   each point consists of a single parameter (and ``params`` has length
                   2) or the array only contains a single point (in which case, the array
                   will be reshaped to ``(1, D - 1)``).
        :type x1: array-like
        :param x2: Second input array. The same restrictions that apply to ``x1`` also
                   apply here.
        :type x2: array-like
        :param params: Hyperparameter array. Must be 1-D with length one greater than
                       the last dimension of ``x1`` and ``x2``.
        :type params: array-like
        :returns: Array holding all kernel values between points in arrays ``x1``
                  and ``x2``. Will be an array with shape ``(n1, n2)``, where ``n1``
                  is the length of the first axis of ``x1`` and ``n2`` is the length
                  of the first axis of ``x2``.
        :rtype: ndarray
        """

        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)

        return np.prod(self.calc_K(self.calc_r2(x1, x2, params)), axis=-1)
        
    def kernel_deriv(self, x1, x2, params):
        r"""
        Compute kernel gradient for a set of inputs

        Returns the value of the kernel gradient for two sets of input points and a choice of
        hyperparameters. This function should not need to be modified for different choices
        of the kernel function or distance metric, as after checking the inputs it simply
        calls the routine to compute the distance metric, kernel function, and the appropriate
        derivative functions of the distance and kernel functions.

        :param x1: First input array. Must be a 1-D or 2-D array, with the length of
                   the last dimension matching the last dimension of ``x2`` and
                   one less than the length of ``params``. ``x1`` may be 1-D if either
                   each point consists of a single parameter (and ``params`` has length
                   2) or the array only contains a single point (in which case, the array
                   will be reshaped to ``(1, D - 1)``).
        :type x1: array-like
        :param x2: Second input array. The same restrictions that apply to ``x1`` also
                   apply here.
        :type x2: array-like
        :param params: Hyperparameter array. Must be 1-D with length one greater than
                       the last dimension of ``x1`` and ``x2``.
        :type params: array-like
        :returns: Array holding the gradient of the kernel function between points in arrays
                  ``x1`` and ``x2`` with respect to the hyperparameters. Will be an array with
                  shape ``(D, n1, n2)``, where ``D`` is the length of ``params``, ``n1`` is the
                  length of the first axis of ``x1`` and ``n2`` is the length of the first axis
                  of ``x2``. The first axis indicates the different derivative components
                  (i.e. the derivative with respect to the first parameter is [0,:,:], etc.)
        :rtype: ndarray
        """

        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)

        r2_matrix = self.calc_r2(x1, x2, params)
        
        diag = self.calc_dKdr2(r2_matrix)*r2_matrix
        
        dKdtheta = np.broadcast_to(self.calc_K(r2_matrix), (D, n1, n2, D)).copy()
        
        idx = np.arange(0, D, 1)
        
        dKdtheta[idx, :, :, idx] = np.transpose(diag, (2, 0, 1))
        
        return np.prod(dKdtheta, axis=-1)

    def kernel_hessian(self, x1, x2, params):
        r"""
        Calculate the Hessian of the kernel evaluated for all pairs of points with
        respect to the hyperparameters

        Returns the value of the kernel Hessian for two sets of input points and a choice of
        hyperparameters. This function should not need to be modified for different choices
        of the kernel function or distance metric, as after checking the inputs it simply
        calls the routine to compute the distance metric, kernel function, and the appropriate
        derivative functions of the distance and kernel functions.

        :param x1: First input array. Must be a 1-D or 2-D array, with the length of
                   the last dimension matching the last dimension of ``x2`` and
                   one less than the length of ``params``. ``x1`` may be 1-D if either
                   each point consists of a single parameter (and ``params`` has length
                   2) or the array only contains a single point (in which case, the array
                   will be reshaped to ``(1, D - 1)``).
        :type x1: array-like
        :param x2: Second input array. The same restrictions that apply to ``x1`` also
                   apply here.
        :type x2: array-like
        :param params: Hyperparameter array. Must be 1-D with length one greater than
                       the last dimension of ``x1`` and ``x2``.
        :type params: array-like
        :returns: Array holding the Hessian of the pair-wise distances between points in arrays
                  ``x1`` and ``x2`` with respect to the hyperparameters. Will be an array with
                  shape ``(D, D, n1, n2)``, where ``D`` is the length of ``params``, ``n1`` is
                  the length of the first axis of ``x1`` and ``n2`` is the length of the first
                  axis of ``x2``. The first two axes indicates the different derivative components
                  (i.e. the second derivative with respect to the first parameter is [0,0,:,:],
                  the mixed partial with respect to the first and second parameters is [0,1,:,:]
                  or [1,0,:,:], etc.)
        :rtype: ndarray
        """

        x1, n1, x2, n2, params, D = self._check_inputs(x1, x2, params)

        r2_matrix = self.calc_r2(x1, x2, params)
        
        off_diag = self.calc_dKdr2(r2_matrix)*r2_matrix
        
        diag = self.calc_d2Kdr22(r2_matrix)*r2_matrix**2 + self.calc_dKdr2(r2_matrix)*r2_matrix
        
        d2Kdtheta2 = np.broadcast_to(self.calc_K(r2_matrix), (D, D, n1, n2, D)).copy()
        
        idx = np.arange(0, D, 1)
        
        print(d2Kdtheta2)
        d2Kdtheta2[idx, :, :, :, idx] = np.transpose(off_diag[np.newaxis, :, :, :], (0, 3, 1, 2))
        print(d2Kdtheta2)
        d2Kdtheta2[:, idx, :, :, idx] = np.transpose(off_diag[np.newaxis, :, :, :], (0, 3, 1, 2))
        print(d2Kdtheta2)
        
        d2Kdtheta2[idx, idx, :, :, idx] = np.transpose(diag, (2, 0, 1))
        print(d2Kdtheta2)
        
        return np.prod(d2Kdtheta2, axis=-1)

class SqExpBase(object):
    r"""
    Base Implementation of the squared exponential kernel

    Class representing the spatial functions for the squared exponential kernel.
    """

    def calc_K(self, r2):
        r"""
        Compute K(r^2) for the squared exponential kernel

        This method implements the squared exponential kernel function as a function of distance.
        Given an array of distances, this function evaluates the kernel function of those values,
        returning an array of the same shape.

        :param r2: Array holding distances between all points. All values in this array must be
                  non-negative.
        :type r2: array-like
        :returns: Array holding kernel evaluations, with the same shape as the input ``r``
        :rtype: ndarray
        """

        assert np.all(r2 >= 0.), "kernel distances must be positive"

        r2 = np.array(r2)

        return np.exp(-0.5*r2)

    def calc_dKdr2(self, r2):
        r"""
        Calculate first derivative of the squared exponential kernel as a function of
        squared distance

        This method implements the first derivative of the squared exponential kernel
        function as a function of squared distance. Given an array of squared distances,
        this function evaluates the derivative function of those values, returning an
        array of the same shape.

        :param r2: Array holding squared distances between all points. All values in
                   this array must be non-negative.
        :type r2: array-like
        :returns: Array holding kernel derivatives, with the same shape as the input ``r2``
        :rtype: ndarray
        """

        assert np.all(r2 >= 0.), "kernel distances must be positive"

        r2 = np.array(r2)

        return -0.5*np.exp(-0.5*r2)

    def calc_d2Kdr22(self, r2):
        r"""
        Calculate second derivative of the squared exponential kernel as a
        function of squared distance

        This method implements the second derivative of the squared exponential
        kernel function as a function of squared distance. Given an array of
        squared distances, this function evaluates the second derivative
        function of those values, returning an array of the same shape.

        :param r2: Array holding distances between all points. All values in
                   this array must be non-negative.
        :type r2: array-like
        :returns: Array holding kernel second derivatives, with the same shape
                  as the input ``r2``
        :rtype: ndarray
        """

        assert np.all(r2 >= 0.), "kernel distances must be positive"

        r2 = np.array(r2)

        return 0.25*np.exp(-0.5*r2)

    def __str__(self):
        r"""
        Defines a string representation of the squared exponential kernel

        Returns a string representation of the squared exponential kernel. Note that since
        the kernel just collects methods for kernel evaluations together with no data, this
        is just a basic string that will not change for different instances of the class.

        :returns: String representation of the kernel
        :rtype: str
        """
        return "Squared Exponential Kernel"

class Mat52Base(object):
    r"""
    Implementation of the Matern 5/2 kernel

    Class representing the Matern 5/2 kernel. It derives from the base class for a
    stationary kernel, using the scaled Euclidean distance metric. The subclass then just
    defines the kernel function and its derivatives.
    """
    def calc_K(self, r2):
        r"""
        Compute K(r^2) for the Matern 5/2 kernel

        This method implements the Matern 5/2 kernel function as a function of 
        squared distance. Given an array of squared distances, this function
        evaluates the kernel function of those values, returning an array of
        the same shape.

        :param r2: Array holding squared distances between all points. All
                   values in this array must be non-negative.
        :type r2: array-like
        :returns: Array holding kernel evaluations, with the same shape as
                  the input ``r2``
        :rtype: ndarray
        """

        assert np.all(r2 >= 0.), "kernel distances must be positive"

        r2 = np.array(r2)

        return (1.+np.sqrt(5.*r2)+5./3.*r2)*np.exp(-np.sqrt(5.*r2))

    def calc_dKdr2(self, r2):
        r"""
        Calculate first derivative of the Matern 5/2 kernel as a function
        of squared distance

        This method implements the first derivative of the Matern 5/2 kernel
        function as a function of squared distance. Given an array of
        squared distances, this function evaluates the derivative
        function of those values, returning an array of the same shape.

        :param r2: Array holding squared distances between all points. All
                   values in this array must be non-negative.
        :type r2: array-like
        :returns: Array holding kernel derivatives, with the same shape as
                  the input ``r2``
        :rtype: ndarray
        """

        assert np.all(r2 >= 0.), "kernel distances must be positive"

        r2 = np.array(r2)

        return -5./6.*(1.+np.sqrt(5.*r2))*np.exp(-np.sqrt(5*r2))

    def calc_d2Kdr22(self, r2):
        r"""
        Calculate second derivative of the squared exponential kernel as
        a function of squared distance

        This method implements the second derivative of the squared
        exponential kernel function as a function of squared distance.
        Given an array of squared distances, this function evaluates the
        second derivative function of those values, returning an array
        of the same shape.

        :param r2: Array holding squared distances between all points.
                   All values in this array must be non-negative.
        :type r2: array-like
        :returns: Array holding kernel second derivatives, with the same
                  shape as the input ``r2``
        :rtype: ndarray
        """

        assert np.all(r2 >= 0.), "kernel distances must be positive"

        r2 = np.array(r2)

        return 25./12.*np.exp(-np.sqrt(5.*r2))

    def __str__(self):
        r"""
        Defines a string representation of the Matern 5/2 kernel

        Returns a string representation of the Matern 5/2 kernel. Note that since
        the kernel just collects methods for kernel evaluations together with no data, this
        is just a basic string that will not change for different instances of the class.

        :returns: String representation of the kernel
        :rtype: str
        """
        return "Matern 5/2 Kernel"

class SquaredExponential(SqExpBase,StationaryKernel):
    """
    Squared Exponential Kernel
    
    Inherits from ``SqExpBase`` and ``StationaryKernel``, so this will be
    a stationary kernel with one correlation length per input dimension
    and a squared exponential fall-off with distance.
    """
    pass

class UniformSqExp(SqExpBase,UniformKernel):
    """
    Uniform Squared Exponential Kernel
    
    Inherits from ``SqExpBase`` and ``UniformKernel``, so this will be
    a uniform kernel with one correlation length (independent of the
    number of dimensions) and a squared exponential fall-off with distance.
    """
    pass

class Matern52(Mat52Base,StationaryKernel):
    """
    Matern 5/2 Kernel
    
    Inherits from ``Mat52Base`` and ``StationaryKernel``, so this will be
    a stationary kernel with one correlation length per input dimension
    and a Matern 5/2 fall-off with distance.
    """
    pass
    
class UniformMat52(Mat52Base,UniformKernel):
    """
    Uniform Matern 5/2 Kernel
    
    Inherits from ``Mat52Base`` and ``UniformKernel``, so this will be
    a uniform kernel with one correlation length (independent of the
    number of dimensions) and a Matern 5/2 fall-off with distance.
    """
    pass
    
class ProductMat52(Mat52Base,ProductKernel):
    """
    Product Matern 5/2 Kernel
    
    Inherits from ``Mat52Base`` and ``ProductKernel``, so this will
    be a kernel with one correlation length per input dimension.
    The Matern 5/2 fall-off function is applied to each dimension
    *before* taking the product of all dimensions in this case.
    Generally results in a slighly smoother kernel than the
    stationary version of the Matern 5/2 kernel.
    """
    pass