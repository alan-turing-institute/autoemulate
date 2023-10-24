"""
**MeanFunction Module**

The MeanFunction module contains classes used for constructing mean functions for GP emulators.
A base ``MeanBase`` class is provided, which implements basic operations to combine fixed
functions and fitting parameters. The basic operations ``f1 + f2``, ``f1*f2``, ``f1**f2``
and ``f1(f2)`` are available, though not all possible combinations will make sense.
Particular cases where combinations do not make sense often use classes that represent free
fitting parameters, or if you attempt to raise a mean function to a power that is not
independent of the inputs. These operations will create new derived classes ``MeanSum``,
``MeanProduct``, ``MeanPower``, and ``MeanComposite``, from which more complex
regression functions can be formed. The derived sum, product, power, and composite mean classes
call the necessary methods to compute the function and derivatives from the more basic
classes and then combine them using sum, product, power, and chain rules for function evaluation
and derivatives.

The basic building blocks are fixed mean functions, derived from ``FixedMean``, and free
parameters, represented by the ``Coefficient`` class. Incuded fixed functions include
``ConstantMean`` and ``LinearMean``. Additional derived ``FixedMean`` functions can be
created by initializing a new ``FixedMean`` instance where the user provides a fixed
function and its derivative, and these can be combined to form arbitrarily complex mean functions.
Future improvements will extend the number of pre-defined function options.

One implementation note: ``CompositeMean`` does not implement the Hessian, as computing this
requires mixed partials involving inputs and parameters that are not normally implemented.
If a composite mean is required with a Hessian Computation, the user must implement this.

Additionally, note that given mean function may have a number of parameters that depends on
the shape of the input. Since the mean function does not store input, but rather provides
a way to collate functions and derivatives together in a single object, the number of parameters
can vary based on the inputs. This is particularly true for the provided ``PolynomialMean``
class, which fits a polynomial function of a fixed degree to each input parameter. Thus,
the number of parameters depends on the input shape.

In addition to manually creating a mean function by composing fixed functions and fitting parameters,
a ``MeanBase`` subclass can be created by using the ``MeanFunction`` function. ``MeanFunction``
is a functional interface for creating ``MeanBase`` subclasses from a string formula.
The formula langauge supports the operations described above as expected (see below for some
examples), with the option to first parse the formula using the Python library ``patsy``
before converting the terms to the respective subclasses of ``MeanBase``. Formulas specify
input variables using either ``x[i]`` or ``inputs[i]`` to represent the dependent variables,
and can explicitly include a leading ``"y ="`` or ``"y ~"`` (which will be ignored). Optionally,
named variables can be mapped to input dimensions by providing a dictionary mapping strings
to integer indices. Any other variables in the formula will be assumed to be fitting
coefficients. Note that the formula parser does not make any effort to simplify expressions
(such as having identical terms or a term with redundant fitting parameters), so it is up
to the user to get things correct. Converting a mean function instance to a string can
be very helpful in determining if the parsing led to any problems, see below.

Example: ::

    >>> from mogp_emulator.MeanFunction import Coefficient, LinearMean, MeanFunction
    >>> mf1 = Coefficient() + Coefficient()*LinearMean()
    >>> print(mf1)
    c + c*x[0]
    >>> mf2 = LinearMean(1)*LinearMean(2)
    >>> print(mf2)
    x[1]*x[2]
    >>> mf3 = mf1(mf2)
    >>> print(mf3)
    c + c*x[1]*x[2]
    >>> mf4 = Coefficient()*LinearMean()**2
    >>> print(mf4)
    c*x[0]^2
    >>> mf5 = MeanFunction("x[0]")
    >>> print(mf5)
    c + c*x[0]
    >>> mf6 = MeanFunction("y = a + b*x[0]", use_patsy=False)
    >>> print(mf6)
    c + c*x[0]
    >>> mf7 = MeanFunction("a*b", {"a": 0, "b": 1})
    >>> print(mf7)
    c + c*x[0] + c*x[1] + c*x[0]*x[1]
"""
import numpy as np
from functools import partial
from inspect import signature
from mogp_emulator.formula import mean_from_patsy_formula, mean_from_string

def MeanFunction(formula, inputdict={}, use_patsy=True):
    """
    Create a mean function from a formula

    This is the functional interface to creating a mean function from a string formula.
    This method takes a string as an input, an optional dictionary that map strings to
    integer indices in the input data, and an optional boolean flag that indicates if
    the user would like to have the formula parsed with patsy before being converted
    to a mean function.

    The string formulas can be specified in several ways. The formula LHS is implicitly
    always ``"y = "`` or ``"y ~ "``, though these can be explicitly provided as well.
    The RHS may contain a set of terms containing the add, multiply, power, and
    call operations much in the same way that the operations would be entered as
    regular python code. Parentheses are used to indicated prececence as well as
    the call operation, and square brackets indicate an indexing operation on the
    inputs. Inputs may be specified as either a string such as ``"x[0]"``,
    ``"inputs[0]"``, or a string that can be mapped to an integer index with the
    optional dictionary passed to the function. Any strings not representing operations
    or inputs as described above are interpreted as follows: if the string can
    be converted into a number, then it is interpreted as a ``ConstantMean`` fixed
    mean function object; otherwise it is assumed to represent a fitting coefficient.
    Note that this means many characters that do not represent operations within this
    mean function language but would not normally be considered as python variables
    will nonetheless be converted into fitting coefficients -- it is up to the user
    to get this right.

    Expressions that are repeated or redundant will not be simplified, so the user should
    take care that the provided expression is sensible as a mean function and will not
    cause problems when fitting.

    Additional special cases to be aware of:

    * ``call`` cannot be used as a variable name, if this is parsed as a token an exception
      will be raised.
    * ``I`` is the identity operator, it simply returns the given value. It is useful
      if you wish to use patsy to evaluate a formula but protect a part of the string
      formula from being expanded based on the rules in patsy. If ``I`` is encountered
      in any other context, an exception will be raised.

    Examples: ::

        >>> from mogp_emulator.MeanFunction import MeanFunction
        >>> mf1 = MeanFunction("x[0]")
        >>> print(mf1)
        c + c*x[0]
        >>> mf2 = MeanFunction("y = a + b*x[0]", use_patsy=False)
        >>> print(mf2)
        c + c*x[0]
        >>> mf3 = MeanFunction("a*b", {"a": 0, "b": 1})
        >>> print(mf3)
        c + c*x[0] + c*x[1] + c*x[0]*x[1]

    :param formula: string representing the desired mean function formula
    :type formula: str
    :param inputdict: dictionary used to map variables to input indices. Maps
                      strings to integer indices (must be non-negative). Optional,
                      default is ``{}``.
    :type inputdict: dict
    :param use_patsy: Boolean flag indicating if the string is to be parsed using
                      patsy library. Optional, default is ``True``. If patsy is not
                      installed, the basic string parser will be used.
    :type use_patsy: bool
    :returns: New subclass of ``MeanBase`` implementing the given formula
    :rtype: subclass of MeanBase (exact type will depend on the formula that is provided)
    """

    if formula is None or (isinstance(formula, str) and formula.strip() == ""):
        return ConstantMean(0.)

    if not isinstance(formula, str):
        raise ValueError("input formula must be a string")

    if use_patsy:
        mf = mean_from_patsy_formula(formula, inputdict)
    else:
        mf = mean_from_string(formula, inputdict)

    return mf

class MeanBase(object):
    """
    Base mean function class

    The base class for the mean function implementation includes code for checking inputs and
    implements sum, product, power, and composition methods to allow more complicated functions
    to be built up from fixed functions and fitting coefficients. Subclasses need to implement
    the following methods:

    * ``get_n_params`` which returns the number of parameters for a given input size. This is
      usually a constant, but can be more complicated (such as the provided ``PolynomialMean``
      class)
    * ``mean_f`` computes the mean function for the provided inputs and parameters
    * ``mean_deriv`` computes the derivative with respect to the parameters
    * ``mean_hessian`` computes the hessian with respect to the parameters
    * ``mean_inputderiv`` computes the derivate with respect to the inputs

    The base class does not have any attributes, but subclasses will usually have some
    attributes that must be set and so are likely to need a ``__init__`` method.
    """
    def _check_inputs(self, x, params):
        """
        Check the shape of the inputs and reshape if needed

        This method checks that the inputs and parameters are consistent for the provided
        mean function. In particular, the following must be met:

        * The inputs ``x`` must be a 2D numpy array, though if it is 1D it is reshaped to add
          a second dimenion of length 1.
        * ``params`` must be a 1D numpy array. If a multi-dimensional array is provided, it
          will be flattened.
        * ``params`` must have a length that is the same as the return value of ``get_n_params``
          when called with the inputs. Note that some mean functions may have different
          numbers of parameters depending on the inputs, so this may not be known in advance.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: tuple containing the reshaped ``x`` and ``params`` arrays
        :rtype: tuple containing two ndarrays
        """

        x = np.array(x)
        params = np.array(params).flatten()

        if len(x.shape) == 1:
            x = np.reshape(x, (-1, 1))
        assert len(x.shape) == 2, "inputs must be a 1D or 2D array"

        assert len(params.shape) == 1, "params must be a 1D array"

        assert len(params) == self.get_n_params(x), "bad length for params"

        return x, params

    def get_n_params(self, x):
        """
        Determine the number of parameters

        Returns the number of parameters for the mean function, which possibly depends on x.

        :param x: Input array
        :type x: ndarray
        :returns: number of parameters
        :rtype: int
        """

        raise NotImplementedError("base mean function does not implement a particular function")

    def mean_f(self, x, params):
        """
        Returns value of mean function

        Method to compute the value of the mean function for the inputs and parameters provided.
        Shapes of ``x`` and ``params`` must be consistent based on the return value of the
        ``get_n_params`` method. Returns a numpy array of shape ``(x.shape[0],)`` holding
        the value of the mean function for each input point.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function evaluated at all input points, numpy array of shape
                  ``(x.shape[0],)``
        :rtype: ndarray
        """

        raise NotImplementedError("base mean function does not implement a particular function")

    def mean_deriv(self, x, params):
        """
        Returns value of mean function derivative wrt the parameters

        Method to compute the value of the mean function derivative with respect to the
        parameters for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        Returns a numpy array of shape ``(n_params, x.shape[0])`` holding the value of the mean
        function derivative with respect to each parameter (first axis) for each input point
        (second axis).

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function derivative with respect to the parameters evaluated
                  at all input points, numpy array of shape ``(n_params, x.shape[0])``
        :rtype: ndarray
        """

        raise NotImplementedError("base mean function does not implement a particular function")

    def mean_hessian(self, x, params):
        """
        Returns value of mean function Hessian wrt the parameters

        Method to compute the value of the mean function Hessian with respect to the
        parameters for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        Returns a numpy array of shape ``(n_params, n_params, x.shape[0])`` holding the value
        of the mean function second derivaties with respect to each parameter pair (first twp axes)
        for each input point (last axis).

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function Hessian with respect to the parameters evaluated
                  at all input points, numpy array of shape ``(n_parmas, n_params, x.shape[0])``
        :rtype: ndarray
        """

        raise NotImplementedError("base mean function does not implement a particular function")

    def mean_inputderiv(self, x, params):
        """
        Returns value of mean function derivative wrt the inputs

        Method to compute the value of the mean function derivative with respect to the
        inputs for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        Returns a numpy array of shape ``(x.shape[1], x.shape[0])`` holding the value of the mean
        function derivative with respect to each input (first axis) for each input point
        (second axis).

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function derivative with respect to the inputs evaluated
                  at all input points, numpy array of shape ``(x.shape[1], x.shape[0])``
        :rtype: ndarray
        """

        raise NotImplementedError("base mean function does not implement a particular function")

    def __add__(self, other):
        """
        Adds two mean functions

        This method adds two mean functions, returning a ``MeanSum`` object. If the second
        argument is a float or integer, it is converted to a ``ConstantMean`` object. If
        the second argument is neither a subclass of ``MeanBase`` nor a float/int,
        an exception is raised.

        :param other: Second ``MeanBase`` (or float/integer) to be added
        :type other: subclass of MeanBase or float or int
        :returns: ``MeanSum`` instance
        :rtype: MeanSum
        """

        if issubclass(type(other), MeanBase):
            return MeanSum(self, other)
        elif isinstance(other, (float, int)):
            return MeanSum(self, ConstantMean(other))
        else:
            raise TypeError("other function cannot be added with a MeanBase")

    def __radd__(self, other):
        """
        Right adds two mean functions

        This method adds two mean functions, returning a ``MeanSum`` object. If the second
        argument is a float or integer, it is converted to a ``ConstantMean`` object. If
        the second argument is neither a subclass of ``MeanBase`` nor a float/int,
        an exception is raised.

        :param other: Second ``MeanBase`` (or float/integer) to be added
        :type other: subclass of MeanBase or float or int
        :returns: ``MeanSum`` instance
        :rtype: MeanSum
        """

        if issubclass(type(other), MeanBase):
            return MeanSum(other, self)
        elif isinstance(other, (float, int)):
            return MeanSum(ConstantMean(other), self)
        else:
            raise TypeError("other function cannot be added with a MeanBase")

    def __mul__(self, other):
        """
        Multiplies two mean functions

        This method multiples two mean functions, returning a ``MeanProduct`` object. If
        the second argument is a float or integer, it is converted to a ``ConstantMean``
        object. If the second argument is neither a subclass of ``MeanBase`` nor a
        float/int, an exception is raised.

        :param other: Second ``MeanBase`` (or float/integer) to be multiplied
        :type other: subclass of MeanBase or float or int
        :returns: ``MeanProduct`` instance
        :rtype: MeanProduct
        """

        if issubclass(type(other), MeanBase):
            return MeanProduct(self, other)
        elif isinstance(other, (float, int)):
            return MeanProduct(self, ConstantMean(other))
        else:
            raise TypeError("other function cannot be multiplied with a MeanBase")

    def __rmul__(self, other):
        """
        Right multiplies two mean functions

        This method multiples two mean functions, returning a ``MeanProduct`` object. If
        the second argument is a float or integer, it is converted to a ``ConstantMean``
        object. If the second argument is neither a subclass of ``MeanBase`` nor a
        float/int, an exception is raised.

        :param other: Second ``MeanBase`` (or float/integer) to be multiplied
        :type other: subclass of MeanBase or float or int
        :returns: ``MeanProduct`` instance
        :rtype: MeanProduct
        """

        if issubclass(type(other), MeanBase):
            return MeanProduct(other, self)
        elif isinstance(other, (float, int)):
            return MeanProduct(ConstantMean(other), self)
        else:
            raise TypeError("other function cannot be multipled with a MeanBase")

    def __pow__(self, exp):
        """
        Raises a mean function to a power

        This method raises a mean function to a power, returning a ``MeanPower`` object.
        The second argument can only be a mean function that returns a value that is
        independent of its input, in particular a ``Coefficient`` or a ``ConstantMean``
        (or a float or integer, from which a new ``ConstantMean`` will be created)
        are the only acceptable types for the ``exp`` argument.

        :param exp: Mean function exponent, must be a ``Coefficient`` or a ``ConstantMean``
                    object, or a float/int from which a new ``ConstantMean`` will be
                    created.
        :type exp: Coefficient, ConstantMean, float, or int
        :returns: ``MeanPower``instance
        :rtype: MeanPower
        """
        if isinstance(exp, (float, int)):
            return MeanPower(self, ConstantMean(exp))
        elif isinstance(exp, (Coefficient, ConstantMean)):
            return MeanPower(self, exp)
        else:
            raise TypeError("MeanBase can only be raised to a power that is a ConstantMean, " +
                            "Coefficient, or float/int")

    def __rpow__(self, base):
        """
        Right raises a mean function to a power

        This method right raises a mean function to a power, meaning that the base
        is potentially not a mean function. Returns a ``MeanPower`` object. The  ``self``
        argument can only be a mean function that returns a value that is
        independent of its input, in particular a ``Coefficient`` or a ``ConstantMean``.
        The base can be any ``MeanBase`` instance, or a float or integer, from which a new
        ``ConstantMean`` will be created.

        :param base: Mean function base, must be a ``MeanBase`` subclass
                     object, or a float/int from which a new ``ConstantMean`` will be
                     created.
        :type base: MeanBase subclass, float, or int
        :returns: ``MeanPower``instance
        :rtype: MeanPower
        """
        if not isinstance(self, (Coefficient, ConstantMean)):
            raise TypeError("arbitrary mean functions cannot serve as the exponent when " +
                            "raising a mean function to a power")
        if isinstance(base, (float, int)):
            return MeanPower(ConstantMean(base), self)
        elif issubclass(type(base), MeanBase):
            return MeanPower(base, self)
        else:
            raise TypeError("base in a MeanPower must be a MeanBase or a float/int")

    def __call__(self, other):
        """
        Composes two mean functions

        This method multiples two mean functions, returning a ``MeanComposite`` object.
        If the second argument is not a subclass of ``MeanBase``, an exception is
        raised.

        :param other: Second ``MeanBase`` to be composed
        :type other: subclass of MeanBase
        :returns: ``MeanComposite`` instance
        :rtype: MeanComposite
        """

        if issubclass(type(other), MeanBase):
            return MeanComposite(self, other)
        else:
            raise TypeError("other function cannot be composed with a MeanBase")

class MeanSum(MeanBase):
    """
    Class representing the sum of two mean functions

    This derived class represents the sum of two mean functions, and does the necessary
    bookkeeping needed to compute the required function and derivatives. The code does
    not do any checks to confirm that it makes sense to add these particular mean functions --
    in particular, adding two ``Coefficient`` classes is the same as having a single
    one, but the code will not attempt to simplify this so it is up to the user to get it
    right.

    :ivar f1: first ``MeanBase`` to be added
    :type f1: subclass of MeanBase
    :ivar f2: second ``MeanBase`` to be added
    :type f2: subclass of MeanBase
    """
    def __init__(self, f1, f2):
        """
        Create a new instance of two added mean functions

        Creates an instance of to added mean functions. Inputs are the two functions
        to be added, which must be subclasses of the base ``MeanBase`` class.

        :param f1: first ``MeanBase`` to be added
        :type f1: subclass of MeanBase
        :param f2: second ``MeanBase`` to be added
        :type f2: subclass of MeanBase
        :returns: new ``MeanSum`` instance
        :rtype: MeanSum
        """
        if not issubclass(type(f1), MeanBase):
            raise TypeError("inputs to MeanSum must be subclasses of MeanBase")
        if not issubclass(type(f2), MeanBase):
            raise TypeError("inputs to MeanSum must be subclasses of MeanBase")

        self.f1 = f1
        self.f2 = f2

    def get_n_params(self, x):
        """
        Determine the number of parameters

        Returns the number of parameters for the mean function, which possibly depends on x.

        :param x: Input array
        :type x: ndarray
        :returns: number of parameters
        :rtype: int
        """
        return self.f1.get_n_params(x) + self.f2.get_n_params(x)

    def mean_f(self, x, params):
        """
        Returns value of mean function

        Method to compute the value of the mean function for the inputs and parameters provided.
        Shapes of ``x`` and ``params`` must be consistent based on the return value of the
        ``get_n_params`` method. Returns a numpy array of shape ``(x.shape[0],)`` holding
        the value of the mean function for each input point.

        For ``MeanSum``, this method applies the sum rule to the results of computing
        the mean for the individual functions.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function evaluated at all input points, numpy array of shape
                  ``(x.shape[0],)``
        :rtype: ndarray
        """

        switch = self.f1.get_n_params(x)

        return (self.f1.mean_f(x, params[:switch]) +
                self.f2.mean_f(x, params[switch:]))

    def mean_deriv(self, x, params):
        """
        Returns value of mean function derivative wrt the parameters

        Method to compute the value of the mean function derivative with respect to the
        parameters for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        Returns a numpy array of shape ``(n_params, x.shape[0])`` holding the value of the mean
        function derivative with respect to each parameter (first axis) for each input point
        (second axis).

        For ``MeanSum``, this method applies the sum rule to the results of computing
        the derivative for the individual functions.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function derivative with respect to the parameters evaluated
                  at all input points, numpy array of shape ``(n_params, x.shape[0])``
        :rtype: ndarray
        """

        switch = self.f1.get_n_params(x)

        deriv = np.zeros((self.get_n_params(x), x.shape[0]))

        deriv[:switch] = self.f1.mean_deriv(x, params[:switch])
        deriv[switch:] = self.f2.mean_deriv(x, params[switch:])

        return deriv

    def mean_hessian(self, x, params):
        """
        Returns value of mean function Hessian wrt the parameters

        Method to compute the value of the mean function Hessian with respect to the
        parameters for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        Returns a numpy array of shape ``(n_params, n_params, x.shape[0])`` holding the value
        of the mean function second derivaties with respect to each parameter pair (first twp axes)
        for each input point (last axis).

        For ``MeanSum``, this method applies the sum rule to the results of computing
        the Hessian for the individual functions.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function Hessian with respect to the parameters evaluated
                  at all input points, numpy array of shape ``(n_parmas, n_params, x.shape[0])``
        :rtype: ndarray
        """

        switch = self.f1.get_n_params(x)

        hess = np.zeros((self.get_n_params(x), self.get_n_params(x), x.shape[0]))

        hess[:switch, :switch] = self.f1.mean_hessian(x, params[:switch])
        hess[switch:, switch:] = self.f2.mean_hessian(x, params[switch:])

        return hess

    def mean_inputderiv(self, x, params):
        """
        Returns value of mean function derivative wrt the inputs

        Method to compute the value of the mean function derivative with respect to the
        inputs for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        Returns a numpy array of shape ``(x.shape[1], x.shape[0])`` holding the value of the mean
        function derivative with respect to each input (first axis) for each input point
        (second axis).

        For ``MeanSum``, this method applies the sum rule to the results of computing
        the derivative for the individual functions.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function derivative with respect to the inputs evaluated
                  at all input points, numpy array of shape ``(x.shape[1], x.shape[0])``
        :rtype: ndarray
        """

        switch = self.f1.get_n_params(x)

        return (self.f1.mean_inputderiv(x, params[:switch]) +
                self.f2.mean_inputderiv(x, params[switch:]))

    def __str__(self):
        """
        Returns a string representation

        Return a formula-like representation of the Mean Function. Useful for confirming
        that a formula was correctly parsed.
        """

        return "{} + {}".format(self.f1, self.f2)

class MeanProduct(MeanBase):
    """
    Class representing the product of two mean functions

    This derived class represents the product of two mean functions, and does the necessary
    bookkeeping needed to compute the required function and derivatives. The code does
    not do any checks to confirm that it makes sense to multiply these particular mean functions --
    in particular, multiplying two ``Coefficient`` classes is the same as having a single
    one, but the code will not attempt to simplify this so it is up to the user to get it
    right.

    :ivar f1: first ``MeanBase`` to be multiplied
    :type f1: subclass of MeanBase
    :ivar f2: second ``MeanBase`` to be multiplied
    :type f2: subclass of MeanBase
    """
    def __init__(self, f1, f2):
        """
        Create a new instance of two mulitplied mean functions

        Creates an instance of to multiplied mean functions. Inputs are the two functions
        to be multiplied, which must be subclasses of the base ``MeanBase`` class.

        :param f1: first ``MeanBase`` to be multiplied
        :type f1: subclass of MeanBase
        :param f2: second ``MeanBase`` to be multiplied
        :type f2: subclass of MeanBase
        :returns: new ``MeanProduct`` instance
        :rtype: MeanProduct
        """

        if not issubclass(type(f1), MeanBase):
            raise TypeError("inputs to MeanProduct must be subclasses of MeanBase")
        if not issubclass(type(f2), MeanBase):
            raise TypeError("inputs to MeanProduct must be subclasses of MeanBase")

        self.f1 = f1
        self.f2 = f2

    def get_n_params(self, x):
        """
        Determine the number of parameters

        Returns the number of parameters for the mean function, which possibly depends on x.

        :param x: Input array
        :type x: ndarray
        :returns: number of parameters
        :rtype: int
        """
        return self.f1.get_n_params(x) + self.f2.get_n_params(x)

    def mean_f(self, x, params):
        """
        Returns value of mean function

        Method to compute the value of the mean function for the inputs and parameters provided.
        Shapes of ``x`` and ``params`` must be consistent based on the return value of the
        ``get_n_params`` method. Returns a numpy array of shape ``(x.shape[0],)`` holding
        the value of the mean function for each input point.

        For ``MeanProduct``, this method applies the product rule to the results of computing
        the mean for the individual functions.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function evaluated at all input points, numpy array of shape
                  ``(x.shape[0],)``
        :rtype: ndarray
        """

        switch = self.f1.get_n_params(x)

        return (self.f1.mean_f(x, params[:switch])*
                self.f2.mean_f(x, params[switch:]))

    def mean_deriv(self, x, params):
        """
        Returns value of mean function derivative wrt the parameters

        Method to compute the value of the mean function derivative with respect to the
        parameters for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        Returns a numpy array of shape ``(n_params, x.shape[0])`` holding the value of the mean
        function derivative with respect to each parameter (first axis) for each input point
        (second axis).

        For ``MeanProduct``, this method applies the product rule to the results of computing
        the derivative for the individual functions.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function derivative with respect to the parameters evaluated
                  at all input points, numpy array of shape ``(n_params, x.shape[0])``
        :rtype: ndarray
        """

        switch = self.f1.get_n_params(x)

        deriv = np.zeros((self.get_n_params(x), x.shape[0]))

        deriv[:switch] = (self.f1.mean_deriv(x, params[:switch])*
                          self.f2.mean_f(x, params[switch:]))

        deriv[switch:] = (self.f1.mean_f(x, params[:switch])*
                          self.f2.mean_deriv(x, params[switch:]))

        return deriv

    def mean_hessian(self, x, params):
        """
        Returns value of mean function Hessian wrt the parameters

        Method to compute the value of the mean function Hessian with respect to the
        parameters for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        Returns a numpy array of shape ``(n_params, n_params, x.shape[0])`` holding the value
        of the mean function second derivaties with respect to each parameter pair (first twp axes)
        for each input point (last axis).

        For ``MeanProduct``, this method applies the product rule to the results of computing
        the Hessian for the individual functions.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function Hessian with respect to the parameters evaluated
                  at all input points, numpy array of shape ``(n_parmas, n_params, x.shape[0])``
        :rtype: ndarray
        """

        switch = self.f1.get_n_params(x)

        hess = np.zeros((self.get_n_params(x), self.get_n_params(x), x.shape[0]))

        hess[:switch, :switch] = (self.f1.mean_hessian(x, params[:switch])*
                                  self.f2.mean_f(x, params[switch:]))
        hess[:switch, switch:, :] = (self.f1.mean_deriv(x, params[:switch])[:,np.newaxis,:]*
                                     self.f2.mean_deriv(x, params[switch:])[np.newaxis,:,:])
        hess[switch:, :switch, :] = np.transpose(hess[:switch, switch:, :], (1, 0, 2))
        hess[switch:, switch:] = (self.f1.mean_f(x, params[:switch])*
                                  self.f2.mean_hessian(x, params[switch:]))

        return hess


    def mean_inputderiv(self, x, params):
        """
        Returns value of mean function derivative wrt the inputs

        Method to compute the value of the mean function derivative with respect to the
        inputs for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        Returns a numpy array of shape ``(x.shape[1], x.shape[0])`` holding the value of the mean
        function derivative with respect to each input (first axis) for each input point
        (second axis).

        For ``MeanProduct``, this method applies the product rule to the results of computing
        the derivative for the individual functions.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function derivative with respect to the inputs evaluated
                  at all input points, numpy array of shape ``(x.shape[1], x.shape[0])``
        :rtype: ndarray
        """

        switch = self.f1.get_n_params(x)

        return (self.f1.mean_inputderiv(x, params[:switch])*
                self.f2.mean_f(x, params[switch:]) +
                self.f1.mean_f(x, params[:switch])*
                self.f2.mean_inputderiv(x, params[switch:]))

    def __str__(self):
        """
        Returns a string representation

        Return a formula-like representation of the Mean Function. Useful for confirming
        that a formula was correctly parsed.
        """

        return "{}*{}".format(self.f1, self.f2)

class MeanPower(MeanBase):
    """
    Class representing a mean function raised to a power

    This derived class represents a mean function raised to a power, and does the necessary
    bookkeeping needed to compute the required function and derivatives. The code requires
    that the exponent be either a ``Coefficient``, ``ConstantMean``, ``float``, or ``int``
    as the output of the exponent mean function must be independent of the inputs to make
    sense. If input is a float or int, a ``ConstantMean`` instance will be created.

    :ivar f1: first ``MeanBase`` to be raised to the given exponent
    :type f1: subclass of MeanBase
    :ivar f2: second ``MeanBase`` indicating the exponent. Must be a ``Coefficient``,
              ``ConstantMean``, or float/int (from which a ``ConstantMean`` object will
              be created)
    :type f2: Coefficient, ConstantMean, float, or int
    """
    def __init__(self, f1, f2):
        """
        Create a new instance of a mean function raised to a power

        Creates an instance of a mean function raised to a power. Inputs are the two
        functions (base, exponent), the first of which must be subclass of the base
        ``MeanBase`` class, and the second must be a ``Coefficient`` or a
        ``ConstantMean`` (or a float or int, from which a ``ConstantMean`` will
        be created).

        :param f1: first ``MeanBase`` serving as the base
        :type f1: subclass of MeanBase
        :param f2: second ``MeanBase`` serving as the exponent, must be a ``Coefficient``,
                   ``ConstantMean``, ``float``, or ``int``
        :type f2: Coefficient, ConstantMean, float, or int
        :returns: new ``MeanPower`` instance
        :rtype: MeanPower
        """

        if not issubclass(type(f1), MeanBase):
            raise TypeError("first input to MeanPower must be a subclass of MeanBase")
        if isinstance(f2, (float, int)):
            f2 = ConstantMean(f2)
        if not isinstance(f2, (ConstantMean, Coefficient)):
            raise TypeError("second input to MeanPower must be a Coefficient, ConstantMean, "
                            "float, or int")

        self.f1 = f1
        self.f2 = f2

    def get_n_params(self, x):
        """
        Determine the number of parameters

        Returns the number of parameters for the mean function, which possibly depends on x.

        :param x: Input array
        :type x: ndarray
        :returns: number of parameters
        :rtype: int
        """
        return self.f1.get_n_params(x) + self.f2.get_n_params(x)

    def mean_f(self, x, params):
        """
        Returns value of mean function

        Method to compute the value of the mean function for the inputs and parameters provided.
        Shapes of ``x`` and ``params`` must be consistent based on the return value of the
        ``get_n_params`` method. Returns a numpy array of shape ``(x.shape[0],)`` holding
        the value of the mean function for each input point.

        For ``MeanProduct``, this method applies the product rule to the results of computing
        the mean for the individual functions.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function evaluated at all input points, numpy array of shape
                  ``(x.shape[0],)``
        :rtype: ndarray
        """

        switch = self.f1.get_n_params(x)

        return (self.f1.mean_f(x, params[:switch])**
                self.f2.mean_f(x, params[switch:]))

    def mean_deriv(self, x, params):
        """
        Returns value of mean function derivative wrt the parameters

        Method to compute the value of the mean function derivative with respect to the
        parameters for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        Returns a numpy array of shape ``(n_params, x.shape[0])`` holding the value of the mean
        function derivative with respect to each parameter (first axis) for each input point
        (second axis).

        For ``MeanPpwer``, this method applies the power rule to the results of computing
        the derivative for the individual functions.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function derivative with respect to the parameters evaluated
                  at all input points, numpy array of shape ``(n_params, x.shape[0])``
        :rtype: ndarray
        """

        switch = self.f1.get_n_params(x)

        exp = self.f2.mean_f(x, params[switch:])
        nonzeroexp = True
        if np.allclose(exp, 0.):
            nonzeroexp = False

        deriv = np.zeros((self.get_n_params(x), x.shape[0]))

        if nonzeroexp:
            deriv[:switch] = (exp*self.f1.mean_f(x, params[:switch])**(exp - 1.)*
                              self.f1.mean_deriv(x, params[:switch]))

        # only evaluate if f2 has parameters, as f1 could be negative and taking the log will
        # raise an error even though this calculation is ultimately ignored in this case

        if not self.f2.get_n_params(x) == 0:

            deriv[switch:] = (np.log(self.f1.mean_f(x, params[:switch]))*
                              self.f1.mean_f(x, params[:switch])**exp*
                              self.f2.mean_deriv(x, params[switch:]))

        return deriv

    def mean_hessian(self, x, params):
        """
        Returns value of mean function Hessian wrt the parameters

        Method to compute the value of the mean function Hessian with respect to the
        parameters for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        Returns a numpy array of shape ``(n_params, n_params, x.shape[0])`` holding the value
        of the mean function second derivaties with respect to each parameter pair (first twp axes)
        for each input point (last axis).

        For ``MeanPower``, this method applies the power rule to the results of computing
        the Hessian for the individual functions.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function Hessian with respect to the parameters evaluated
                  at all input points, numpy array of shape ``(n_parmas, n_params, x.shape[0])``
        :rtype: ndarray
        """

        switch = self.f1.get_n_params(x)

        exp = self.f2.mean_f(x, params[switch:])
        nonzeroexp = True
        if np.allclose(exp, 0.):
            nonzeroexp = False
        nononeexp = True
        if np.allclose(exp, 1.):
            nononeexp = False

        hess = np.zeros((self.get_n_params(x), self.get_n_params(x), x.shape[0]))

        if nonzeroexp and nononeexp:

            hess[:switch, :switch] = (exp*self.f1.mean_f(x, params[:switch])**(exp - 1.)*
                                      self.f1.mean_hessian(x, params[:switch]) +
                                      exp*(exp - 1.)*self.f1.mean_f(x, params[:switch])**(exp - 2.)*
                                      self.f1.mean_deriv(x, params[:switch]))

        elif nonzeroexp:

            hess[:switch, :switch] = (exp*self.f1.mean_f(x, params[:switch])**(exp - 1.)*
                                      self.f1.mean_hessian(x, params[:switch]))


        if not self.f2.get_n_params(x) == 0:

            if nonzeroexp:
                hess[:switch, switch:, :] = (self.f1.mean_f(x, params[:switch])**(exp - 1.)*
                                             (exp*np.log(self.f1.mean_f(x, params[:switch])) + 1.)*
                                             self.f1.mean_deriv(x, params[:switch])[:,np.newaxis,:]*
                                             self.f2.mean_deriv(x, params[switch:])[np.newaxis,:,:])
                hess[switch:, :switch, :] = np.transpose(hess[:switch, switch:, :], (1, 0, 2))

            hess[switch:, switch:] = (self.f1.mean_f(x, params[:switch])**exp*
                                      (np.log(self.f1.mean_f(x, params[:switch]))**2*
                                       self.f2.mean_deriv(x, params[switch:])**2 +
                                       np.log(self.f1.mean_f(x, params[:switch]))*
                                       self.f2.mean_hessian(x, params[switch:])))

        return hess


    def mean_inputderiv(self, x, params):
        """
        Returns value of mean function derivative wrt the inputs

        Method to compute the value of the mean function derivative with respect to the
        inputs for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        Returns a numpy array of shape ``(x.shape[1], x.shape[0])`` holding the value of the mean
        function derivative with respect to each input (first axis) for each input point
        (second axis).

        For ``MeanPower``, this method applies the power rule to the results of computing
        the derivative for the individual functions.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function derivative with respect to the inputs evaluated
                  at all input points, numpy array of shape ``(x.shape[1], x.shape[0])``
        :rtype: ndarray
        """

        switch = self.f1.get_n_params(x)

        exp = self.f2.mean_f(x, params[switch:])
        nonzeroexp = True
        if np.allclose(exp, 0.):
            nonzeroexp = False

        inputderiv = np.zeros((x.shape[1], x.shape[0]))

        if nonzeroexp:
            inputderiv = (exp*self.f1.mean_f(x, params[:switch])**(exp - 1.)*
                          self.f1.mean_inputderiv(x, params[:switch]))

        return inputderiv

    def __str__(self):
        """
        Returns a string representation

        Return a formula-like representation of the Mean Function. Useful for confirming
        that a formula was correctly parsed.
        """

        return "{}^{}".format(self.f1, self.f2)

class MeanComposite(MeanBase):
    """
    Class representing the composition of two mean functions

    This derived class represents the composition of two mean functions, and does the necessary
    bookkeeping needed to compute the required function and derivatives. The code does
    not do any checks to confirm that it makes sense to compose these particular mean
    functions -- in particular, applying a ``Coefficient`` class to another function will
    simply wipe out the second function. This will not raise an error, but the code will not
    attempt to alert the user to this so it is up to the user to get it right.

    Because the Hessian computation requires mixed partials that are not normally implemented
    in the ``MeanBase`` class, the Hessian computation is not currently implemented.
    If you require Hessian computation for a composite mean function, you must implement
    it yourself.

    Note that since the outer function takes as its input the output of the second function,
    the outer function can only ever have an index of 0 due to the fixed output shape of
    a mean function. This will produce an error when attempting to evaluate the function
    or its derivatives, but will not cause an error when initializing a ``MeanComposite``
    object.

    :ivar f1: first ``MeanBase`` to be applied to the second
    :type f1: subclass of MeanBase
    :ivar f2: second ``MeanBase`` to be composed as the input to the first
    :type f2: subclass of MeanBase
    """
    def __init__(self, f1, f2):
        """
        Create a new instance of two composed mean functions

        Creates an instance of to composed mean functions. Inputs are the two functions
        to be composed (``f1(f2)``), which must be subclasses of the base ``MeanBase``
        class.

        :param f1: first ``MeanBase`` to be applied to the second
        :type f1: subclass of MeanBase
        :param f2: second ``MeanBase`` to be composed as the input to the first
        :type f2: subclass of MeanBase
        :returns: new ``MeanComposite`` instance
        :rtype: MeanComposite
        """
        if not issubclass(type(f1), MeanBase):
            raise TypeError("inputs to MeanComposite must be subclasses of MeanBase")
        if not issubclass(type(f2), MeanBase):
            raise TypeError("inputs to MeanComposite must be subclasses of MeanBase")

        self.f1 = f1
        self.f2 = f2

    def get_n_params(self, x):
        """
        Determine the number of parameters

        Returns the number of parameters for the mean function, which possibly depends on x.

        :param x: Input array
        :type x: ndarray
        :returns: number of parameters
        :rtype: int
        """
        return self.f1.get_n_params(np.zeros((x.shape[0], 1))) + self.f2.get_n_params(x)

    def mean_f(self, x, params):
        """
        Returns value of mean function

        Method to compute the value of the mean function for the inputs and parameters provided.
        Shapes of ``x`` and ``params`` must be consistent based on the return value of the
        ``get_n_params`` method. Returns a numpy array of shape ``(x.shape[0],)`` holding
        the value of the mean function for each input point.

        For ``MeanComposite``, this method applies the output of the second function as
        input to the first function.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function evaluated at all input points, numpy array of shape
                  ``(x.shape[0],)``
        :rtype: ndarray
        """

        switch = self.f1.get_n_params(x)

        return self.f1.mean_f(np.reshape(self.f2.mean_f(x, params[switch:]), (-1, 1)),
                              params[:switch])

    def mean_deriv(self, x, params):
        """
        Returns value of mean function derivative wrt the parameters

        Method to compute the value of the mean function derivative with respect to the
        parameters for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        Returns a numpy array of shape ``(n_params, x.shape[0])`` holding the value of the mean
        function derivative with respect to each parameter (first axis) for each input point
        (second axis).

        For ``MeanComposite``, this method applies the chain rule to the results of computing
        the derivative for the individual functions.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function derivative with respect to the parameters evaluated
                  at all input points, numpy array of shape ``(n_params, x.shape[0])``
        :rtype: ndarray
        """

        switch = self.f1.get_n_params(x)

        deriv = np.zeros((self.get_n_params(x), x.shape[0]))

        f2 = np.reshape(self.f2.mean_f(x, params[switch:]), (-1, 1))

        deriv[:switch] = self.f1.mean_deriv(f2, params[:switch])

        deriv[switch:] = (self.f1.mean_inputderiv(f2, params[:switch])*
                          self.f2.mean_deriv(x, params[switch:]))

        return deriv

    def mean_inputderiv(self, x, params):
        """
        Returns value of mean function derivative wrt the inputs

        Method to compute the value of the mean function derivative with respect to the
        inputs for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        Returns a numpy array of shape ``(x.shape[1], x.shape[0])`` holding the value of the mean
        function derivative with respect to each input (first axis) for each input point
        (second axis).

        For ``MeanComposite``, this method applies the chain rule to the results of computing
        the derivative for the individual functions.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function derivative with respect to the inputs evaluated
                  at all input points, numpy array of shape ``(x.shape[1], x.shape[0])``
        :rtype: ndarray
        """

        switch = self.f1.get_n_params(x)

        return (self.f1.mean_inputderiv(np.reshape(self.f2.mean_f(x, params[switch:]), (-1, 1)),
                                        params[:switch])*
                self.f2.mean_inputderiv(x, params[switch:]))

    def __str__(self):
        """
        Returns a string representation

        Return a formula-like representation of the Mean Function. Useful for confirming
        that a formula was correctly parsed.
        """

        return "{}({})".format(self.f1, self.f2)

class FixedMean(MeanBase):
    """
    Class representing a fixed mean function with no parameters

    Class representing a mean function with a fixed function (and optional derivative)
    and no fitting parameters. The user must provide these functions when initializing
    the instance.

    :ivar f: fixed mean function, must be callable and take a single argument (the inputs)
    :type f: function
    :ivar deriv: fixed derivative function (optional if no derivatives are needed), must
                   be callable and take a single argument (the inputs)
    :type deriv: function or None
    """
    def __init__(self, f, deriv=None):
        """
        Initialize a class instance representing a fixed mean function with no parameters

        Create a class instance representing a mean function with a fixed function
        (and optional derivative) and no fitting parameters. The user must provide these
        functions, though the derivative is optional. The code will check that the provided
        arguments are callable, but will not confirm that the inputs and outputs are the
        correct type/shape.

        :param f: fixed mean function, must be callable and take a single argument (the inputs)
        :type f: function
        :param deriv: fixed derivative function (optional if no derivatives are needed), must
                       be callable and take a single argument (the inputs)
        :type deriv: function or None
        :returns: new ``FixedMean`` instance
        :rtype: FixedMean
        """
        assert callable(f), "fixed mean function must be a callable function"
        if not deriv is None:
            assert callable(deriv), "mean function derivative must be a callable function"

        self.f = f
        self.deriv = deriv

    def get_n_params(self, x):
        """
        Determine the number of parameters

        Returns the number of parameters for the mean function, which possibly depends on x.
        For a ``FixedMean`` class, this is zero.

        :param x: Input array
        :type x: ndarray
        :returns: number of parameters
        :rtype: int
        """
        return 0

    def mean_f(self, x, params):
        """
        Returns value of mean function

        Method to compute the value of the mean function for the inputs and parameters provided.
        Shapes of ``x`` and ``params`` must be consistent based on the return value of the
        ``get_n_params`` method. For ``FixedMean`` classes, there are no parameters so the
        ``params`` argument should be an array of length zero. Returns a numpy array of shape
        ``(x.shape[0],)`` holding the value of the mean function for each input point.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input (zero in this case)
        :type params: ndarray
        :returns: Value of mean function evaluated at all input points, numpy array of shape
                  ``(x.shape[0],)``
        :rtype: ndarray
        """

        x, params = self._check_inputs(x, params)

        return self.f(x)

    def mean_deriv(self, x, params):
        """
        Returns value of mean function derivative wrt the parameters

        Method to compute the value of the mean function derivative with respect to the
        parameters for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        For ``FixedMean`` classes, there are no parameters so the ``params`` argument
        should be an array of length zero. Returns a numpy array of shape
        ``(n_params, x.shape[0])`` holding the value of the mean function derivative with
        respect to each parameter (first axis) for each input point (second axis). Since
        fixed means have no parameters, this will just be an array of zeros.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function derivative with respect to the parameters evaluated
                  at all input points, numpy array of shape ``(n_params, x.shape[0])``
        :rtype: ndarray
        """

        x, params = self._check_inputs(x, params)

        return np.zeros((self.get_n_params(x), x.shape[0]))

    def mean_hessian(self, x, params):
        """
        Returns value of mean function Hessian wrt the parameters

        Method to compute the value of the mean function Hessian with respect to the
        parameters for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        For ``FixedMean`` classes, there are no parameters so the ``params`` argument
        should be an array of length zero. Returns a numpy array of shape
        ``(n_params, n_params, x.shape[0])`` holding the value of the mean function
        second derivaties with respect to each parameter pair (first twp axes) for each
        input point (last axis). Since fixed means have no parameters, this will just
        be an array of zeros.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function Hessian with respect to the parameters evaluated
                  at all input points, numpy array of shape ``(n_parmas, n_params, x.shape[0])``
        :rtype: ndarray
        """

        x, params = self._check_inputs(x, params)

        return np.zeros((self.get_n_params(x), self.get_n_params(x), x.shape[0]))

    def mean_inputderiv(self, x, params):
        """
        Returns value of mean function derivative wrt the inputs

        Method to compute the value of the mean function derivative with respect to the
        inputs for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        For ``FixedMean`` classes, there are no parameters so the ``params`` argument
        should be an array of length zero. Returns a numpy array of shape
        ``(x.shape[1], x.shape[0])`` holding the value of the mean function derivative
        with respect to each input (first axis) for each input point (second axis).

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function derivative with respect to the inputs evaluated
                  at all input points, numpy array of shape ``(x.shape[1], x.shape[0])``
        :rtype: ndarray
        """

        x, params = self._check_inputs(x, params)

        if self.deriv is None:
            raise RuntimeError("Derivative function was not provided with this FixedMean")
        else:
            return self.deriv(x)

    def __str__(self):
        """
        Returns a string representation

        Return a formula-like representation of the Mean Function. Useful for confirming
        that a formula was correctly parsed.
        """

        return "f"

def fixed_f(x, index, f):
    """
    Dummy function to index into x and apply a function

    Usage is intended to be with a fixed mean function, where an index and specific mean
    function are meant to be bound using partial before setting it as the ``f`` attribute of
    ``FixedMean``

    :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
    :type x: ndarray
    :param index: integer index to be applied to the second axis of ``x``, used to select
                  a particular input variable. Must be non-negative and less than the
                  length of the second axis of the inputs.
    :type index: int
    :param f: fixed mean function, must be callable and take a single argument (the inputs)
    :type f: function
    :returns: Value of mean function evaluated at all input points, numpy array of shape
              ``(x.shape[0],)``
    :rtype: ndarray
    """
    assert callable(f), "fixed mean function must be callable"
    assert index >= 0, "provided index cannot be negative"
    assert x.ndim == 2, "x must have 2 dimensions"

    try:
        val = f(x[:,index])
    except IndexError:
        raise IndexError("provided mean function index is out of range")

    return val

def fixed_inputderiv(x, index, deriv):
    """
    Dummy function to index into x and apply a derivative function

    Usage is intended to be with a fixed mean function, where an index and specific derivative
    function are meant to be bound using partial before setting it as the ``deriv`` attribute of
    ``FixedMean``

    :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
    :type x: ndarray
    :param index: integer index to be applied to the second axis of ``x``, used to select
                  a particular input variable. Must be non-negative and less than the
                  length of the second axis of the inputs.
    :type index: int
    :param deriv: fixed derivative function, must be callable and take a single argument (the inputs)
    :type deriv: function
    :returns: Value of mean derivative evaluated at all input points, numpy array of shape
              ``(x.shape[1], x.shape[0])``
    :rtype: ndarray
    """
    assert callable(deriv), "fixed mean function derivative must be callable"
    assert index >= 0, "provided index cannot be negative"
    assert x.ndim == 2, "x must have 2 dimensions"

    try:
        out = np.zeros((x.shape[1], x.shape[0]))
        out[index, :] = deriv(np.transpose(x[:, index]))
    except IndexError:
        raise IndexError("provided mean function index is out of range")

    return out

def one(x):
    """
    Function to return an array of ones with the same shape as the input

    Function to return a numpy array of ones with the same shape as the input. Used in
    linear mean functions to evaluate derivatives.

    :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
    :type x: ndarray
    :returns: Numpy array of ones with the same shape as x
    :rtype: ndarray
    """
    return np.ones(x.shape)

def const_f(x, val):
    """
    Function to return an array of a constant value

    Function to return a numpy array of a constant value with the correct shape for a given
    input. Used in constant mean functions to evaluate the function.

    :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
    :type x: ndarray
    :param val: value of output, must be a float
    :type val: float
    :returns: Numpy array of ``val`` with shape ``(x.shape[0],)``
    :rtype: ndarray
    """
    assert x.ndim == 2, "x must have 2 dimensions"

    return np.broadcast_to(val, x.shape[0])

def const_deriv(x):
    """
    Function to return an array of zeros with the transposed shape of the inputs

    Function to return a numpy array of zeros with the shape that is transpose of the
    shape of the input. Used in constant mean functions to evaluate the derivative.

    :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
    :type x: ndarray
    :returns: Numpy array of zeros with shape ``(x.shape[1], x.shape[0])``
    :rtype: ndarray
    """
    assert x.ndim == 2, "x must have 2 dimensions"

    return np.zeros((x.shape[1], x.shape[0]))

class ConstantMean(FixedMean):
    """
    Class representing a constant fixed mean function

    Subclass of ``FixedMean`` where the function is a constant, with the value
    provided when ``ConstantMean`` is initialized. Uses utility functions to bind the
    value to the ``fixed_f`` function and sets that as the ``f`` attribute.

    :ivar f: fixed mean function, must be callable and take a single argument (the inputs)
    :type f: function
    :ivar deriv: fixed derivative function (optional if no derivatives are needed), must
                   be callable and take a single argument (the inputs)
    :type deriv: function
    """
    def __init__(self, val):
        """
        Initialize a new ConstantMean

        Create a new ``ConstantMean`` instance with the given constant value.

        :param val: Constant mean function value, must be a float or an integer
        :type val: float or int
        :returns: new ``ConstantMean`` instance
        :rtype: ConstantMean
        """
        if not isinstance(val, (float, int)):
            raise TypeError("val must be a float or an integer")
        self.f = partial(const_f, val=val)
        self.deriv = const_deriv

    def __str__(self):
        """
        Returns a string representation

        Return a formula-like representation of the Mean Function. Useful for confirming
        that a formula was correctly parsed.
        """
        val = signature(self.f).parameters['val'].default
        return "{}".format(val)

class LinearMean(FixedMean):
    """
    Class representing a linear fixed mean function

    Subclass of ``FixedMean`` where the function is a linear function. By default the
    function is linear in the first input dimension, though any non-negative integer index
    can be provided to control which input is used in the linear function. Uses utility
    functions to bind the correct function to the ``fixed_f`` function and sets that as
    the ``f`` attribute and similar with the ``fixed_deriv`` utility function and the
    ``deriv`` attribute.

    :ivar f: fixed mean function, must be callable and take a single argument (the inputs)
    :type f: function
    :ivar deriv: fixed derivative function, must be callable and take a single argument
                   (the inputs)
    :type deriv: function
    """
    def __init__(self, index=0):
        """
        Initialize a new LinearMean

        Create a new ``LinearMean`` instance with the given index value. This index is used
        to select the dimension of the input for evaluating the function.

        :param index: integer index to be applied to the second axis of ``x``, used to select
                      a particular input variable. Must be non-negative and less than the
                      length of the second axis of the inputs.
        :type index: int
        :returns: new ``LinearMean`` instance
        :rtype: LinearMean
        """
        self.f = partial(fixed_f, index=index, f=np.array)
        self.deriv = partial(fixed_inputderiv, index=index, deriv=one)

    def __str__(self):
        """
        Returns a string representation

        Return a formula-like representation of the Mean Function. Useful for confirming
        that a formula was correctly parsed.
        """

        index = signature(self.f).parameters["index"].default

        return "x[{}]".format(index)

class Coefficient(MeanBase):
    """
    Class representing a single fitting parameter in a mean function

    Class representing a mean function with single free fitting parameter. Does not require any
    internal state as the parameter value is stored/set externally through fitting routines.
    """
    def get_n_params(self, x):
        """
        Determine the number of parameters

        Returns the number of parameters for the mean function, which possibly depends on x.
        For a ``Coefficient`` class, this is always 1.

        :param x: Input array
        :type x: ndarray
        :returns: number of parameters
        :rtype: int
        """
        return 1

    def mean_f(self, x, params):
        """
        Returns value of mean function

        Method to compute the value of the mean function for the inputs and parameters provided.
        Shapes of ``x`` and ``params`` must be consistent based on the return value of the
        ``get_n_params`` method. For ``Coefficient`` classes, the inputs are ignored and the
        function returns the value of the parameter broadcasting it appropriately given the
        shape of the inputs. Thus, the ``params`` argument should always be an array of length
        one. Returns a numpy array of shape ``(x.shape[0],)`` holding the value of the
        parameter for each input point.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input (one in this case)
        :type params: ndarray
        :returns: Value of mean function evaluated at all input points, numpy array of shape
                  ``(x.shape[0],)``
        :rtype: ndarray
        """

        x, params = self._check_inputs(x, params)

        return np.broadcast_to(params, x.shape[0])

    def mean_deriv(self, x, params):
        """
        Returns value of mean function derivative wrt the parameters

        Method to compute the value of the mean function derivative with respect to the
        parameters for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        For ``Coefficient`` classes, the inputs are ignored and the derivative function
        returns one, broadcasting it appropriately given the shape of the inputs.
        Returns a numpy array of ones with shape ``(1, x.shape[0])`` holding the value
        of the mean function derivative with respect to each parameter (first axis) for
        each input point (second axis). Since coefficients are single parameters, this
        will just be an array of ones.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function derivative with respect to the parameters evaluated
                  at all input points, numpy array of shape ``(n_params, x.shape[0])``
        :rtype: ndarray
        """

        x, params = self._check_inputs(x, params)

        return np.ones((self.get_n_params(x), x.shape[0]))

    def mean_hessian(self, x, params):
        """
        Returns value of mean function Hessian wrt the parameters

        Method to compute the value of the mean function Hessian with respect to the
        parameters for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        For ``Coefficient`` classes, there is only a single parameter so the ``params``
        argument should be an array of length one. Returns a numpy array of shape
        ``(n_params, n_params, x.shape[0])`` holding the value of the mean function
        second derivaties with respect to each parameter pair (first twp axes) for each
        input point (last axis). Since coefficients depend linearly on a single parameter,
        this will always be an array of zeros.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function Hessian with respect to the parameters evaluated
                  at all input points, numpy array of shape ``(n_parmas, n_params, x.shape[0])``
        :rtype: ndarray
        """

        x, params = self._check_inputs(x, params)

        return np.zeros((self.get_n_params(x), self.get_n_params(x), x.shape[0]))

    def mean_inputderiv(self, x, params):
        """
        Returns value of mean function derivative wrt the inputs

        Method to compute the value of the mean function derivative with respect to the
        inputs for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        For ``Coefficient`` classes, there is a single parameters so the ``params`` argument
        should be an array of length one. Returns a numpy array of shape
        ``(x.shape[1], x.shape[0])`` holding the value of the mean function derivative
        with respect to each input (first axis) for each input point (second axis).
        Since coefficients do not depend on the inputs, this is just an array of zeros.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function derivative with respect to the inputs evaluated
                  at all input points, numpy array of shape ``(x.shape[1], x.shape[0])``
        :rtype: ndarray
        """

        x, params = self._check_inputs(x, params)

        return np.zeros((x.shape[1], x.shape[0]))

    def __str__(self):
        """
        Returns a string representation

        Return a formula-like representation of the Mean Function. Useful for confirming
        that a formula was correctly parsed.
        """
        return "c"

class PolynomialMean(MeanBase):
    """
    Polynomial mean function class

    A ``PolynomialMean`` is a mean function where every input dimension is fit to a fixed
    degree polynomial. The degree must be provided when creating the class instance. The
    number of parameters depends on the degree and the shape of the inputs, since a separate
    set of parameters are used for each input dimension.

    :ivar degree: Polynomial degree, must be a positive integer
    :type degree: int
    """
    def __init__(self, degree):
        """
        Create a new polynomial mean function instance

        A ``PolynomialMean`` is a mean function where every input dimension is fit to a fixed
        degree polynomial. The degree must be provided when creating the class instance. The
        number of parameters depends on the degree and the shape of the inputs, since a separate
        set of parameters are used for each input dimension. Must provide the degree when
        initializing.

        :param degree: Polynomial degree, must be a positive integer
        :type degree: int
        :returns: new ``PolynomialMean`` instance
        :rtype: PolynomialMean
        """
        assert int(degree) > 0, "degree must be a positive integer"

        self.degree = int(degree)

    def get_n_params(self, x):
        """
        Determine the number of parameters

        Returns the number of parameters for the mean function, which depends on x.

        :param x: Input array
        :type x: ndarray
        :returns: number of parameters
        :rtype: int
        """
        return x.shape[1]*self.degree + 1

    def mean_f(self, x, params):
        """
        Returns value of mean function

        Method to compute the value of the mean function for the inputs and parameters provided.
        Shapes of ``x`` and ``params`` must be consistent based on the return value of the
        ``get_n_params`` method. Returns a numpy array of shape ``(x.shape[0],)`` holding
        the value of the mean function for each input point.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function evaluated at all input points, numpy array of shape
                  ``(x.shape[0],)``
        :rtype: ndarray
        """
        x, params = self._check_inputs(x, params)

        n_params = self.get_n_params(x)

        indices = np.arange(0, n_params - 1) % x.shape[1]
        expon = np.arange(0, n_params - 1) // x.shape[1] + 1

        output = params[0] + np.sum(params[1:]*x[:, indices]**expon, axis = 1)

        return output

    def mean_deriv(self, x, params):
        """
        Returns value of mean function derivative wrt the parameters

        Method to compute the value of the mean function derivative with respect to the
        parameters for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        Returns a numpy array of shape ``(n_params, x.shape[0])`` holding the value of the mean
        function derivative with respect to each parameter (first axis) for each input point
        (second axis).

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function derivative with respect to the parameters evaluated
                  at all input points, numpy array of shape ``(n_params, x.shape[0])``
        :rtype: ndarray
        """

        x, params = self._check_inputs(x, params)

        n_params = self.get_n_params(x)

        deriv = np.zeros((n_params, x.shape[0]))
        deriv[0] = np.ones(x.shape[0])

        indices = np.arange(0, n_params - 1) % x.shape[1]
        expon = np.arange(0, n_params - 1) // x.shape[1] + 1

        deriv[1:,:] = np.transpose(x[:, indices]**expon)

        return deriv

    def mean_hessian(self, x, params):
        """
        Returns value of mean function Hessian wrt the parameters

        Method to compute the value of the mean function Hessian with respect to the
        parameters for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        Returns a numpy array of shape ``(n_params, n_params, x.shape[0])`` holding the
        value of the mean function second derivaties with respect to each parameter pair
        (first twp axes) for each input point (last axis). Since polynomial means depend
        linearly on all input parameters, this will always be an array of zeros.

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function Hessian with respect to the parameters evaluated
                  at all input points, numpy array of shape ``(n_parmas, n_params, x.shape[0])``
        :rtype: ndarray
        """

        x, params = self._check_inputs(x, params)

        n_params = self.get_n_params(x)

        hess = np.zeros((n_params, n_params, x.shape[0]))

        return hess

    def mean_inputderiv(self, x, params):
        """
        Returns value of mean function derivative wrt the inputs

        Method to compute the value of the mean function derivative with respect to the
        inputs for the inputs and parameters provided. Shapes of ``x`` and ``params``
        must be consistent based on the return value of the ``get_n_params`` method.
        Returns a numpy array of shape ``(x.shape[1], x.shape[0])`` holding the value of the mean
        function derivative with respect to each input (first axis) for each input point
        (second axis).

        :param x: Inputs, must be a 1D or 2D numpy array (if 1D a second dimension will be added)
        :type x: ndarray
        :param params: Parameters, must be a 1D numpy array (of more than 1D will be flattened)
                       and have the same length as the number of parameters required for the
                       provided input
        :type params: ndarray
        :returns: Value of mean function derivative with respect to the inputs evaluated
                  at all input points, numpy array of shape ``(x.shape[1], x.shape[0])``
        :rtype: ndarray
        """

        x, params = self._check_inputs(x, params)

        expon = np.reshape(np.arange(0, x.shape[0]*x.shape[1]*self.degree)//x.shape[1]//self.degree,
                           (self.degree, x.shape[0]*x.shape[1]))
        x_indices = np.reshape(np.arange(0, x.shape[0]*x.shape[1]*self.degree) % (x.shape[0]*x.shape[1]),
                               (self.degree, x.shape[0]*x.shape[1]))
        param_indices = np.reshape(np.arange(0, x.shape[0]*x.shape[1]*self.degree) % x.shape[1],
                                   (self.degree, x.shape[0]*x.shape[1])) + expon*x.shape[1]
        param_indices = np.reshape(param_indices, (self.degree, x.shape[0]*x.shape[1]))

        output = np.sum((expon + 1.)*params[1:][param_indices]*x.flatten()[x_indices]**expon, axis=0)

        return np.transpose(np.reshape(output, (x.shape[0], x.shape[1])))

    def __str__(self):
        """
        Returns a string representation

        Return a string representation of the polynomial mean
        """

        return "Polynomial mean of degree {}".format(self.degree)
