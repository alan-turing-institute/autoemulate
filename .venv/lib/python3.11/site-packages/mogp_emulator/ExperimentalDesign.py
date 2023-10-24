import numpy as np
import scipy.stats
from scipy.spatial.distance import pdist
from inspect import signature

class ExperimentalDesign(object):
    """
    Base class representing a generic one-shot design of experiments with uncorrelated parameters
    
    This class provides the base implementation for a class for designing experiments to sample
    the parameter space of a complex model. The parameter space can be specified in a variety
    of ways, but essentially the user must provide a Probability Point Function (PPF, or inverse 
    of the Cumulative Distribution Function) for each input parameter. Each PPF function takes a 
    single numeric input and maps from the interval :math:`[0,1]` to the desired parameter
    distribution value for a given parameter, and each parameter has a separate function describing 
    its distribution. Note that this makes the assumption of no correlations between any of the
    parameter values (a future version may implement an experimental design where there are such
    parameter correlations). Once the design is initialized, a desired number of samples can be
    drawn from the design, returning an array holding the desired number of samples from the
    parameter space.
    
    Internally, the class holds the set of PPFs for all of the parameter values, and samples are
    drawn by calling the ``sample`` method. To draw the samples, a specific method ``_draw_samples``
    must be defined that generates a series of points in the :math:`[0,1]^n` hypercube, where
    :math:`n` is the number of paramters. This set of samples from the hypercube is then mapped to
    the parameter space using the given PPF functions. Thus, defining a new design protocol only
    requires defining a new ``_draw_samples`` method and redefining the ``__init__`` method to set
    the internal ``method`` attribute. By default, no ``_draw_samples`` method is defined, so the
    base ``ExperimentalDesign`` class is only intended to be used to define new protocols (trying 
    to sample from an ``ExperimentalDesign`` instance will return a ``NotImplementedError``).
    """
    def __init__(self, *args):
        """
        Create a new instance of an experimental design
        
        Creates a new instance of a design of experiments, which draws samples from the parameter
        space of a complex model. It is often used to generate data for a Gaussian Process emulator
        to fit the outputs of the complex model. This is a base class that does not implement the
        method for sampling from the distribution; to use an experimental design in practice you
        should use one of the derived classes provided or create your own.
        
        The experimental design can be initialized in several ways depending on the arguments
        provided, ranging from the simplest to the most complicated.
        
        1. Provide an integer ``n`` indicating the number of input parameters. If this is used to
           create an instance, it is assumed that all parameters are unformly distributed over
           the :math:`n`-dimensional hypercube.
        2. Provide an integer ``n`` and a tuple ``(a, b)`` of length 2 containing two numeric values
           (where :math:`a < b`). In this case, all parameters are assumed to be uniformly distributed
           over the interval :math:`[a,b]`.
        3. Provide an integer ``n`` and a function that takes a single numeric input in the interval
           :math:`[0,1]` and maps it to the parameter space. In this case, all parameters are assumed
           to follow the provided Probability Point Function.
        4. Provide a list of tuples of length 2 containing numeric values (as above, the first number
           must smaller than the second number). The design then assumes that the number of parameters
           is the length of the list, and each parameter follows a uniform distribution with the bounds
           given by the respective tuple in the given list.
        5. Provide a list of functions taking a single input (as above, each function must map the
           interval :math:`[0,1]` to the parameter space). The number of parameters in the design is the
           length of the list, and the given PPF functions define the parameter space for each input.
        
        More concretely, if one input parameter is given, you may initilize the class in any of the
        following ways:
        
        :param n_parameters: Integer specifying the number of parameters (must be positive). The
                             design will sample each parameter over the interval :math:`[0,1]`
        :type n_parameters: int
        
        or
        
        :param bounds_list: List of tuples containing two numeric values, each of which has the
                           smaller number first. Each parameter then takes a uniform distribution
                           with bounds given by each tuple.
        :type bounds_list: list
        
        or
        
        :param ppf_list: List of functions or other callable, each of which accepts one argument
                         and maps the interval :math:`[0,1]` to the parameter space. Each parameter
                         follows the distribution given by the respective PPF function.
        :type ppf_list: list
        
        and if two input parameters are given:
        
        :param n_parameters: Integer specifying the number of parameters (must be positive). The
                             design will sample each parameter over the interval :math:`[0,1]`
        :type n_parameters: int
        :param bounds: Tuple or other iterable containing two numeric values, where the smaller
                       number must come first. Each parameter then takes a uniform distribution
                       with bounds given by the numbers provided in the tuple.
        :type bounds: tuple
        
        or
        
        :param n_parameters: Integer specifying the number of parameters (must be positive). The
                             design will sample each parameter over the interval :math:`[0,1]`
        :type n_parameters: int
        :param ppf: Function or other callable, which accepts one argument and maps the interval
                    :math:`[0,1]` to the parameter space. Each parameter follows the distribution
                    given by the PPF function.
        :type ppf: function
        
        The ``scipy.stats`` package provides implementations of a wide range of distributions, with
        pre-defined PPF functions. See the Scipy user manual for more details. Note that in order to
        get a compatible PPF function that only takes a single input, you will need to set any parameters
        needed to define the distribution.
        
        Internally, the class defines any PPF functions based on the input data and collects all of
        the PPF functions in a list. The class also contains information on the method used to draw
        samples from the design.
        
        To create a usable implementation based on this class, the user must define the method
        ``_draw_samples``, which takes a positive integer input ``n_samples`` and draws ``n_samples``
        from the :math:`[0,1]^n` hypercube, where :math:`n` is the number of parameters. The user must
        also modify the ``method`` attribute of the design in order to have the ``__str__`` method work
        correctly. All other functionality should not require any changes from the base class.
        """
        
        if len(args) == 1:
            try:
                n_parameters = int(args[0])
                bounds = None
            except TypeError:
                try:
                    n_parameters = len(list(args[0]))
                    bounds = list(args[0])
                except TypeError:
                    raise TypeError("bad input type for ExperimentalDesign")
        elif len(args) == 2:
            try:
                n_parameters = int(args[0])
            except TypeError:
                raise TypeError("bad input type for ExperimentalDesign")
                
            if callable(args[1]):
                bounds = args[1]
            else:
                try:
                    bounds = list(args[1])
                    try:
                        if (len(bounds) == 2 and isinstance(float(bounds[0]), float) and isinstance(float(bounds[1]), float)):
                            if float(bounds[1]) <= float(bounds[0]):
                                raise ValueError("bad value for parameter bounds in ExperimentalDesign")
                            bounds = (float(bounds[0]), float(bounds[1]))
                    except TypeError:
                        pass
                except TypeError:
                    raise TypeError("bad input type for ExperimentalDesign")
        else:
            raise ValueError("bad inputs for ExperimentalDesign")
                
        if n_parameters <= 0:
            raise ValueError("number of parameters must be positive in Experimental Design")

        self.n_parameters = n_parameters
        
        if bounds is None:
            self.distributions = [scipy.stats.uniform(loc = 0., scale = 1.).ppf]*n_parameters
        elif isinstance(bounds, tuple):
            self.distributions = [scipy.stats.uniform(loc = bounds[0], scale = bounds[1]-bounds[0]).ppf]*n_parameters
        elif callable(bounds):
            if len(signature(bounds).parameters) == 1:
                self.distributions = [bounds]*n_parameters
            else:
                raise ValueError("PPF distribution provided must accept a single argument")
        else: # bounds is a list
            if not len(bounds) == n_parameters:
                raise ValueError("list of parameter distributions must have the same length")
            self.distributions = []
            for item in bounds:
                if callable(item):
                    if len(signature(item).parameters) == 1:
                        self.distributions.append(item)
                    else:
                        raise ValueError("PPF distribution provided must accept a single argument")
                else:
                    try:
                        if (len(item) == 2 and isinstance(float(item[0]), float) and isinstance(float(item[1]), float)):
                            if float(item[1]) <= float(item[0]):
                                raise ValueError("bad value for parameter bounds in ExperimentalDesign")
                            self.distributions.append(scipy.stats.uniform(loc = float(item[0]), 
                                                                          scale = float(item[1])- float(item[0])).ppf)
                        else:
                            raise ValueError("bounds for each parameter must be a tuple of two floats")
                    except TypeError:
                        raise TypeError("bounds for each parameter must be a tuple of two floats")
            
        
    def get_n_parameters(self):
        """
        Returns number of parameters in the experimental design
        
        This method returns the number of parameters in the experimental design. This is set when
        initializing the object, an cannot be modified.
        
        :returns: Number of parameters in the experimental design.
        :rtype: int
        """
        return self.n_parameters
        
    def get_method(self):
        """
        Returns the method used to draw samples from the design
        
        This method returns the method used to draw samples from the experimental design. The base
        class does not implement a method, so if you try to call this on the base class the code
        will raise a ``NotImplementedError``. When deriving new designs from the base class,
        the method should be set when calling the ``__init__`` method.
        
        :returns: Method used to draw samples from the design.
        :rtype: str
        """
        try:
            return self.method
        except AttributeError:
            raise NotImplementedError("base class of ExperimentalDesign does not implement a method")
    
    def _draw_samples(self, n_samples):
        """
        Low level method for drawing random samples from a design
        
        This method implements the low level method for sampling from a design. All samples drawn
        using this method will lie in the :math:`[0,1]^n` hypercube, where :math:`n` is the number
        of parameters in the design. Once these are drawn, the actual parameter values are obtained
        by transforming the samples using the provided PPF functions.
        
        This method is not implemented for the base class. Derived classes will only need to
        implement this method to get a working design, as all other pieces of code should be mostly
        identical.
        
        :param n_samples: Number of samples to be drawn from the design (must be a positive integer)
        :type n_samples: int
        :returns: Samples drawn from the :math:`[0,1]^n` hypercube as a numpy array with shape
                  ``(n_samples, n_parameters)``
        :rtype: ndarray
        """
        raise NotImplementedError
    
    def sample(self, n_samples, **kwargs):
        """
        Draw parameter samples from the experimental design
        
        This method implements drawing parameter samples from the experimental design. The method does
        this by calling the ``_draw_samples`` method to obtain samples from the :math:`[0,1]^n` hypercube,
        where :math:`n` is the number of parameters. The ``sample`` method then transforms these samples
        drawn from the low level method to the actual parameter values using the PPF functions provided
        when initilizing the object. Note that this method also checks that all parameter values are
        finite; if any ``NaN`` values are returned, an error will be raised.
        
        Note that by implementing the sampling in this way, modifications to the method to draw samples
        using a different protocol only needs to change the ``_draw_samples`` method. This makes it
        simpler to define new designs, as only a single method needs to be altered.
        
        Also accepts a ``kwargs`` argument to allow other derived classes to implement additional
        keyword arguments.
        
        :param n_samples: Number of samples to be drawn from the design (must be a positive integer)
        :type n_samples: int
        :returns: Samples drawn from the design parameter space as a numpy array with shape
                  ``(n_samples, n_parameters)``
        :rtype: ndarray
        """
        
        n_samples = int(n_samples)
        assert n_samples > 0, "number of samples must be positive"
        
        sample_values = np.zeros((n_samples, self.get_n_parameters()))
        random_draws = self._draw_samples(n_samples, **kwargs)
        
        assert np.all(random_draws >= 0.) and np.all(random_draws <= 1.), "error in generating random samples"
        
        for (dist, index) in zip(self.distributions, range(self.get_n_parameters())):
            for sample_index in range(n_samples):
                sample_values[sample_index, index] = dist(random_draws[sample_index,index])
        
        assert np.all(np.isfinite(sample_values)), "error due to non-finite values of parameters"
        
        return sample_values
        
    def __str__(self):
        """
        Returns a string representation of the ExperimentalDesign object
        
        This method returns a string representation of the Experimental Design object. If a specific
        method is set for a derived class from the base class, the string will include this method
        in the string output. Otherwise, a generic string will be returned.
        
        :returns: String representation of the object
        :rtype: str
        """
        try:
            method = self.get_method()+" "
        except NotImplementedError:
            method = ""
        return method+"Experimental Design with "+str(self.get_n_parameters())+" parameters"
        
class MonteCarloDesign(ExperimentalDesign):
    """
    Class representing a one-shot design of experiments with uncorrelated parameters using Monte
    Carlo Sampling
    
    This class provides an implementation for a class for designing experiments to sample
    the parameter space of a complex model using random Monte Carlo sampling. The parameter space can
    be specified in a variety of ways, but essentially the user must provide a Probability Point
    Function (PPF, or inverse of the Cumulative Distribution Function) for each input parameter. Each
    PPF function takes a single numeric input and maps from the interval :math:`[0,1]` to the desired
    parameter distribution value for a given parameter, and each parameter has a separate function
    describing its distribution. Note that this makes the assumption of no correlations between any of
    the parameter values (a future version may implement an experimental design where there are such
    parameter correlations). Once the design is initialized, a desired number of samples can be
    drawn from the design, returning an array holding the desired number of samples from the
    parameter space.
    
    Internally, the class holds the set of PPFs for all of the parameter values, and samples are
    drawn by calling the ``sample`` method. To draw the samples, the ``_draw_samples`` is used
    to generate a series of points in the :math:`[0,1]^n` hypercube using random Monte Carlo sampling,
    where :math:`n` is the number of paramters. This set of samples from the hypercube is then mapped to
    the parameter space using the given PPF functions.
    
    This design does not attempt to uniformly sample the space, but rather just makes random draws from
    the parameter distributions. For this reason, small designs using this method may not sample the
    parameter space very efficiently. However, generating a large number of samples using this protocol
    can be done very efficiently, as drawing a sample only requires generating a series of pseduorandom
    numbers. 
    """
    def __init__(self, *args):
        """
        Create a new instance of a Monte Carlo experimental design
    
        Creates a new instance of a Monte Carlo design of experiments, which draws samples randomly 
        from the parameter space of a complex model. It can be used to generate data for a Gaussian
        Process emulator to fit the outputs of the complex model. Because the samples are drawn
        randomly, small designs may not sample the space very well, but the samples can be drawn
        quickly as they are entirely random.
    
        The experimental design can be initialized in several ways depending on the arguments
        provided, ranging from the simplest to the most complicated.
    
        1. Provide an integer ``n`` indicating the number of input parameters. If this is used to
           create an instance, it is assumed that all parameters are unformly distributed over
           the :math:`n`-dimensional hypercube.
        2. Provide an integer ``n`` and a tuple ``(a, b)`` of length 2 containing two numeric values
           (where :math:`a < b`). In this case, all parameters are assumed to be uniformly distributed
           over the interval :math:`[a,b]`.
        3. Provide an integer ``n`` and a function that takes a single numeric input in the interval
           :math:`[0,1]` and maps it to the parameter space. In this case, all parameters are assumed
           to follow the provided Probability Point Function.
        4. Provide a list of tuples of length 2 containing numeric values (as above, the first number
           must smaller than the second number). The design then assumes that the number of parameters
           is the length of the list, and each parameter follows a uniform distribution with the bounds
           given by the respective tuple in the given list.
        5. Provide a list of functions taking a single input (as above, each function must map the
           interval :math:`[0,1]` to the parameter space). The number of parameters in the design is the
           length of the list, and the given PPF functions define the parameter space for each input.
    
        More concretely, if one input parameter is given, you may initilize the class in any of the
        following ways:
    
        :param n_parameters: Integer specifying the number of parameters (must be positive). The
                             design will sample each parameter over the interval :math:`[0,1]`
        :type n_parameters: int
    
        or
    
        :param bounds_list: List of tuples containing two numeric values, each of which has the
                           smaller number first. Each parameter then takes a uniform distribution
                           with bounds given by each tuple.
        :type bounds_list: list
    
        or
    
        :param ppf_list: List of functions or other callable, each of which accepts one argument
                         and maps the interval :math:`[0,1]` to the parameter space. Each parameter
                         follows the distribution given by the respective PPF function.
        :type ppf_list: list
    
        and if two input parameters are given:
    
        :param n_parameters: Integer specifying the number of parameters (must be positive). The
                             design will sample each parameter over the interval :math:`[0,1]`
        :type n_parameters: int
        :param bounds: Tuple or other iterable containing two numeric values, where the smaller
                       number must come first. Each parameter then takes a uniform distribution
                       with bounds given by the numbers provided in the tuple.
        :type bounds: tuple
    
        or
    
        :param n_parameters: Integer specifying the number of parameters (must be positive). The
                             design will sample each parameter over the interval :math:`[0,1]`
        :type n_parameters: int
        :param ppf: Function or other callable, which accepts one argument and maps the interval
                    :math:`[0,1]` to the parameter space. Each parameter follows the distribution
                    given by the PPF function.
        :type ppf: function
    
        The ``scipy.stats`` package provides implementations of a wide range of distributions, with
        pre-defined PPF functions. See the Scipy user manual for more details. Note that in order to
        get a compatible PPF function that only takes a single input, you will need to set any parameters
        needed to define the distribution.
    
        Internally, the class defines any PPF functions based on the input data and collects all of
        the PPF functions in a list. The class also contains information on the method used to draw
        samples from the design.
        """
        
        self.method = "Monte Carlo"
        super().__init__(*args)
        
    def _draw_samples(self, n_samples, **kwargs):
        """
        Low level method for drawing random samples from a Monte Carlo design
        
        This method implements the low level method for sampling from a Monte Carlo design. All samples
        drawn using this method are random points in the :math:`[0,1]^n` hypercube, where :math:`n` is
        the number of parameters in the design. Once these are drawn, the actual parameter values are
        obtained by transforming the samples using the provided PPF functions in the ``sample`` method.
        
        :param n_samples: Number of samples to be drawn from the design (must be a positive integer)
        :type n_samples: int
        :returns: Random Monte Carlo samples drawn from the :math:`[0,1]^n` hypercube as a numpy
                  array with shape ``(n_samples, n_parameters)``
        :rtype: ndarray
        """
        
        n_samples = int(n_samples)
        assert n_samples > 0, "number of samples must be positive"
        
        return np.random.random((n_samples, self.get_n_parameters()))
        
        
class LatinHypercubeDesign(ExperimentalDesign):
    """
    Class representing a one-shot design of experiments with uncorrelated parameters using Latin
    Hypercube Sampling
    
    This class provides an implementation for a class for designing experiments to sample
    the parameter space of a complex model using Latin Hypercube sampling. The parameter space can
    be specified in a variety of ways, but essentially the user must provide a Probability Point
    Function (PPF, or inverse of the Cumulative Distribution Function) for each input parameter. Each
    PPF function takes a single numeric input and maps from the interval :math:`[0,1]` to the desired
    parameter distribution value for a given parameter, and each parameter has a separate function
    describing its distribution. Note that this makes the assumption of no correlations between any of
    the parameter values (a future version may implement an experimental design where there are such
    parameter correlations). Once the design is initialized, a desired number of samples can be
    drawn from the design, returning an array holding the desired number of samples from the
    parameter space.
    
    Internally, the class holds the set of PPFs for all of the parameter values, and samples are
    drawn by calling the ``sample`` method. To draw the samples, the ``_draw_samples`` is used
    to generate a series of points in the :math:`[0,1]^n` hypercube using Latin Hypercube sampling,
    where :math:`n` is the number of paramters. This set of samples from the Latin Hypercube is then
    mapped to the parameter space using the given PPF functions.
    
    Unlike Monte Carlo sampling, Latin Hypercube designs attempt to sample more uniformly from the
    parameter space. Latin Hypercube sampling ensures that each sample is drawn from a different
    part of the space for each parameter. For example, if four samples are drawn, then for each
    parameter, one sample is guaranteed to be drawn from each quartile of the distribution. This
    ensures a more uniform sampling when compared on Monte Carlo sampling, but requires slightly
    more computation to generate the samples. Note however, that for very large numbers of parameters,
    Latin Hypercubes still may not sample very efficiently. This is due to the fact that the size of
    the parameter space grows exponentially with the number of dimensions, so a fixed number of
    samples will sample the space more poorly as the number of parameters increases.
    """
    def __init__(self, *args):
        """
        Create a new instance of a Latin Hypercube experimental design
    
        Creates a new instance of a Latin Hypercube design of experiments, which draws samples 
        from the parameter space of a complex model in a more uniform fashion when compared to
        random Monte Carlo sampling. It can be used to generate data for a Gaussian
        Process emulator to fit the outputs of the complex model. Because the samples are drawn
        more uniformly than in Monte Carlo sampling, these designs may perform better in high
        dimensional parameter spaces.
    
        The experimental design can be initialized in several ways depending on the arguments
        provided, ranging from the simplest to the most complicated.
    
        1. Provide an integer ``n`` indicating the number of input parameters. If this is used to
           create an instance, it is assumed that all parameters are unformly distributed over
           the :math:`n`-dimensional hypercube.
        2. Provide an integer ``n`` and a tuple ``(a, b)`` of length 2 containing two numeric values
           (where :math:`a < b`). In this case, all parameters are assumed to be uniformly distributed
           over the interval :math:`[a,b]`.
        3. Provide an integer ``n`` and a function that takes a single numeric input in the interval
           :math:`[0,1]` and maps it to the parameter space. In this case, all parameters are assumed
           to follow the provided Probability Point Function.
        4. Provide a list of tuples of length 2 containing numeric values (as above, the first number
           must smaller than the second number). The design then assumes that the number of parameters
           is the length of the list, and each parameter follows a uniform distribution with the bounds
           given by the respective tuple in the given list.
        5. Provide a list of functions taking a single input (as above, each function must map the
           interval :math:`[0,1]` to the parameter space). The number of parameters in the design is the
           length of the list, and the given PPF functions define the parameter space for each input.
    
        More concretely, if one input parameter is given, you may initilize the class in any of the
        following ways:
    
        :param n_parameters: Integer specifying the number of parameters (must be positive). The
                             design will sample each parameter over the interval :math:`[0,1]`
        :type n_parameters: int
    
        or
    
        :param bounds_list: List of tuples containing two numeric values, each of which has the
                           smaller number first. Each parameter then takes a uniform distribution
                           with bounds given by each tuple.
        :type bounds_list: list
    
        or
    
        :param ppf_list: List of functions or other callable, each of which accepts one argument
                         and maps the interval :math:`[0,1]` to the parameter space. Each parameter
                         follows the distribution given by the respective PPF function.
        :type ppf_list: list
    
        and if two input parameters are given:
    
        :param n_parameters: Integer specifying the number of parameters (must be positive). The
                             design will sample each parameter over the interval :math:`[0,1]`
        :type n_parameters: int
        :param bounds: Tuple or other iterable containing two numeric values, where the smaller
                       number must come first. Each parameter then takes a uniform distribution
                       with bounds given by the numbers provided in the tuple.
        :type bounds: tuple
    
        or
    
        :param n_parameters: Integer specifying the number of parameters (must be positive). The
                             design will sample each parameter over the interval :math:`[0,1]`
        :type n_parameters: int
        :param ppf: Function or other callable, which accepts one argument and maps the interval
                    :math:`[0,1]` to the parameter space. Each parameter follows the distribution
                    given by the PPF function.
        :type ppf: function
    
        The ``scipy.stats`` package provides implementations of a wide range of distributions, with
        pre-defined PPF functions. See the Scipy user manual for more details. Note that in order to
        get a compatible PPF function that only takes a single input, you will need to set any parameters
        needed to define the distribution.
    
        Internally, the class defines any PPF functions based on the input data and collects all of
        the PPF functions in a list. The class also contains information on the method used to draw
        samples from the design.
        """
        
        self.method = "Latin Hypercube"
        super().__init__(*args)
    
    def _draw_samples(self, n_samples, **kwargs):
        """
        Low level method for drawing random samples from a Latin Hypercube design
        
        This method implements the low level method for sampling from a Latin Hypercube design. All
        samples drawn using this method are random points in the :math:`[0,1]^n` hypercube, where
        :math:`n` is the number of parameters in the design. Once these are drawn, the actual parameter
        values are obtained by transforming the samples using the provided PPF functions in the ``sample``
        method. A Latin Hypercube design guarantees that each sample is drawn from a unique portion
        of the parameter space. For example, if 10 samples are drawn from a design, then for each
        parameter the ten samples will each come from a different decile of the distribution.
        
        :param n_samples: Number of samples to be drawn from the design (must be a positive integer)
        :type n_samples: int
        :returns: Random samples drawn from the :math:`[0,1]^n` hypercube as a numpy
                  array with shape ``(n_samples, n_parameters)``
        :rtype: ndarray
        """
        
        n_samples = int(n_samples)
        assert n_samples > 0, "number of samples must be positive"
        
        n_parameters = self.get_n_parameters()

        random_samples = np.reshape(np.tile(np.arange(n_samples, dtype = np.float64)/float(n_samples), n_parameters),
                                            (n_parameters, n_samples))
                    
        for row in random_samples:
            np.random.shuffle(row)
            
        random_samples =  np.transpose(random_samples) + np.random.random((n_samples, n_parameters))/float(n_samples)
        
        assert np.all(random_samples >= 0.) and np.all(random_samples <= 1.), "error in generating latin hypercube samples"
        
        return random_samples
        
class MaxiMinLHC(LatinHypercubeDesign):
    def __init__(self, *args):
        """
        Class representing a one-shot design of experiments with uncorrelated parameters using
        MaxiMin Latin Hypercube Sampling

        This class provides an implementation for a class for designing experiments to sample
        the parameter space of a complex model using MaxiMin Latin Hypercube sampling. MaxiMin
        LHCs repeatedly draw samples from the base LHC design, keeping the realization that
        maximizes the minimum pairwise distance between all design points. Because of this,
        MaxiMin designs tend to spread their samples closer to the edge of the parameter
        space and in many cases result in more accurate sampling than a single LHC draw.

        The parameter space can be specified in a variety of ways, but essentially the user must
        provide a Probability Point Function (PPF, or inverse of the Cumulative Distribution Function)
        for each input parameter. Each PPF function takes a single numeric input and maps from
        the interval :math:`[0,1]` to the desired parameter distribution value for a given parameter,
        and each parameter has a separate function describing its distribution. Note that this makes
        the assumption of no correlations between any of the parameter values (a future version may
        implement an experimental design where there are such parameter correlations). Once the
        design is initialized, a desired number of samples can be drawn from the design, returning
        an array holding the desired number of samples from the parameter space.

        Internally, the class holds the set of PPFs for all of the parameter values, and samples are
        drawn by calling the ``sample`` method. To draw the samples, the ``_draw_samples`` is used
        to generate a series of points in the :math:`[0,1]^n` hypercube using MaxiMin Latin Hypercube
        sampling, where :math:`n` is the number of paramters. This set of samples from the Latin
        Hypercube is then mapped to the parameter space using the given PPF functions.

        Unlike Monte Carlo sampling, Latin Hypercube designs attempt to sample more uniformly from the
        parameter space. Latin Hypercube sampling ensures that each sample is drawn from a different
        part of the space for each parameter. For example, if four samples are drawn, then for each
        parameter, one sample is guaranteed to be drawn from each quartile of the distribution. This
        ensures a more uniform sampling when compared on Monte Carlo sampling, but requires slightly
        more computation to generate the samples. Note however, that for very large numbers of parameters,
        Latin Hypercubes still may not sample very efficiently. This is due to the fact that the size of
        the parameter space grows exponentially with the number of dimensions, so a fixed number of
        samples will sample the space more poorly as the number of parameters increases.
        """
        self.method = "MaxiMinLHC"
        super().__init__(*args)

    def _draw_samples(self, n_samples, n_tries=1000, **kwargs):
        """
        Sampling method for MaxiMin LHCs

        Iterates over multiple LHCs and return the one that maximizes the
        minimum distance between pairs of points. The number of iterations
        over which to look for this MaxiMin critera can be specified by
        ``n_tries``.

        Distances are computed by ``scipy.spatial.distance.pdist``. Any
        additional ``kwargs`` passed here will be sent on to the ``pdist``
        function.

        :param n_samples: Number of samples to be drawn from the design (must be a positive integer)
        :type n_samples: int
        :param n_tries: Number of LHC realizations to use in maximizing
                        the MaxiMin criteria
        :type n_tries: int
        :param **kwargs: Keyword arguments to be passed to the
                         ``scipy.spatial.distance.pdist`` function.
        :returns: Random Monte Carlo samples drawn from the :math:`[0,1]^n` hypercube as a numpy
                  array with shape ``(n_samples, n_parameters)``
        :rtype: ndarray
        """

        n_samples = int(n_samples)
        assert n_samples > 0, "number of samples must be positive"
        assert n_tries > 0, "n_tries must be a positive integer"
        n_tries = int(n_tries)

        n_parameters = self.get_n_parameters()

        best_samples = np.empty((n_samples, n_parameters))
        max_dist = -np.inf

        for i in range(n_tries):
            random_samples = super()._draw_samples(n_samples)
            min_dist = np.min(pdist(random_samples, **kwargs))
            if min_dist > max_dist:
                max_dist = min_dist
                best_samples = random_samples

        assert np.all(best_samples >= 0.0) and np.all(
            best_samples <= 1.0
        ), "error in generating latin hypercube samples"

        return best_samples
