import numpy as np
from scipy.spatial.distance import cdist
from inspect import signature
from mogp_emulator.ExperimentalDesign import ExperimentalDesign
from mogp_emulator.GaussianProcess import GaussianProcess
from mogp_emulator.fitting import fit_GP_MAP
from numpy.linalg import LinAlgError

class SequentialDesign(object):
    """
    Base class representing a sequential experimental design

    This class provides the base implementation of a class for designing experiments sequentially. This
    means that rather than picking all simulation points in a single step, the points are selected one
    by one, taking into account the information obtained by determining the true parameter value at each
    design point when selecting the next one. Sequential designs can be very useful when running expensive,
    high-dimensional simulations to ensure that a limited computational budget is used effectvely.

    Instead of choosing all points at once, which is the case in a one-shot design, a sequential design
    does some additional computation work at each step to more carefully choose the next point. This means
    that sequential designs are better suited for very expensive simulations, where the additional
    cost of choosing the next point is small compared to the overall computational cost of running
    the simulations.

    A sequential design is built on top of a base design (which must be a subclass of the
    ``ExperimentalDesign`` class. In addition to the base design, the class must contain information on
    how many points are used in the initial design (i.e. the number of starting points used before starting
    the sequential steps in the design) and the number of candidate points that are considered during each
    iteration. Optionally, a function for evaluating the actual simulation can be optionally bound to the
    class instance, which allows the entire design process to be automated. If such a function is not
    provided, then the steps to run the design must be carried out manually, with the evaluated
    simulation values provided to the class at the end of each simulation in order to determine the
    next point.

    To use the base class to create an experimental design, a new subclass must be created that provides
    a method ``_eval_metric``, which considers all candidate points and returns the index of the best
    candidate. Otherwise, all other code provided here allows for a generic sequential design to be
    easily run and managed.
    """
    def __init__(self, base_design, f = None, n_samples = None, n_init = 10, n_cand = 50):
        """
        Create a new instance of a sequential experimental design

        Creates a new instance of a sequential experimental design, which sequentially chooses
        points to be evaluated from a complex simulation function. It is often used for
        expensive computational models, where the cost of running a single evaluation is
        large and must be done in series due to computational limitations, and thus some
        additional computation done at each step to select new points is small compared
        to the overall cost of running a single simulation.

        Sequential designs require specifying a base design using a subclass of ``ExperimentalDesign``
        as well as information on the number of points to use in each step in the design
        process. Additionally, the function to evaluated can be bound to the class to allow
        automatic evaluation of the function at each step.

        :param base_design: Base one-shot experimental design (must be a subclass of
                            ``ExperimentalDesign``). This contains the information on the
                            parameter space to be sampled.
        :type base_design: ExperimentalDesign
        :param f: Function to be evaluated for the design. Must take all parameter values as a single
                  input array and return a single float or an array of length 1
        :type f: function or other callable
        :param n_samples: Number of sequential design points to be drawn. If specified, this must be
                          a non-negative integer. Note that this is in addition to the number of initial
                          points, meaning that the total design size will be ``n_samples + n_init``.
                          This can also be specified when running the full design. This parameter is
                          optional, and defaults to ``None`` (meaning the number of samples is set when
                          running the design, or that samples will be added manually).
        :type n_samples: int or None
        :param n_init: Number of points in the inital design before the sequential steps begin. Must
                       be a positive integer. Optional, default value is 10.
        :type n_init: int
        :param n_cand: Number of candidates to consider at each sequential design step. Must be a positive
                       integer. Optional, default value is 50.
        """

        if not isinstance(base_design, ExperimentalDesign):
            raise TypeError("base design must be a one-shot experimental design")

        if not f is None:
            if not callable(f):
                raise TypeError("simulator f must be a function or other callable")

            if not len(signature(f).parameters) == 1:
                raise ValueError("simulator f must accept all parameters as a single input array")

        if (not n_samples is None) and int(n_samples) < 0:
            raise ValueError("number of samples must be nonzero")

        if int(n_init) <= 0:
            raise ValueError("number of initial design points must be positive")

        if int(n_cand) <= 0:
            raise ValueError("number of candidate design points must be positive")

        self.base_design = base_design
        self.f = f
        if n_samples is None:
            self.n_samples = None
        else:
            self.n_samples = int(n_samples)
        self.n_init = int(n_init)
        self.n_cand = int(n_cand)

        self.current_iteration = 0
        self.initialized = False
        self.inputs = None
        self.targets = None
        self.candidates = None

    def save_design(self, filename):
        """
        Save current state of the sequential design

        Saves the current state of the sequential design by writing the current
        values of ``inputs``, ``targets``, and ``candidates`` to file as a ``.npz``
        file. To re-load a saved design, use the ``load_design`` method.

        Note that this method only dumps the arrays holding the inputs, targets, and
        candidates to a ``.npz`` file. It does not ensure that the function or base
        design are consistent, so it is up to the user to ensure that the new design
        parameters are the same as the parameters for the old one.

        :param filename: Filename or file object where design will be saved
        :type filename: str or file
        :returns: None
        """

        design_dict = {}
        design_dict['inputs'] = self.inputs
        design_dict['targets'] = self.targets
        design_dict['candidates'] = self.candidates

        np.savez(filename, **design_dict)

    def load_design(self, filename):
        """
        Load previously saved sequential design

        Loads a previously saved sequential design from file. Loads the arrays for
        ``inputs``, ``targets``, and ``candidates`` from file and sets other internal
        data to be consistent. It performs a few checks for consistency to ensure
        that the loaded design is compatible with the selected parameters, however,
        it does not completely check everything for consistency (in particular, it does
        not make any attempt to ensure that the exact base design or function are
        identical to what was previously used). It is up to the user to ensure that
        these are consistent with the previous instance of the design.

        :param filename: Filename or file object from which the design will be loaded
        :type filename: str or file
        :returns: None
        """

        design_file = np.load(filename, allow_pickle=True)

        self.inputs = np.array(design_file['inputs'])
        if np.all(self.inputs) == None:
            self.inputs = None

        self.targets = np.array(design_file['targets'])
        if np.all(self.targets) == None:
            self.targets = None

        self.candidates = np.array(design_file['candidates'])
        if np.all(self.candidates) == None:
            self.candidates = None

        # perform some checks (note this is not exhaustive)

        if self.inputs is None:
            assert self.targets is None, "Cannot have targets without corresponding inputs"
        else:
            if not self.targets is None:
                assert self.targets.ndim == 1, "bad number of dimensions for targets"
                assert self.targets.shape[0] <= self.inputs.shape[0], "targets cannot be longer than inputs"
                self.initialized = True
                self.current_iteration = self.targets.shape[0]
            assert self.get_n_parameters() == self.inputs.shape[1], "Bad shape for inputs"
            if self.inputs.shape[1] < self.n_init:
                print("n_init greater than number of inputs, changing n_init")
                self.n_init = self.inputs.shape[1]

        if not self.candidates is None:
            assert self.get_n_parameters() == self.candidates.shape[1], "Bad shape for candidates"
            if self.candidates.shape[0] != self.n_cand:
                print("shape of candidates differs from n_cand, candidates will be overridden")

    def has_function(self):
        """
        Determines if class contains a function for running the simulator

        This method checks to see if a function has been provided for running the simulation.

        :returns: Whether or not the design has a bound function for evaluting the simulation.
        :rtype: bool
        """
        return (not self.f is None)

    def get_n_parameters(self):
        """
        Get number of parameters in design

        Returns the number of parameters in the design (note that this is specified in the base
        design that must be provided when initializing the class instance).

        :returns: Number of parameters in the design
        :rtype: int
        """

        return self.base_design.get_n_parameters()

    def get_n_init(self):
        """
        Get number of initial design points

        Returns the number of initial design points used before beginning the sequential design
        steps. Note that this means that the total number of samples to be drawn for the design
        is ``n_init + n_samples``.

        :returns: Number of initial design points
        :rtype: int
        """

        return self.n_init

    def get_n_samples(self):
        """
        Get number of sequential design points

        Returns the number of sequential design points used in the sequential design steps. This
        parameter can be ``None`` to indicate that the number of samples will be specified when
        running the design, or that the samples will be updated manually. Note that the total number
        of samples to be drawn for the design is ``n_init + n_samples``.

        :returns: Number of sequential design points
        :rtype: int
        """

        return self.n_samples

    def get_n_cand(self):
        """
        Get number of candidate design points

        Returns the number of candidate design points used in each sequential design step. Candidates
        are re-drawn at each step, so this number of points will be drawn each time and all points
        will be considered at each iteration.

        :returns: Number of candidate design points
        :rtype: int
        """

        return self.n_cand

    def get_current_iteration(self):
        """
        Get number of current iteration in the experimental design

        Returns the current iteration during the sequential design process. This is mostly useful
        if the sequential design is being updated manually to know the current iteration.

        :returns: Current iteration number
        :rtype: int
        """

        return self.current_iteration

    def get_inputs(self):
        """
        Get current design input points

        Returns a numpy array holding the current design points. The array is 2D and has shape
        ``(current_iteration, n_parameters)`` (i.e. it is resized after each iteration when a new
        design point is chosen).

        :returns: Current value of the design inputs
        :rtype: ndarray
        """

        return self.inputs

    def get_targets(self):
        """
        Get current design target points

        Returns a numpy array holding the current target points. The array is 1D and has shape
        ``(current_iteration,)`` (i.e. it is resized after each iteration when a new target point
        is added). Note that simulation outputs must be a single number, so if considering a
        simulation has multiple outputs, the user must decide how to combine them to form the
        relevant target value for deciding which point to simulate next.

        :returns: Current value of the target inputs
        :rtype: ndarray
        """

        return self.targets

    def get_candidates(self):
        """
        Get current candidate design input points

        Returns a numpy array holding the current candidate design points. The array is 2D and
        has shape ``(n_cand, n_parameters)``. It always has the same size once it is initialized,
        but the values will change acros iterations as new candidate points are considered at
        each iteration.

        :returns: Current value of the candidate design inputs
        :rtype: ndarray
        """

        return self.candidates

    def get_base_design(self):
        """
        Get type of base design

        Returns the type of the base design. The base design must be a subclass of ``ExperimentalDesign``,
        but any one-shot design method can be used to generate the initial design and the candidates.

        :returns: Base design type as a string
        :rtype: str
        """

        return type(self.base_design).__name__

    def generate_initial_design(self):
        """
        Create initial design

        Method to set the initial design inputs. Generates the desired number of points for the initial
        design by drawing from the base design. Method sets the ``inputs`` attribute of the
        ``SequentialDesign`` instance, but also returns the initial design as a numpy array if the
        simulations are to be run manually. This method can be run repeatedly to draw different
        initial designs if the initial target values have not been set, but once the targets have been
        set the method will not overwrite them to prevent corruption of the design.

        :returns: Initial design points, a 2D numpy array with shape ``(n_init, n_parameters)``
        :rtype: ndarray
        """

        assert not self.initialized, "initial design has already been created"

        self.inputs = self.base_design.sample(self.n_init)
        self.current_iteration = self.n_init
        return self.inputs

    def set_initial_targets(self, targets):
        """
        Set initial design target values

        Method to set the initial design targets. Generates the desired number of points for the initial
        design by drawing from the base design. Method sets the ``inputs`` attribute of the
        ``SequentialDesign`` instance, but also returns the initial design as a numpy array if the
        simulations are to be run manually. This method can be run repeatedly to draw different
        initial designs if the initial target values have not been set, but once the targets have been
        set the method will not overwrite them to prevent corruption of the design.

        Target values must be an array with length ``(n_init,)``, with values obtained by running
        the initial design through the simulation. Note that this means the initial design must
        be created prior to running this method -- if this method is called prior to
        ``generate_initial_design``, the code will raise an error.

        :param targets: Initial value of targets, must be a 1D numpy array with shape ``(n_init,)``
        :type targets: ndarray
        :returns: None
        :rtype: None
        """

        if self.inputs is None:
            raise ValueError("Initial design has not been generated")
        else:
            assert self.inputs.shape == (self.n_init, self.get_n_parameters()), "inputs have not been initialized correctly"

        targets = np.atleast_1d(np.squeeze(np.array(targets)))
        assert np.array(targets).shape == (self.n_init,), "initial targets must have shape (n_init,)"

        self.targets = np.array(targets)
        self.initialized = True

    def run_initial_design(self):
        """
        Run initial design

        Method to run the initial design by generating the initial design, evaluating the function on
        all design points, and setting the target values. Note that this requires having a bound function
        to the class in order to evaluate the design points internally. It is a shortcut to running
        ``generate_initial_design``, evaluating the initial design points, and then using
        ``set_initial_targets`` to set the target values, with some additional checks along the way.

        If the initial design has already been fully run, this method will raise an error as the
        method to generate the initial design checks this prior to overwriting the initial targets.
        Note also that this method checks that the outputs of the bound function match up with
        the expected array sizes and that all outputs are finite before updating the initial targets.

        :returns: None
        :rtype: None
        """


        assert self.has_function(), "Design must have a bound function to use run_initial_design"

        inputs = self.generate_initial_design()
        targets = np.full((self.n_init,), np.nan)

        for i in range(self.n_init):
            targets[i] = np.array(self.f(inputs[i,:]))

        assert np.all(np.isfinite(targets)), "error in initializing sequential design, function outputs may not be the correct shape"
        self.set_initial_targets(targets)

    def _generate_candidates(self):
        """
        Generate candidates for next iteration

        Internal method for generating candidates for the next iteration of the sequential design.
        Draws the desired number of points from the base design and sets the internal ``candidates``
        attribute to the resuting candidate design points.

        :returns: None
        :rtype: None
        """

        self.candidates = self.base_design.sample(self.n_cand)

    def _eval_metric(self):
        """
        Evaluate metric for selecting next point

        Apply the metric used for sequential design to all candidate points and returns the index of the
        best candidate. This is not implemented for the base implementation, and is the only method
        that should need to be updated in order to create a new type of sequential design.

        :returns: Index of best candidate from the possible next design points. Must be an integer
                  0 <= index < n_cand
        :rtype: int
        """
        raise NotImplementedError("Base class for Sequential Design does not implement an evaluation metric")

    def _estimate_next_target(self, next_point):
        """
        Estimate value of simulator for a point in a Sequential design

        This method is used for the batch version of a sequential design. Instead of updating
        the targets with the known solution, this method is used to estimate the function
        instead. This is method-specific, so this is not defined for the base class but instead
        should be defined in the subclass. Returns an array of length 1 holding the prediction.

        :param next_point: Input to be simulated. Must be an array of shape ``(n_parameters,)``
        :type next_point: ndarray
        :returns: Estimated simulation value for the given input as an array of length 1
        :rtype: ndarray
        """
        raise NotImplementedError("_estimate_next_point not implemented for base SequentialDesign")

    def get_batch_points(self, n_points):
        """
        Batch version of get_next_point for a Sequential Design

        This method returns a batch of design points to run from a Sequential Design. This is
        useful if simulations can be run in parallel, which speeds up the ability to
        generate designs efficiently. The method simply calls ``get_next_point`` the
        required number of times, but rather than using the true value of the simulation
        it instead substitutes the predicted value that is method-specific. This can be
        implemented in a subclass by defining the method ``_estimate_next_target``.

        :param n_points: Size of batch to generate for the next set of simulation points.
                         This parameter determines the shape of the output array. Must
                         be a positive integer.
        :type n_points: int
        :returns: Set of batch points chosen using the batch version of the design
                  as a numpy array with shape ``(n_points, n_parameters)``
        :rtype: ndarray
        """

        assert n_points > 0, "n_points must be positive"

        batch_points = np.zeros((n_points, self.get_n_parameters()))

        for i in range(n_points):
            batch_points[i] = self.get_next_point()
            next_target = self._estimate_next_target(batch_points[i])
            self.set_next_target(next_target)

        self.current_iteration = self.current_iteration - n_points
        new_targets = np.array(self.targets[:self.current_iteration])
        self.targets = np.array(new_targets)

        return batch_points

    def get_next_point(self):
        """
        Evaluate candidates to determine next point

        Public method for determining the next point in the design. Internally, it checks that the inputs
        and target arrays are as expected for correctly drawing a new point, generates prospective candidates,
        and then evaluates them using the desired metric in order to select the best one. It updates the
        ``inputs`` array and returns the next point to be evaluated as a 1D numpy array of length
        ``n_parameters``.

        :returns: Next design point, a 1D numpy array of length ``n_parameters``
        :rtype: ndarray
        """

        if self.inputs is None:
            raise ValueError("Initial design has not been generated")
        else:
            assert self.inputs.shape == (self.current_iteration, self.get_n_parameters()), "inputs have not been correctly updated"

        if self.targets is None:
            raise ValueError("Initial targets have not been generated")
        else:
            assert self.targets.shape == (self.current_iteration,), "targets have not been correctly updated"

        self._generate_candidates()
        next_index = self._eval_metric()

        new_inputs = np.empty((self.current_iteration + 1, self.get_n_parameters()))
        new_inputs[:-1, :] = self.inputs

        next_point = self.candidates[next_index,:]
        new_inputs[-1,:] = next_point

        self.inputs = np.array(new_inputs)

        return next_point

    def set_batch_targets(self, new_targets):
        """
        Batch version of set_next_target for a Sequential Design

        This method updates the targets array for a batch set of simulations. The input
        array must have shape ``(n_points,)``, where ``n_points`` is the number of points
        selected when calling ``get_batch_points``. Disagreement between these two values
        will result in an error.

        :param new_targets: Array holding results from the simulations. Must be an array
                            of shape ``(n_points,)``, where ``n_points`` is set when
                            calling ``get_batch_points``
        :type new_targets: ndarray
        :returns: None
        """
        if self.inputs is None:
            raise ValueError("Initial design has not been generated")
        else:
            n_points = self.inputs.shape[0] - self.current_iteration
            assert self.inputs.shape == (self.current_iteration + n_points, self.get_n_parameters()), "inputs have not been correctly updated"

        if self.targets is None:
            raise ValueError("Initial targets have not been generated")
        else:
            assert self.targets.shape == (self.current_iteration,), "targets have not been correctly updated"

        new_targets = np.atleast_1d(np.array(new_targets))
        new_targets = np.reshape(new_targets, (len(new_targets),))
        assert new_targets.shape == (n_points,), "new targets must have length n_points"

        updated_targets = np.empty((self.current_iteration + n_points,))
        updated_targets[:-n_points] = self.targets
        updated_targets[-n_points:] = np.array(new_targets)

        self.targets = np.array(updated_targets)
        self.current_iteration = self.current_iteration + n_points

    def set_next_target(self, target):
        """
        Set value of next target

        Updates the target array with the correct value (from running the actual simulation) of the
        latest design point determined using ``get_next_point``. The target input must be a float
        or an array of length 1. The code internally checks the inputs and targets for any problems
        that may have occurred in updating them correctly, and if all is well then updates the
        target array and increments the number of iterations. If the design has not been
        correctly initialized, or ``get_next_point`` has not been previously run, this method
        will raise an error.

        :param target: New target value found from evaluating the simulation on the latest design
                       point found from the ``get_next_point`` method.
        :type target: float or length 1 array
        :returns: None
        :rtype: None
        """

        if self.inputs is None:
            raise ValueError("Initial design has not been generated")
        else:
            assert self.inputs.shape == (self.current_iteration + 1, self.get_n_parameters()), "inputs have not been correctly updated"

        if self.targets is None:
            raise ValueError("Initial targets have not been generated")
        else:
            assert self.targets.shape == (self.current_iteration,), "targets have not been correctly updated"

        target = np.atleast_1d(np.array(target))
        target = np.reshape(target, (len(target),))
        assert target.shape == (1,), "new target must have length 1"

        new_targets = np.empty((self.current_iteration + 1,))
        new_targets[:-1] = self.targets
        new_targets[-1] = np.array(target)

        self.targets = np.array(new_targets)
        self.current_iteration = self.current_iteration + 1

    def run_next_point(self):
        """
        Perform one iteration of the sequential design process

        Method for performing an iteration of the sequential design process. This is a shortcut for
        generating and evaluating the candidates to find the best next design point, evaluating
        the function on the next point, and then updating the targets array with the value.
        This requires a function be bound to the class instance to automatically run the
        simulation. This will also automatically update the ``current_iteration`` attribute,
        which can be used to determine the number of sequential design steps that have been run.

        :returns: None
        :rtype: None
        """

        assert self.has_function(), "Design must have a bound function to use run_next_point"

        next_point = self.get_next_point()
        next_target = np.array(self.f(next_point))
        self.set_next_target(next_target)

    def run_sequential_design(self, n_samples = None):
        """
        Run the entire sequential design

        Method to run all steps of the sequential design process. Note that the class instance must
        have a bound function for evaluating the design points to run all steps automatically. If
        such a method is not provided, the design steps must be run manually.

        The desired number of samples to be drawn can either be specified when initializing the
        class instance or when calling this method. If a number of samples is provided on
        both occasions, then the number provided when calling ``run_sequential_design`` is used.

        Internally, this method is a wrapper to ``run_initial_design`` and then calling
        ``run_next_point`` a total of ``n_samples`` times. Note that this means that the total
        number of design points is ``n_init + n_samples``.

        :param n_samples: Number of sequential design steps to be run. Optional if the number was
                          specified upon initialization. Default is ``None`` (default to number
                          set when initializing). If numbers are provided on both occasions, the
                          number set here is used. If a number is provided, must be non-negative.
        :type n_samples: int or None
        :returns: None
        :rtype: None
        """

        assert self.has_function(), "Design must have a bound function to use run_sequential_design"

        if n_samples is None and self.n_samples is None:
            raise ValueError("must specify n_samples either when initializing or calling run_sequential_design")

        if n_samples is None:
            n_iter = self.n_samples
        else:
            n_iter = n_samples

        assert n_iter >= 0, "number of samples must be non-negative"

        self.run_initial_design()

        for i in range(n_iter):
            self.run_next_point()

    def __str__(self):
        """
        Returns string representation of a sequential design

        Returns a string representation of the sequential design. Contains information on the base
        design, the number of points used in the different steps, and the input and target values.

        :returns: String representation of the sequential design
        :rtype: str
        """

        output_string = ""
        output_string += type(self).__name__+" with\n"
        output_string += self.get_base_design()+" base design\n"
        if self.has_function():
            output_string += "a bound simulator function\n"
        output_string += str(self.get_n_samples())+" total samples\n"
        output_string += str(self.get_n_init())+" initial points\n"
        output_string += str(self.get_n_cand())+" candidate points\n"
        output_string += str(self.get_current_iteration())+" current samples\n"
        output_string += "current inputs: "+str(self.get_inputs())+"\n"
        output_string += "current targets: "+str(self.get_targets())

        return output_string


class MICEFastGP(GaussianProcess):
    """
    Derived GaussianProcess class implementing the Woodbury matrix identity for fast predictions

    This class implements a Gaussian Process that is used in the MICE Sequential Design. The GP
    is fit using all candidate points from the sequential design, and the uses the Woodbury
    matrix identity to correct that fit to exclude the candidate point in question. This reduces
    the cost of fitting the GP from O(n^3) to O(n^2), which can dramatically speed up this
    process for large numbers of candidate points. This is mostly used for the particular
    application to the MICE sequential design, but could potentially have other applications
    where many candidate points are to be considered one at a time.
    """
    def fast_predict(self, index):
        """
        Make a fast prediction using one input point to a fit GP

        This method is used to correct a Gaussian Process fit to a set of candidate points to
        evaluate the uncertainty at the candidate point. It is used in the MICE sequential
        design procedure to examine the mutual information between candidate points by determining
        how well correlated the design point is in question to the remainder of the candidates.
        It uses the Woodbury matrix identity to correct the existing GP fit (which requires
        O(n^3) operations) using O(n^2) operations, speeding up the process significantly for
        large candidate design sizes.

        The method requires a fit GP, and the index of the input point that is to be excluded.
        The method then corrects the GP fit and computes the uncertainty of the prediction
        on the excluded point returning the uncertainty as a float.

        :param index: Index of input point to be excluded in the fit and to which the prediction
                      will be applied. Must be an integer with 0 <= index < n (where n is the number
                      of target points in the fit GP, or the number of candidate points when
                      applied to the MICE procedure).
        :type index: int
        :returns: Uncertainty in the corrected fit applied to the given index point
        :rtype: float
        """

        index = int(index)
        assert index >= 0 and index < self.n, "index must be 0 <= index < n"

        indices = (np.arange(self.n) != index)

        switch = self.theta.n_mean
        sigma_2 = self.theta.cov + self.theta.nugget

        Ktest = self.theta.cov*self.kernel.kernel_f(np.reshape(self.inputs[indices,:], (self.n - 1, self.D)),
                                                    np.reshape(self.inputs[index, :], (1, self.D)),
                                                    self.theta.corr_raw)

        invQ = np.linalg.solve(self.L.T, np.linalg.solve(self.L, np.eye(self.n)))
        invQ_mod = (invQ[indices][:, indices] -
                    1./invQ[index, index]*np.outer(invQ[indices, index], invQ[indices, index]))

        var = np.maximum(sigma_2 - np.sum(Ktest * np.dot(invQ_mod, Ktest), axis=0), 0.)

        return var

class MICEDesign(SequentialDesign):
    """
    Class representing a Mutual Information for Computer Experiments (MICE) sequential
    experimental design

    This class provides an implementation of the MICE algorithm, which uses Mutual Information
    as the criterion for selecting new points in a sequential design. The idea in MICE is to
    select design points based on the point that provides the most information on the function
    values in the entire design space. This is a straightforward application of a sequential
    design procedure, though the class requires a few additional parameters in order to
    compute the MICE criteria.

    These additional parameters are nugget parameters provided to the Gaussian Process fit to
    smooth the predictions when evaluating the Mutual Information criteria. Essentially, since
    experimental design often requires sampling from a high dimensional space, this cannot be
    done in a way that guarantees that all candidate points are equally spaced. The Mutual
    Information criterion is sensitive to how these candidate points are distributed in space,
    so the nugget parameter provides some smoothing that makes the criterion less dependent on
    the distribution of the candidate points. Typical values of the smoothing nugget parameters
    (``nugget_s`` in this implementation) are 1, though this may depend on the application.

    Other than the smoothing parameters, the implementation follows the base procedure for a
    sequential design. The implementation adds methods for querying the nugget parameters
    and an additional helper function for computing the Mutual Information criterion, but
    other methods are identical.
    """
    def __init__(self, base_design, f = None, n_samples = None, n_init = 10, n_cand = 50,
                 nugget = "adaptive", nugget_s = 1.):
        """
        Create new instance of a MICE sequential design

        Method to initialize a new MICE design. Parameters are largely the same as for the base
        ``SequentialDesign`` class, with a few additional nugget parameters for computing the
        Mutual Information criterion. A base design must be provided (must be a subclass of the
        ``ExperimentalDesign`` class), plus optionally a function to be evaluated in the design.
        Additional parameters include the number of samples, the number of initial design points,
        the number of candidate points, the nugget parameter for the base GP, and the smoothing
        nugget parameter for smoothing the uncertainty predictions on the candidate design points.
        Note that the total number of design points is ``n_init + n_samples``.

        :param base_design: Base one-shot experimental design (must be a subclass of
                            ``ExperimentalDesign``). This contains the information on the
                            parameter space to be sampled.
        :type base_design: ExperimentalDesign
        :param f: Function to be evaluated for the design. Must take all parameter values as a single
                  input array and return a single float or an array of length 1
        :type f: function or other callable
        :param n_samples: Number of sequential design points to be drawn. If specified, this must be
                          a positive integer. Note that this is in addition to the number of initial
                          points, meaning that the total design size will be ``n_samples + n_init``.
                          This can also be specified when running the full design. This parameter is
                          optional, and defaults to ``None`` (meaning the number of samples is set when
                          running the design, or that samples will be added manually).
        :type n_samples: int or None
        :param n_init: Number of points in the inital design before the sequential steps begin. Must
                       be a positive integer. Optional, default value is 10.
        :type n_init: int
        :param n_cand: Number of candidates to consider at each sequential design step. Must be a positive
                       integer. Optional, default value is 50.
        :param nugget: Nugget parameter for base GP predictions. Must be a non-negative float or ``None``,
                       where ``None`` indicates that the nugget parameter is selected adaptively. Optional,
                       default value is ``None``.
        :type nugget: float or None
        :param nugget_s: Smoothing nugget parameter for smoothing the predictions on the candidate space.
                         Must be a non-negative float. Default value is 1.
        :type nugget_s: float
        """

        if not isinstance(nugget, str):
            try:
                float(nugget)
            except TypeError:
                raise TypeError("nugget must be a string or convertible to a float")
            if nugget < 0.:
                raise ValueError("nugget parameter cannot be negative")

        if nugget_s < 0.:
            raise ValueError("nugget smoothing parameter cannot be negative")

        if isinstance(nugget, str):
            self.nugget = nugget
        else:
            self.nugget = float(nugget)
        self.nugget_s = float(nugget_s)

        super().__init__(base_design, f, n_samples, n_init, n_cand)

    def get_nugget(self):
        """
        Get value of nugget parameter for base GP

        Returns the nugget value for the base GP (used to actually fit the inputs to targets).
        Can be a float or None (meaning fitting will adaptively add noise to stabilize matrix
        inversion as needed).

        :returns: Nugget parameter, can be a float or None for adaptive noise addition.
        :rtype: float or None
        """
        return self.nugget

    def get_nugget_s(self):
        """
        Get value of smoothing nugget parameter

        Returns the value of the smoothing nugget parameter for the GP used to evaluate the mutual
        information criterion. This GP examines the correlation between a candidate design point and
        the other candidate points, which requires smoothing to ensure that the correlation measure is
        not biased by the distribution of the candidate points in space. This parameter must be a
        nonnegative float (typical values used are 1, though this may depend on the application).

        :returns: Nugget parameter for smoothing predictions from candidate points made on a candidate
                  point. Typical values are 1.
        :rtype: float
        """
        return self.nugget_s

    def _estimate_next_target(self, next_point):
        """
        Estimate value of simulator for a point in a MICE design

        This method is used for the batch version of a sequential design. Instead of updating
        the targets with the known solution, this method is used to estimate the function
        instead. For the MICEDesign, this is just the prediction of the current design GP
        for the point. Returns an array of length 1 holding the prediction.

        :param next_point: Input to be simulated. Must be an array of shape ``(n_parameters,)``
        :type next_point: ndarray
        :returns: Estimated simulation value for the given input as an array of length 1
        :rtype: ndarray
        """

        next_point = np.array(next_point)
        assert next_point.shape == (self.get_n_parameters(),), "bad shape for next_point"

        return self.gp.predict(next_point)[0]

    def _MICE_criterion(self, data_point):
        """
        Compute the MICE criterion for a single candidate point

        This internal method computes the MICE criterion for a single candidate point. Requires
        input of the index of a candidate point to be considered (must be an integer satisfying
        ``0 <= index < n_cand``). It involves fitting a corrected GP to the candidate points other
        than the one under consideration (using the ``MICEFastGP`` class to correct the fit
        via the Woodbury matrix identity), and then computing the MICE criterion based on
        the predictions of the base GP on the point and the corrected GP fit. The MICE
        criterion is then the variance of the base GP divided by the variance of the corrected
        candidate GP. Value returned is the MICE criterion for the point in question.

        :param data_point: Index of the candidate point under consideration. Must be an integer
                           with ``0 <= index < n_cand``.
        :type data_point: int
        :returns: MICE criterion for the data point in question
        :rtype: float
        """

        data_point = int(data_point)

        assert data_point >= 0 and data_point < self.n_cand, "test point index is out of range"

        _, unc1, _ = self.gp.predict(self.candidates[data_point], unc=True, deriv=False)
        unc2 = self.gp_fast.fast_predict(data_point)

        mice_criter =  unc1/unc2

        assert np.isfinite(mice_criter), "error in computing MICE critera"

        return float(mice_criter)

    def _eval_metric(self):
        """
        Evaluate MICE criterion on all candidate points and select new design point

        This internal method computes the MICE criterion on all candidate points and returns
        the index of the point with the maximum value. It does so by first fitting a base GP
        to all points in the current design, and then fitting a dummy GP to all candidate
        design points using the parameter values determined from the base GP fit. The MICE
        criterion does not depend on the target values, since the parameters are determined
        via the base GP and the MICE criterion only depends on the uncertainty of the
        candidate GP (which is independent of the target values). These fit GPs are then used
        to compute the MICE criterion for each candidate point, and the method returns the
        index of the point that had the maximum value of the MICE criterion.

        :returns: Index of the candidate with the maximum MICE score (integer with
                  ``0 <= index < n_cand``)
        :rtype: int
        """

        numtries = 10

        for i in range(numtries):
            try:
                self.gp = GaussianProcess(self.inputs, self.targets, nugget=self.nugget)
                self.gp = fit_GP_MAP(self.gp)

                self.gp_fast = MICEFastGP(self.candidates, np.ones(self.n_cand), nugget=self.gp.theta.nugget*self.nugget_s)
                self.gp_fast.theta = self.gp.theta
                break
            except FloatingPointError:
                if i < numtries - 1:
                    continue
                else:
                    raise FloatingPointError("Unable to find parameters suitable for both GPs")
            except LinAlgError:
                if i < numtries - 1:
                    continue
                else:
                    raise LinAlgError("Unable to find parameters suitable for both GPs")

        results = []

        for point in range(self.n_cand):
            results.append(self._MICE_criterion(point))

        return np.argmax(results)
