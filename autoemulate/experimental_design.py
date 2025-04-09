from abc import ABC
from abc import abstractmethod

import mogp_emulator


class ExperimentalDesign(ABC):
    """
    Abstract class for experimental design.

    This class is used to sample points from the parameter space
    to be used in the training of the emulator.

    Attributes
    ----------
    bounds_list : list
        List tuples with two numeric values.
        Each tuple corresponds to the lower and
        upper bounds of a parameter.
    """

    def __init__(self, bounds_list):
        """Initializes a Sampler object."""
        pass

    @abstractmethod
    def sample(self, n: int):
        """Samples n points from the sample space.

        Parameters
        ----------
        n : int
            The number of points to sample.
        """
        pass

    @abstractmethod
    def get_n_parameters(self):
        """Returns the number of parameters in the sample space.

        Returns
        -------
        n_parameters : int
            The number of parameters in the sample space.
        """
        pass


class LatinHypercube(ExperimentalDesign):
    """
    LatinHypercube class for experimental design.

    This class is used to sample points from the parameter space
    to be used in the training of the emulator.

    Attributes
    ----------
    bounds_list : list
        List tuples with two numeric values.
        Each tuple corresponds to the lower and
        upper bounds of a parameter.
    """

    def __init__(self, bounds_list):
        """Initializes a LatinHypercube object."""
        self.sampler = mogp_emulator.LatinHypercubeDesign(bounds_list)

    def sample(self, n: int):
        """Samples n points from the sample space.

        Parameters
        ----------
        n : int
            The number of points to sample.

        Returns
        -------
        samples : numpy.ndarray
            A numpy array of shape (n, dim) containing the sampled points.
        """
        return self.sampler.sample(n)

    def get_n_parameters(self):
        """Returns the number of parameters in the sample space.

        Returns
        -------
        n_parameters : int
            The number of parameters in the sample space.
        """
        return self.sampler.get_n_parameters()
