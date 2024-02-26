from abc import ABC
from abc import abstractmethod

import mogp_emulator
import numpy as np

from .types import List


class ExperimentalDesign(ABC):
    def __init__(self, bounds_list: List[tuple[float, float]]):
        """Initializes a Sampler object.

        Parameters
        ----------
        bounds_list : list
            List tuples with two numeric values.
            Each tuple corresponds to the lower and
            upper bounds of a parameter.
        """
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
    def get_n_parameters(self) -> int:
        """Returns the number of parameters in the sample space.

        Returns
        -------
        n_parameters : int
            The number of parameters in the sample space.
        """
        pass


class LatinHypercube(ExperimentalDesign):
    def __init__(self, bounds_list: List[tuple[float, float]]):
        """Initializes a LatinHypercube object.

        Parameters
        ----------
        bounds_list : list
            List tuples with two numeric values.
            Each tuple corresponds to the lower and
            upper bounds of a parameter.
        """
        self.sampler = mogp_emulator.LatinHypercubeDesign(bounds_list)

    def sample(self, n: int) -> np.ndarray:
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

    def get_n_parameters(self) -> int:
        """Returns the number of parameters in the sample space.

        Returns
        -------
        n_parameters : int
            The number of parameters in the sample space.
        """
        return self.sampler.get_n_parameters()
