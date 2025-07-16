from abc import ABC, abstractmethod

import mogp_emulator
import torch

from autoemulate.experimental.types import TensorLike


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

    @abstractmethod
    def __init__(self, bounds_list: list[tuple[float, float]]):
        """Initializes a Sampler object."""

    @abstractmethod
    def sample(self, n: int) -> TensorLike:
        """
        Samples n points from the sample space.

        Parameters
        ----------
        n: int
            The number of points to sample.
        """

    @abstractmethod
    def get_n_parameters(self) -> int:
        """
        Returns the number of parameters in the sample space.

        Returns
        -------
        int
            The number of parameters in the sample space.
        """


class LatinHypercube(ExperimentalDesign):
    """
    LatinHypercube class for experimental design.

    This class is used to sample points from the parameter space
    to be used in the training of the emulator.

    Attributes
    ----------
    bounds_list: list
        List tuples with two numeric values.
        Each tuple corresponds to the lower and
        upper bounds of a parameter.
    """

    def __init__(self, bounds_list: list[tuple[float, float]]):
        """Initializes a LatinHypercube object."""
        self.sampler = mogp_emulator.LatinHypercubeDesign(bounds_list)

    def sample(self, n: int) -> TensorLike:
        """
        Samples n points from the sample space.

        Parameters
        ----------
        n: int
            The number of points to sample.

        Returns
        -------
        samples : numpy.ndarray
            A numpy array of shape (n, dim) containing the sampled points.
        """
        sample_array = self.sampler.sample(n)
        return torch.tensor(sample_array, dtype=torch.float32)

    def get_n_parameters(self) -> int:
        """Returns the number of parameters in the sample space.

        Returns
        -------
        int
            The number of parameters in the sample space.
        """
        return self.sampler.get_n_parameters()
