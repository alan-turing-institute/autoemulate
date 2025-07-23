from abc import ABC, abstractmethod

import mogp_emulator
import torch

from autoemulate.experimental.types import TensorLike


class ExperimentalDesign(ABC):
    """
    Abstract class for experimental design.

    This class is used to sample points from the parameter space to be used in the
    training of an emulator.
    """

    @abstractmethod
    def __init__(self, bounds_list: list[tuple[float, float]]):
        """
        Initializes a Sampler object.

        Parameters
        ----------
        bounds_list : list
            List tuples with two numeric values. Each tuple corresponds to the lower and
            upper bounds of a parameter.
        """

    @abstractmethod
    def sample(self, n: int) -> TensorLike:
        """
        Samples n points from the sample space.

        Parameters
        ----------
        n: int
            The number of points to sample.

        Returns
        -------
        TensorLike
            A tensor of shape (n, dim) containing the sampled points.
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
    LatinHypercube experimental design class.
    """

    def __init__(self, bounds_list: list[tuple[float, float]]):
        """Initializes a LatinHypercube object."""
        self.sampler = mogp_emulator.LatinHypercubeDesign(bounds_list)

    def sample(self, n: int) -> TensorLike:
        sample_array = self.sampler.sample(n)
        return torch.tensor(sample_array, dtype=torch.float32)

    def get_n_parameters(self) -> int:
        return self.sampler.get_n_parameters()
