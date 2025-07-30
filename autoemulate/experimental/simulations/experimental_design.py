from abc import ABC, abstractmethod

import mogp_emulator
import torch

from autoemulate.experimental.core.types import TensorLike


class ExperimentalDesign(ABC):
    """
    Abstract class for experimental design.

    This class is used to sample points from the parameter space to be used in the
    training of an emulator.
    """

    @abstractmethod
    def __init__(self, bounds_list: list[tuple[float, float]]):
        """
        Initialize a Sampler object.

        Parameters
        ----------
        bounds_list: list
            List tuples with two numeric values. Each tuple corresponds to the lower and
            upper bounds of a parameter.
        """

    @abstractmethod
    def sample(self, n: int) -> TensorLike:
        """
        Sample n points from the sample space.

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
        Return the number of parameters in the sample space.

        Returns
        -------
        int
            The number of parameters in the sample space.
        """


class LatinHypercube(ExperimentalDesign):
    """LatinHypercube experimental design class."""

    def __init__(self, bounds_list: list[tuple[float, float]]):
        """Initialize a LatinHypercube object."""
        self.bounds_list = bounds_list

        # Partition indices and bounds into differing and constant parameters
        self.diff_idxs = []
        self.same_idxs = []
        bounds_list_diff = []
        for idx, bounds in enumerate(bounds_list):
            if bounds[0] != bounds[1]:
                self.diff_idxs.append(idx)
                bounds_list_diff.append(bounds)
            else:
                self.same_idxs.append(idx)

        self.sampler = mogp_emulator.LatinHypercubeDesign(bounds_list_diff)

    def sample(self, n: int) -> TensorLike:
        """
        Sample n points from the parameter space.

        Parameters
        ----------
        n: int
            The number of points to sample.

        Returns
        -------
        TensorLike
            A tensor of shape (n, dim) containing the sampled points.
        """
        # Sample only from parameters that differ
        sample_array_diff = self.sampler.sample(n)

        # Create full sample array filling in constant parameters
        sample_array_full = torch.zeros((n, len(self.bounds_list)), dtype=torch.float32)
        sample_array_full[:, self.diff_idxs] = torch.tensor(
            sample_array_diff, dtype=torch.float32
        )
        for idx in self.same_idxs:
            sample_array_full[:, idx] = self.bounds_list[idx][0]

        return sample_array_full

    def get_n_parameters(self) -> int:
        """
        Return the number of parameters in the sample space.

        Returns
        -------
        int
            The number of parameters in the sample space.
        """
        return len(self.bounds_list)
